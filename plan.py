import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from thefuzz import fuzz
import re
import random

# --- CONFIGURATION ---
st.set_page_config(page_title="Planification Soutenances v2", layout="wide", page_icon="🎓")

# --- STYLES CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    .step-container { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb; }
    .error-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 5px; border: 1px solid #f5c6cb; }
    </style>
""", unsafe_allow_html=True)

# --- GESTION DE L'ÉTAT (SESSION STATE) ---
DEFAULT_STATE = {
    "etape": 1,
    "etudiants": [],
    "co_jurys": [],
    "dates": [],
    "creneaux": {},
    "disponibilites": {},
    "planning": [],
    "nb_salles": 2,
    "duree": 50,
    "logs": []
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- FONCTIONS UTILITAIRES (LECTURE ROBUSTE) ---

def clean_text(text):
    """Nettoie les chaînes de caractères (accents, espaces)."""
    if not isinstance(text, str): return str(text)
    return text.strip()

def lire_csv_robuste(uploaded_file):
    """Tente de lire un CSV avec différents encodages et séparateurs."""
    content = uploaded_file.getvalue()
    encodings = ['utf-8', 'latin-1', 'cp1252']
    separators = [';', ',']
    
    for enc in encodings:
        try:
            decoded = content.decode(enc)
            for sep in separators:
                # Heuristique simple : si le séparateur est fréquent dans la 1ère ligne
                first_line = decoded.split('\n')[0]
                if first_line.count(sep) > 1:
                    return pd.read_csv(StringIO(decoded), sep=sep), None
        except:
            continue
    return None, "Impossible de lire le fichier (Format inconnu)."

def importer_etudiants(uploaded_file):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], error
    
    # Nettoyage des colonnes
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    # Mapping intelligent
    col_map = {}
    for col in df.columns:
        if "PRENOM" in col: col_map['prenom'] = col
        elif "NOM" in col and "ENSEIGNANT" not in col: col_map['nom'] = col
        elif "PAYS" in col and "SERVICE" in col: col_map['pays'] = col # Priorité Service Accueil
        elif "PAYS" in col and 'pays' not in col_map: col_map['pays'] = col
        elif "ENSEIGNANT" in col or "TUTEUR" in col: col_map['tuteur'] = col
    
    required = ['prenom', 'nom', 'tuteur'] # Pays optionnel
    if not all(k in col_map for k in required):
        return [], f"Colonnes manquantes. Trouvé : {list(col_map.keys())}"
    
    etudiants = []
    for _, row in df.iterrows():
        prenom = row.get(col_map.get('prenom'), '')
        nom = row.get(col_map.get('nom'), '')
        if pd.isna(prenom) or pd.isna(nom): continue
        
        etudiants.append({
            "Prénom": str(prenom).strip(),
            "Nom": str(nom).strip(),
            "Pays": str(row.get(col_map.get('pays'), 'Inconnu')).strip(),
            "Tuteur": str(row.get(col_map.get('tuteur'), '')).strip()
        })
    return etudiants, None

def importer_disponibilites(uploaded_file, tuteurs_connus, co_jurys_connus, horaires_config):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], [], [error]
    
    # Détection des colonnes de dates (ex: 26/01/2026 08:00)
    date_cols_map = {} # {NomColonneCSV : "Lundi 26/01/2026 | 08:00 - 08:50"}
    
    for col in df.columns:
        # Regex pour trouver JJ/MM/AAAA et HH:MM
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2})", str(col))
        if match:
            d_csv, h_csv = match.group(1), match.group(2)
            # Chercher correspondance dans notre config
            for jour_app, creneaux_app in horaires_config.items():
                if d_csv in jour_app:
                    for c in creneaux_app:
                        if c.startswith(h_csv):
                            date_cols_map[col] = f"{jour_app} | {c}"
                            break
    
    if not date_cols_map:
        return [], [], ["Aucune colonne de date valide trouvée. Vérifiez l'année (2026?) dans l'étape 5."]

    personnes_reconnues = set(tuteurs_connus + co_jurys_connus)
    dispos_data = {}
    logs = []
    
    col_nom = df.columns[0] # On suppose que le nom est en 1er
    
    for _, row in df.iterrows():
        nom_brut = str(row[col_nom]).strip()
        if not nom_brut: continue
        
        # Fuzzy Matching
        best_match, score = None, 0
        for p in personnes_reconnues:
            s = fuzz.token_sort_ratio(nom_brut.lower(), p.lower())
            if s > score: score, best_match = s, p
        
        final_name = None
        if score >= 75: final_name = best_match
        
        if final_name:
            if final_name not in dispos_data: dispos_data[final_name] = {}
            for col_csv, key_app in date_cols_map.items():
                try:
                    val = row[col_csv]
                    is_open = bool(int(float(val))) if not pd.isna(val) else False
                    dispos_data[final_name][key_app] = is_open
                except: pass
        else:
            logs.append(f"Ignoré : '{nom_brut}' (Match max {score}% avec '{best_match}')")
            
    return dispos_data, list(dispos_data.keys()), logs

# --- ALGORITHME (MOTEUR DE PLANIFICATION) ---

class SchedulerEngine:
    def __init__(self, etudiants, dates, nb_salles, duree, dispos, co_jurys_pool):
        self.etudiants = etudiants
        self.nb_salles = nb_salles
        self.duree = duree
        self.dispos = dispos
        self.dates = dates # Liste objets date
        self.co_jurys_pool = list(set(co_jurys_pool)) # Liste unique
        self.slots = self._generate_slots()
        
        # Stats pour équilibrage
        self.charge_tuteur = defaultdict(int)
        self.charge_cojury = defaultdict(int)
        
        # Identifier les tuteurs
        self.tuteurs_actifs = list(set(e['Tuteur'] for e in etudiants))
        # Le pool de co-jury inclut aussi les tuteurs (ils peuvent être co-jury des autres)
        self.all_possible_jurys = list(set(self.co_jurys_pool + self.tuteurs_actifs))

    def _generate_slots(self):
        """Génère tous les créneaux atomiques (Jour, Heure, Salle)."""
        slots = []
        slot_id = 0
        for d in self.dates:
            d_str = d.strftime("%A %d/%m/%Y")
            # Matin et Après-midi standards
            for period in [("08:00", "12:10"), ("14:00", "18:10")]:
                try:
                    start = datetime.combine(d, datetime.strptime(period[0], "%H:%M").time())
                    end = datetime.combine(d, datetime.strptime(period[1], "%H:%M").time())
                    curr = start
                    while curr + timedelta(minutes=self.duree) <= end:
                        fin = curr + timedelta(minutes=self.duree)
                        h_str = f"{curr.strftime('%H:%M')} - {fin.strftime('%H:%M')}"
                        key = f"{d_str} | {h_str}"
                        
                        for s in range(1, self.nb_salles + 1):
                            slots.append({
                                "id": slot_id,
                                "key": key, # Clé pour dict dispo
                                "jour": d_str,
                                "heure": h_str,
                                "salle": f"Salle {s}",
                                "start": curr,
                                "end": fin
                            })
                            slot_id += 1
                        curr = fin
                except: continue
        return slots

    def is_available(self, person, slot_key):
        """Vérifie la dispo (par défaut True si non renseigné, pour éviter les blocages, ou False si strict)."""
        # Ici, on suppose False si pas d'info, sauf si liste vide
        if person not in self.dispos: return False # Pas de données = Pas dispo (sécurité)
        return self.dispos[person].get(slot_key, False)

    def solve(self):
        """
        Algorithme Glouton avec Priorisation par Contraintes.
        1. Calcule la 'difficulté' de chaque étudiant (nb de créneaux possibles pour son tuteur).
        2. Trie les étudiants : les plus contraints en premier.
        3. Pour chaque étudiant, cherche le premier créneau où :
           - Salle libre
           - Tuteur libre
           - Un Co-jury existe (libre et besoin d'équilibrage)
        """
        planning = []
        unassigned = []
        
        # Suivi de l'occupation : slot_id -> bool
        occupied_slots = set()
        # Suivi des jurys occupés : slot_key -> set(noms)
        busy_jurys = defaultdict(set)
        
        # 1. Analyse des contraintes (Score de difficulté)
        student_queue = []
        for etu in self.etudiants:
            tuteur = etu['Tuteur']
            # Compte combien de slots le tuteur a de dispos (approx)
            nb_dispos = 0
            if tuteur in self.dispos:
                nb_dispos = sum(1 for v in self.dispos[tuteur].values() if v)
            else:
                nb_dispos = 0 # Tuteur sans aucune dispo saisie
            
            # Score faible = Très contraint (prioritaire)
            student_queue.append((nb_dispos, etu))
            
        # Trie : nb_dispos croissant (les plus chiants en premier)
        student_queue.sort(key=lambda x: x[0])
        
        # 2. Placement
        for _, etu in student_queue:
            placed = False
            tuteur = etu['Tuteur']
            
            # Mélanger les slots pour éviter de toujours remplir le Lundi 8h en premier
            # Mais on peut aussi trier chronologiquement. Random est mieux pour l'équilibrage salles.
            my_slots = self.slots.copy()
            # Optionnel : Trier par jour pour regrouper ? Non, restons simple.
            
            for slot in my_slots:
                # Vérifications de base
                if slot['id'] in occupied_slots: continue
                if tuteur in busy_jurys[slot['key']]: continue
                if not self.is_available(tuteur, slot['key']): continue
                
                # Chercher un co-jury
                candidates = []
                for cj in self.all_possible_jurys:
                    if cj == tuteur: continue
                    if cj in busy_jurys[slot['key']]: continue
                    if not self.is_available(cj, slot['key']): continue
                    candidates.append(cj)
                
                if not candidates: continue # Personne dispo pour co-jury ici
                
                # Choisir le MEILLEUR co-jury (celui qui a le moins travaillé pour l'instant)
                # Tri par : 1. Charge Co-jury, 2. Charge Totale
                candidates.sort(key=lambda x: (self.charge_cojury[x], self.charge_tuteur[x] + self.charge_cojury[x]))
                best_cojury = candidates[0]
                
                # Bingo !
                planning.append({
                    "Étudiant": f"{etu['Prénom']} {etu['Nom']}",
                    "Tuteur": tuteur,
                    "Co-jury": best_cojury,
                    "Jour": slot['jour'],
                    "Heure": slot['heure'],
                    "Salle": slot['salle'],
                    "Début": slot['start'],
                    "Fin": slot['end']
                })
                
                # Marquer comme occupé
                occupied_slots.add(slot['id'])
                busy_jurys[slot['key']].add(tuteur)
                busy_jurys[slot['key']].add(best_cojury)
                
                # Mise à jour stats
                self.charge_tuteur[tuteur] += 1
                self.charge_cojury[best_cojury] += 1
                
                placed = True
                break # Passons à l'étudiant suivant
            
            if not placed:
                unassigned.append(etu)
                
        return planning, unassigned

# --- INTERFACE UTILISATEUR (WIZARD) ---

# Sidebar : Navigation
with st.sidebar:
    st.header("🧭 Navigation")
    steps = {
        1: "1. Étudiants",
        2: "2. Paramètres",
        3: "3. Dates & Co-jurys",
        4: "4. Import Disponibilités",
        5: "5. Génération"
    }
    selected_step = st.radio("Aller à :", list(steps.keys()), format_func=lambda x: steps[x], index=st.session_state.etape -1)
    if selected_step != st.session_state.etape:
        st.session_state.etape = selected_step
        st.rerun()
    
    st.divider()
    st.markdown("### 📊 Données")
    st.write(f"👨‍🎓 Étudiants : {len(st.session_state.etudiants)}")
    st.write(f"📅 Jours : {len(st.session_state.dates)}")
    st.write(f"✅ Dispos reçues : {len(st.session_state.disponibilites)}")

# --- ETAPE 1 : ETUDIANTS ---
if st.session_state.etape == 1:
    st.title("📂 Étape 1 : Import des Étudiants")
    st.markdown("Importez le fichier CSV contenant la liste des étudiants et leur tuteur.")
    
    file = st.file_uploader("Fichier Étudiants (CSV)", type=['csv', 'xlsx'])
    if file:
        data, err = importer_etudiants(file)
        if err:
            st.error(err)
        else:
            st.session_state.etudiants = data
            st.success(f"{len(data)} étudiants chargés avec succès !")
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            
    col1, col2 = st.columns([1, 5])
    if st.session_state.etudiants:
        if col2.button("Suivant ➡️", type="primary"):
            st.session_state.etape = 2
            st.rerun()

# --- ETAPE 2 : PARAMETRES ---
elif st.session_state.etape == 2:
    st.title("⚙️ Étape 2 : Configuration Logistique")
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.nb_salles = st.number_input("Nombre de salles disponibles", 1, 10, st.session_state.nb_salles)
    with c2:
        st.session_state.duree = st.number_input("Durée soutenance (min)", 30, 120, st.session_state.duree)
        st.caption("Conseil : Pour votre CSV, mettez 50 minutes.")

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("⬅️ Retour"):
        st.session_state.etape = 1
        st.rerun()
    if c2.button("Suivant ➡️", type="primary"):
        st.session_state.etape = 3
        st.rerun()

# --- ETAPE 3 : DATES & CO-JURYS ---
elif st.session_state.etape == 3:
    st.title("📅 Étape 3 : Dates et Co-jurys")
    
    st.subheader("1. Dates des soutenances")
    st.info("⚠️ Important : Sélectionnez les jours exacts présents dans votre fichier de disponibilités (ex: 26/01/2026).")
    
    nb_jours = st.number_input("Combien de jours ?", 1, 10, max(1, len(st.session_state.dates)))
    
    dates_list = []
    cols = st.columns(4)
    for i in range(nb_jours):
        # Default logic to help user
        def_val = st.session_state.dates[i] if i < len(st.session_state.dates) else datetime(2026, 1, 26).date() + timedelta(days=i)
        d = cols[i%4].date_input(f"Jour {i+1}", def_val)
        dates_list.append(d)
    
    st.subheader("2. Co-jurys supplémentaires")
    st.markdown("Les tuteurs sont automatiquement considérés comme co-jurys potentiels. Ajoutez ici des personnes externes si besoin.")
    
    new_cj = st.text_input("Ajouter un nom (Entrée pour valider)")
    if new_cj:
        if new_cj not in st.session_state.co_jurys:
            st.session_state.co_jurys.append(new_cj)
            st.rerun()
            
    if st.session_state.co_jurys:
        st.write(f"Co-jurys externes : {', '.join(st.session_state.co_jurys)}")
        if st.button("Effacer liste"): 
            st.session_state.co_jurys = []
            st.rerun()

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("⬅️ Retour"):
        st.session_state.etape = 2
        st.rerun()
    if c2.button("Suivant ➡️", type="primary"):
        st.session_state.dates = dates_list
        st.session_state.etape = 4
        st.rerun()

# --- ETAPE 4 : DISPONIBILITES ---
elif st.session_state.etape == 4:
    st.title("🗓️ Étape 4 : Import des Disponibilités")
    
    # Génération temporaire des créneaux théoriques pour le mapping
    temp_engine = SchedulerEngine([], st.session_state.dates, 1, st.session_state.duree, {}, [])
    slots_theoriques = defaultdict(list)
    for s in temp_engine.slots:
        key_parts = s['key'].split(" | ") # Jour | Heure
        slots_theoriques[key_parts[0]].append(key_parts[1])
    
    st.write(f"Créneaux théoriques générés : {len(temp_engine.slots)} sur {len(st.session_state.dates)} jours.")
    
    file = st.file_uploader("Fichier Disponibilités (CSV)", type=['csv'])
    if file:
        tuteurs = list(set(e['Tuteur'] for e in st.session_state.etudiants))
        dispos, found, logs = importer_disponibilites(file, tuteurs, st.session_state.co_jurys, slots_theoriques)
        
        if dispos:
            st.session_state.disponibilites = dispos
            st.success(f"✅ Disponibilités importées pour {len(dispos)} personnes.")
            
            with st.expander("🔍 Voir le détail des imports"):
                for l in logs: st.text(l)
                st.write("Personnes trouvées :", found)
        else:
            st.error("Échec de l'import.")
            for l in logs: st.error(l)

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("⬅️ Retour"):
        st.session_state.etape = 3
        st.rerun()
    if c2.button("Suivant ➡️", type="primary"):
        st.session_state.etape = 5
        st.rerun()

# --- ETAPE 5 : GENERATION ---
elif st.session_state.etape == 5:
    st.title("🚀 Étape 5 : Génération du Planning")
    
    # Diagnostic avant lancement
    tuteurs_requis = set(e['Tuteur'] for e in st.session_state.etudiants)
    tuteurs_sans_dispo = [t for t in tuteurs_requis if t not in st.session_state.disponibilites]
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Étudiants à placer : {len(st.session_state.etudiants)}")
    with col2:
        if tuteurs_sans_dispo:
            st.warning(f"⚠️ {len(tuteurs_sans_dispo)} tuteurs n'ont AUCUNE disponibilité (Risque d'échec).")
            with st.expander("Voir liste"): st.write(tuteurs_sans_dispo)
        else:
            st.success("Tous les tuteurs ont des disponibilités saisies.")

    if st.button("Lancer l'algorithme d'optimisation", type="primary", use_container_width=True):
        with st.spinner("Calcul du meilleur emploi du temps..."):
            engine = SchedulerEngine(
                st.session_state.etudiants,
                st.session_state.dates,
                st.session_state.nb_salles,
                st.session_state.duree,
                st.session_state.disponibilites,
                st.session_state.co_jurys
            )
            planning, failed = engine.solve()
            st.session_state.planning = planning
            st.session_state.failed = failed
            
    if st.session_state.planning:
        st.divider()
        st.success(f"🎉 Planning généré avec {len(st.session_state.planning)} soutenances planifiées !")
        
        if st.session_state.failed:
            st.error(f"❌ {len(st.session_state.failed)} étudiants n'ont pas pu être placés (conflits insolubles).")
            with st.expander("Voir les étudiants non placés"):
                st.dataframe(pd.DataFrame(st.session_state.failed))
        
        df_res = pd.DataFrame(st.session_state.planning)
        
        # Onglets de visualisation
        tab1, tab2, tab3 = st.tabs(["📋 Tableau", "📊 Gantt", "📥 Export"])
        
        with tab1:
            st.dataframe(df_res[['Jour', 'Heure', 'Salle', 'Étudiant', 'Tuteur', 'Co-jury']], use_container_width=True)
            
        with tab2:
            if not df_res.empty:
                df_res['Label'] = df_res['Étudiant'] + " (" + df_res['Salle'] + ")"
                fig = px.timeline(df_res, x_start="Début", x_end="Fin", y="Tuteur", color="Jour", hover_data=['Co-jury', 'Salle'])
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                
        with tab3:
            csv = df_res.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                "Télécharger le planning (CSV)",
                csv,
                "planning_final.csv",
                "text/csv",
                key='download-csv'
            )
