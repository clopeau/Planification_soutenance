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
st.set_page_config(page_title="Planification Soutenances v4 (Contiguïté)", layout="wide", page_icon="🎓")

# --- STYLES CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb; }
    .error-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 5px; border: 1px solid #f5c6cb; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALISATION STATE ---
DEFAULT_STATE = {
    "etape": 1,
    "etudiants": [],
    "co_jurys": [],
    "dates": [],
    "disponibilites": {},
    "planning": [],
    "nb_salles": 2,
    "duree": 50,
    "failed": []
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- FONCTIONS DE LECTURE ROBUSTE ---

def clean_str(val):
    """Nettoie une valeur (enlève nan, None, espaces)"""
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '']:
        return ""
    return str(val).strip()

def lire_csv_robuste(uploaded_file):
    """Tente de lire le CSV avec différents séparateurs et encodages."""
    uploaded_file.seek(0)
    content = uploaded_file.getvalue()
    encodings = ['utf-8', 'latin-1', 'cp1252']
    separators = [';', ','] 
    
    for enc in encodings:
        try:
            decoded = content.decode(enc)
            for sep in separators:
                if decoded.count(sep) > decoded.count('\n'):
                    return pd.read_csv(StringIO(decoded), sep=sep), None
        except: continue
    
    return None, "Format de fichier non reconnu (essayez UTF-8 ou Latin-1, séparateur ; ou ,)"

def importer_etudiants(uploaded_file):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], error
    
    raw_cols = list(df.columns)
    cols_normalized = {c: c.upper().replace('É','E').replace('È','E').replace('Ê','E') for c in raw_cols}
    col_map = {}
    
    for c_up, c_real in cols_normalized.items():
        if "PRENOM" in c_up and "NOM" not in c_up: col_map['prenom'] = c_real; break
    for c_up, c_real in cols_normalized.items():
        if "NOM" in c_up and "PRENOM" not in c_up and "REFERENT" not in c_up and "ENSEIGNANT" not in c_up: col_map['nom'] = c_real; break
    for c_up, c_real in cols_normalized.items():
        if "REFERENT" in c_up or "ENSEIGNANT" in c_up or "TUTEUR" in c_up: col_map['tuteur'] = c_real; break
    for c_up, c_real in cols_normalized.items():
        if "SERVICE" in c_up and "PAYS" in c_up: col_map['pays'] = c_real; break
    if 'pays' not in col_map:
        for c_up, c_real in cols_normalized.items():
            if "PAYS" in c_up: col_map['pays'] = c_real; break

    required = ['nom', 'tuteur']
    missing = [r for r in required if r not in col_map]
    if missing: return [], f"Colonnes introuvables : {missing}. Colonnes lues : {raw_cols}"

    etudiants = []
    for _, row in df.iterrows():
        nom = clean_str(row.get(col_map.get('nom')))
        prenom = clean_str(row.get(col_map.get('prenom'), ''))
        tuteur = clean_str(row.get(col_map.get('tuteur')))
        pays = clean_str(row.get(col_map.get('pays'), ''))
        
        if nom and tuteur:
            etudiants.append({"Prénom": prenom, "Nom": nom, "Pays": pays, "Tuteur": tuteur})
            
    return etudiants, None

def importer_disponibilites(uploaded_file, tuteurs_connus, co_jurys_connus, horaires_config):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], [], [error]
    
    personnes_reconnues = {p for p in (tuteurs_connus + co_jurys_connus) if p and str(p).lower() != 'nan'}
    if not personnes_reconnues: return {}, [], ["Aucun tuteur valide n'a été trouvé à l'étape 1."]

    date_cols_map = {} 
    for col in df.columns:
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2})", str(col))
        if match:
            d_csv, h_csv = match.group(1), match.group(2)
            for jour_app, creneaux_app in horaires_config.items():
                if d_csv in jour_app:
                    for c in creneaux_app:
                        if c.startswith(h_csv):
                            date_cols_map[col] = f"{jour_app} | {c}"
                            break
    
    if not date_cols_map: return {}, [], ["Aucune colonne de date (ex: 26/01/2026) reconnue. Vérifiez l'année à l'étape 5."]

    dispos_data = {}
    logs = []
    treated = set()
    col_nom = df.columns[0]
    
    for _, row in df.iterrows():
        nom_brut = clean_str(row[col_nom])
        if not nom_brut: continue
        best_match, best_score = None, 0
        for p in personnes_reconnues:
            score = fuzz.token_sort_ratio(nom_brut.lower(), p.lower())
            if score > best_score: best_score, best_match = score, p
        
        if best_score >= 70:
            final_name = best_match
            if final_name not in dispos_data: dispos_data[final_name] = {}
            for col_csv, key_app in date_cols_map.items():
                val = row.get(col_csv, 0)
                try:
                    is_open = bool(int(float(val))) if pd.notna(val) else False
                    dispos_data[final_name][key_app] = is_open
                except: pass
            treated.add(final_name)
        else:
            logs.append(f"⚠️ Ignoré : '{nom_brut}' (Meilleur candidat : '{best_match}' à {best_score}%)")
            
    return dispos_data, list(treated), logs

# --- MOTEUR DE PLANIFICATION (GLOUTON + CONTIGUÏTÉ) ---

class SchedulerEngine:
    def __init__(self, etudiants, dates, nb_salles, duree, dispos, co_jurys_pool):
        self.etudiants = etudiants
        self.nb_salles = nb_salles
        self.duree = duree
        self.dispos = dispos
        self.dates = dates
        self.co_jurys_pool = list(set(co_jurys_pool))
        self.slots = self._generate_slots()
        
        # Statistiques de charge
        self.charge_tuteur = defaultdict(int)
        self.charge_cojury = defaultdict(int)
        
        # Suivi précis pour la contiguïté (Qui est occupé quand ?)
        # Format: self.jury_occupied_times[NomJury] = set(datetime_debut)
        self.jury_occupied_times = defaultdict(set)
        # Format: self.jury_occupied_days[NomJury] = set(jour_str)
        self.jury_occupied_days = defaultdict(set)
        
        self.tuteurs_actifs = list(set(e['Tuteur'] for e in etudiants if e['Tuteur']))
        self.all_possible_jurys = list(set(self.co_jurys_pool + self.tuteurs_actifs))

    def _generate_slots(self):
        slots = []
        slot_id = 0
        for d in self.dates:
            d_str = d.strftime("%A %d/%m/%Y")
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
                                "id": slot_id, "key": key, "jour": d_str, "heure": h_str,
                                "salle": f"Salle {s}", "start": curr, "end": fin
                            })
                            slot_id += 1
                        curr = fin
                except: continue
        return slots

    def is_available(self, person, slot_key):
        if person not in self.dispos: return False
        return self.dispos[person].get(slot_key, False)

    def solve(self):
        planning = []
        unassigned = []
        occupied_slots = set() # Par ID de créneau (salle/heure)
        busy_jurys_at_slot = defaultdict(set) # Par clé temps (jour/heure)
        
        # 1. TRI DES ÉTUDIANTS (Les plus contraints en premier)
        student_queue = []
        for etu in self.etudiants:
            tut = etu['Tuteur']
            # Nb de créneaux dispos pour ce tuteur
            nb = sum(1 for v in self.dispos.get(tut, {}).values() if v)
            # On ajoute un petit random pour casser les égalités parfaites
            student_queue.append((nb, random.random(), etu))
        
        # Tri croissant (moins de dispos = prioritaire)
        student_queue.sort(key=lambda x: x[0])
        
        # 2. PLACEMENT
        for _, _, etu in student_queue:
            tuteur = etu['Tuteur']
            placed = False
            
            # --- ALGORITHME DE CONTIGUÏTÉ ---
            # On va trier les slots disponibles pour ce tuteur
            # Critères de tri :
            # 1. Adjacence immédiate (Tuteur déjà là juste avant ou juste après) -> +100 pts
            # 2. Même jour (Tuteur déjà là ce jour là) -> +10 pts
            # 3. Ordre chronologique (pour remplir le matin d'abord par défaut) -> -timestamp
            
            candidate_slots = []
            
            for slot in self.slots:
                # Filtres durs
                if slot['id'] in occupied_slots: continue
                if tuteur in busy_jurys_at_slot[slot['key']]: continue
                if not self.is_available(tuteur, slot['key']): continue
                
                # Calcul du score de contiguïté pour le TUTEUR
                score = 0
                t_start = slot['start']
                t_prev = t_start - timedelta(minutes=self.duree)
                t_next = slot['end'] # Car end = start + duree
                
                # Est-il occupé juste avant ?
                if t_prev in self.jury_occupied_times[tuteur]: score += 100
                # Est-il occupé juste après ?
                if t_next in self.jury_occupied_times[tuteur]: score += 100
                # Est-il occupé ce jour-là ?
                if slot['jour'] in self.jury_occupied_days[tuteur]: score += 10
                
                # Ajout à la liste des candidats
                candidate_slots.append((score, slot))
            
            # Tri des slots : Score décroissant (le plus haut en premier), puis chronologique
            # random.random() permet de varier les salles si égalité parfaite
            candidate_slots.sort(key=lambda x: (x[0], x[1]['start'].timestamp(), random.random()), reverse=True)
            
            # Essayer de placer dans le meilleur slot
            for score_slot, slot in candidate_slots:
                
                # Trouver un CO-JURY
                # On applique aussi la logique de contiguïté pour le co-jury !
                cj_candidates = []
                
                for cj in self.all_possible_jurys:
                    if cj == tuteur: continue
                    if cj in busy_jurys_at_slot[slot['key']]: continue
                    if not self.is_available(cj, slot['key']): continue
                    
                    # Score Co-jury
                    cj_score = 0
                    t_start = slot['start']
                    t_prev = t_start - timedelta(minutes=self.duree)
                    t_next = slot['end']
                    
                    if t_prev in self.jury_occupied_times[cj]: cj_score += 100
                    if t_next in self.jury_occupied_times[cj]: cj_score += 100
                    if slot['jour'] in self.jury_occupied_days[cj]: cj_score += 10
                    
                    # On veut :
                    # 1. Maximiser la contiguïté (cj_score haut)
                    # 2. Minimiser la charge de travail (charge_cojury bas)
                    cj_candidates.append((cj_score, self.charge_cojury[cj], cj))
                
                if not cj_candidates: continue
                
                # Tri des co-jurys : Score DESC, Charge ASC
                # Astuce python : on trie par (-score, charge) pour avoir score max et charge min
                cj_candidates.sort(key=lambda x: (-x[0], x[1]))
                
                best_cj = cj_candidates[0][2]
                
                # --- VALIDATION DU PLACEMENT ---
                planning.append({
                    "Étudiant": f"{etu['Prénom']} {etu['Nom']}",
                    "Tuteur": tuteur, "Co-jury": best_cj,
                    "Jour": slot['jour'], "Heure": slot['heure'],
                    "Salle": slot['salle'], "Début": slot['start'], "Fin": slot['end']
                })
                
                # Marquer tout occupé
                occupied_slots.add(slot['id'])
                busy_jurys_at_slot[slot['key']].add(tuteur)
                busy_jurys_at_slot[slot['key']].add(best_cj)
                
                # Mettre à jour les structures de contiguïté
                self.jury_occupied_times[tuteur].add(slot['start'])
                self.jury_occupied_times[best_cj].add(slot['start'])
                self.jury_occupied_days[tuteur].add(slot['jour'])
                self.jury_occupied_days[best_cj].add(slot['jour'])
                
                # Mettre à jour charges
                self.charge_tuteur[tuteur] += 1
                self.charge_cojury[best_cj] += 1
                
                placed = True
                break # Sortir de la boucle des slots, étudiant placé
            
            if not placed: unassigned.append(etu)
            
        return planning, unassigned

# --- INTERFACE ---

with st.sidebar:
    st.header("🧭 Navigation")
    steps = {1: "1. Étudiants", 2: "2. Paramètres", 3: "3. Dates", 4: "4. Import Dispos", 5: "5. Génération"}
    sel = st.radio("Aller à :", list(steps.keys()), format_func=lambda x: steps[x], index=st.session_state.etape -1)
    if sel != st.session_state.etape:
        st.session_state.etape = sel
        st.rerun()
    st.divider()
    st.write(f"Étudiants : {len(st.session_state.etudiants)}")
    st.write(f"Dispos Tuteurs : {len(st.session_state.disponibilites)}")

if st.session_state.etape == 1:
    st.title("1. Import des Étudiants")
    f = st.file_uploader("Fichier CSV (celui avec NOM, PRENOM, ENSEIGNANT REFERENT...)", type=['csv', 'xlsx'])
    if f:
        data, err = importer_etudiants(f)
        if err: st.error(err)
        else:
            st.session_state.etudiants = data
            st.success(f"{len(data)} étudiants importés.")
            st.dataframe(pd.DataFrame(data).head())
            if st.button("Suivant"): st.session_state.etape = 2; st.rerun()

elif st.session_state.etape == 2:
    st.title("2. Paramètres")
    c1, c2 = st.columns(2)
    st.session_state.nb_salles = c1.number_input("Salles", 1, 10, st.session_state.nb_salles)
    st.session_state.duree = c2.number_input("Durée (min)", 30, 120, st.session_state.duree)
    st.info("Mettez 50 min pour correspondre à votre CSV de disponibilités.")
    if st.button("Suivant"): st.session_state.etape = 3; st.rerun()

elif st.session_state.etape == 3:
    st.title("3. Dates")
    st.info("Sélectionnez les jours exacts de votre CSV (26, 27, 29 Janvier 2026).")
    nb = st.number_input("Nb Jours", 1, 5, max(3, len(st.session_state.dates)))
    ds = []
    cols = st.columns(4)
    for i in range(nb):
        # Aide à la saisie : initialise à 2026
        d_def = st.session_state.dates[i] if i < len(st.session_state.dates) else datetime(2026, 1, 26).date() + timedelta(days=i)
        ds.append(cols[i%4].date_input(f"Jour {i+1}", d_def))
    st.session_state.dates = ds
    
    st.subheader("Co-jurys supplémentaires (Optionnel)")
    txt = st.text_input("Nom du co-jury")
    if txt and txt not in st.session_state.co_jurys: st.session_state.co_jurys.append(txt)
    if st.session_state.co_jurys: st.write(st.session_state.co_jurys)
    
    if st.button("Suivant"): st.session_state.etape = 4; st.rerun()

elif st.session_state.etape == 4:
    st.title("4. Import Disponibilités")
    
    eng = SchedulerEngine([], st.session_state.dates, 1, st.session_state.duree, {}, [])
    mapping_config = defaultdict(list)
    for s in eng.slots:
        k = s['key'].split(" | ")
        mapping_config[k[0]].append(k[1])
    
    f = st.file_uploader("Fichier Disponibilités (CSV avec virgules ou points-virgules)", type=['csv'])
    if f:
        tuteurs_propres = [e['Tuteur'] for e in st.session_state.etudiants if e['Tuteur']]
        dispos, treated, logs = importer_disponibilites(f, tuteurs_propres, st.session_state.co_jurys, mapping_config)
        
        if dispos:
            st.session_state.disponibilites = dispos
            st.success(f"✅ {len(dispos)} personnes importées avec succès.")
            with st.expander("Voir les détails / Erreurs"):
                for l in logs: st.write(l)
                st.write("Reconnus :", treated)
        else:
            st.error("Aucune disponibilité valide trouvée.")
            for l in logs: st.error(l)
            
    if st.button("Suivant"): st.session_state.etape = 5; st.rerun()

elif st.session_state.etape == 5:
    st.title("5. Génération Optimisée")
    st.markdown("💡 L'algorithme va essayer de **grouper** les soutenances d'un même jury pour éviter les trous.")
    
    if st.button("Lancer la planification", type="primary"):
        eng = SchedulerEngine(
            st.session_state.etudiants, st.session_state.dates,
            st.session_state.nb_salles, st.session_state.duree,
            st.session_state.disponibilites, st.session_state.co_jurys
        )
        plan, fail = eng.solve()
        st.session_state.planning = plan
        st.session_state.failed = fail
        
    if st.session_state.planning:
        st.success(f"Planifié : {len(st.session_state.planning)} / Échecs : {len(st.session_state.failed)}")
        
        df = pd.DataFrame(st.session_state.planning)
        tab1, tab2 = st.tabs(["Tableau", "Gantt"])
        with tab1: st.dataframe(df)
        with tab2:
            if not df.empty:
                # Tri par salle et heure pour le Gantt
                df = df.sort_values(by=['Jour', 'Heure'])
                fig = px.timeline(df, x_start="Début", x_end="Fin", y="Tuteur", color="Jour", text="Étudiant")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
        
        csv = df.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("Télécharger CSV", csv, "planning.csv", "text/csv")
        
        if st.session_state.failed:
            st.error("Étudiants non placés :")
            st.dataframe(pd.DataFrame(st.session_state.failed))
