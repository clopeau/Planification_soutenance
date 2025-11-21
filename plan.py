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
import unicodedata

# --- CONFIGURATION ---
st.set_page_config(page_title="Planification Soutenances v8 (Global Search)", layout="wide", page_icon="🗓️")

# --- STYLES CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb; }
    .error-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 5px; border: 1px solid #f5c6cb; }
    </style>
""", unsafe_allow_html=True)

# --- STATE ---
DEFAULT_STATE = {
    "etape": 1, "etudiants": [], "co_jurys": [], "dates": [],
    "disponibilites": {}, "planning": [], "nb_salles": 2,
    "duree": 50, "failed": []
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state: st.session_state[key] = value

# --- HELPERS ---
def clean_str(val):
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '']: return ""
    return str(val).strip()

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    text = text.upper()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return text

def lire_csv_robuste(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.getvalue()
    encodings = ['utf-8', 'latin-1', 'cp1252']
    separators = [';', ','] 
    for enc in encodings:
        try:
            decoded = content.decode(enc)
            for sep in separators:
                if decoded.count(sep) > decoded.count('\n'):
                    return pd.read_csv(StringIO(decoded), sep=sep, engine='python'), None
        except: continue
    return None, "Format non reconnu."

# --- IMPORTERS (GARDÉS DE LA V5) ---
def importer_etudiants(uploaded_file):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], error
    
    raw_cols = list(df.columns)
    cols_map_normalized = {normalize_text(c): c for c in raw_cols}
    mapping = {}
    
    for c_norm, c_real in cols_map_normalized.items():
        if c_norm == "NOM": mapping['nom'] = c_real; break
    if 'nom' not in mapping:
        for c_norm, c_real in cols_map_normalized.items():
            if "NOM" in c_norm and "PRENOM" not in c_norm and "REFERENT" not in c_norm and "ACCUEIL" not in c_norm:
                mapping['nom'] = c_real; break

    for c_norm, c_real in cols_map_normalized.items():
        if "PRENOM" in c_norm and "NOM" not in c_norm: mapping['prenom'] = c_real; break
    if 'prenom' not in mapping:
         for c_norm, c_real in cols_map_normalized.items():
            if "PRENOM" in c_norm: mapping['prenom'] = c_real; break

    for c_norm, c_real in cols_map_normalized.items():
        if ("REFERENT" in c_norm or "ENSEIGNANT" in c_norm or "TUTEUR" in c_norm) and "ENTREPRISE" not in c_norm:
            mapping['tuteur'] = c_real; break
            
    for c_norm, c_real in cols_map_normalized.items():
        if "PAYS" in c_norm and "ACCUEIL" in c_norm: mapping['pays'] = c_real; break
    if 'pays' not in mapping:
        for c_norm, c_real in cols_map_normalized.items():
            if "PAYS" in c_norm: mapping['pays'] = c_real; break

    missing = [k for k in ['nom', 'tuteur'] if k not in mapping]
    if missing: return [], f"Colonnes introuvables : {missing}"

    etudiants = []
    for _, row in df.iterrows():
        nom = clean_str(row.get(mapping.get('nom')))
        prenom = clean_str(row.get(mapping.get('prenom'), ''))
        tuteur = clean_str(row.get(mapping.get('tuteur')))
        pays = clean_str(row.get(mapping.get('pays'), ''))
        if nom and tuteur:
            etudiants.append({"Prénom": prenom, "Nom": nom, "Pays": pays, "Tuteur": tuteur})
    return etudiants, None

def importer_disponibilites(uploaded_file, tuteurs_connus, co_jurys_connus, horaires_config):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], [], [error]
    
    personnes_reconnues = {p for p in (tuteurs_connus + co_jurys_connus) if p and str(p).lower() != 'nan'}
    if not personnes_reconnues: return {}, [], ["Aucun tuteur valide."]

    date_cols_map = {} 
    for col in df.columns:
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2})", str(col))
        if match:
            d_csv, h_csv = match.group(1), match.group(2)
            for jour_app, creneaux_app in horaires_config.items():
                if d_csv in jour_app:
                    for c in creneaux_app:
                        if c.startswith(h_csv): date_cols_map[col] = f"{jour_app} | {c}"; break
    
    if not date_cols_map: return {}, [], ["Aucune colonne de date reconnue."]

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
            logs.append(f"Ignoré : '{nom_brut}'")
            
    return dispos_data, list(treated), logs

# --- MOTEUR DE PLANIFICATION (RECHERCHE GLOBALE) ---

class SchedulerEngine:
    def __init__(self, etudiants, dates, nb_salles, duree, dispos, co_jurys_pool):
        self.etudiants = etudiants
        self.nb_salles = nb_salles
        self.duree = duree
        self.dispos = dispos
        self.dates = dates
        self.co_jurys_pool = list(set(co_jurys_pool))
        self.slots = self._generate_slots()
        
        self.charge_tuteur = defaultdict(int)
        self.charge_cojury = defaultdict(int)
        
        # Cibles pour l'équilibre
        self.target_cojury = defaultdict(int)
        for e in self.etudiants:
            self.target_cojury[e['Tuteur']] += 1
            
        self.jury_occupied_times = defaultdict(set)
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
        if person not in self.dispos: return False # Par défaut non dispo si pas d'info
        return self.dispos[person].get(slot_key, False)

    def solve(self):
        planning = []
        unassigned = []
        occupied_slots = set()
        busy_jurys_at_slot = defaultdict(set)
        
        # Tri : Profs les plus contraints en premier (moins de créneaux dispos)
        student_queue = []
        for etu in self.etudiants:
            tut = etu['Tuteur']
            nb_dispos = sum(1 for v in self.dispos.get(tut, {}).values() if v)
            student_queue.append((nb_dispos, random.random(), etu))
        student_queue.sort(key=lambda x: x[0])
        
        for _, _, etu in student_queue:
            tuteur = etu['Tuteur']
            best_move = None # (Score, Slot, Co-Jury)
            best_score = -float('inf')
            
            # Recherche EXPLORATOIRE : On teste tout les slots X tout les co-jurys
            # C'est un peu plus lent mais beaucoup plus puissant
            
            # 1. Identifier les slots où le tuteur est dispo et la salle libre
            available_slots_for_tutor = []
            for slot in self.slots:
                if slot['id'] in occupied_slots: continue
                if tuteur in busy_jurys_at_slot[slot['key']]: continue
                if not self.is_available(tuteur, slot['key']): continue
                available_slots_for_tutor.append(slot)
            
            # S'il n'y a aucun slot pour le tuteur, c'est mort d'avance
            if not available_slots_for_tutor:
                unassigned.append(etu)
                continue

            # 2. Pour chaque slot viable, chercher le meilleur co-jury
            for slot in available_slots_for_tutor:
                
                # Score Tuteur (Contiguïté)
                t_score = 0
                t_prev = slot['start'] - timedelta(minutes=self.duree)
                t_next = slot['end']
                if t_prev in self.jury_occupied_times[tuteur]: t_score += 2000 # Très fort bonus
                if t_next in self.jury_occupied_times[tuteur]: t_score += 2000
                if slot['jour'] in self.jury_occupied_days[tuteur]: t_score += 100
                
                # Chercher co-jurys dispos
                for cj in self.all_possible_jurys:
                    if cj == tuteur: continue
                    if cj in busy_jurys_at_slot[slot['key']]: continue
                    if not self.is_available(cj, slot['key']): continue
                    
                    # Score Co-Jury
                    cj_score = 0
                    if t_prev in self.jury_occupied_times[cj]: cj_score += 1000
                    if t_next in self.jury_occupied_times[cj]: cj_score += 1000
                    if slot['jour'] in self.jury_occupied_days[cj]: cj_score += 50
                    
                    # Score Équilibre (Dette)
                    # Dette positive = Il DOIT faire des jurys
                    dette = self.target_cojury[cj] - self.charge_cojury[cj]
                    balance_score = dette * 500 # Priorité forte à ceux qui ont de la dette
                    
                    # Score total
                    # On ajoute un petit random pour varier si égalité
                    total_score = t_score + cj_score + balance_score + random.random()
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_move = (slot, cj)
            
            # 3. Appliquer le meilleur mouvement
            if best_move:
                slot, best_cj = best_move
                
                planning.append({
                    "Étudiant": f"{etu['Prénom']} {etu['Nom']}",
                    "Tuteur": tuteur, "Co-jury": best_cj,
                    "Jour": slot['jour'], "Heure": slot['heure'],
                    "Salle": slot['salle'], "Début": slot['start'], "Fin": slot['end']
                })
                
                occupied_slots.add(slot['id'])
                busy_jurys_at_slot[slot['key']].add(tuteur)
                busy_jurys_at_slot[slot['key']].add(best_cj)
                
                for p in [tuteur, best_cj]:
                    self.jury_occupied_times[p].add(slot['start'])
                    self.jury_occupied_days[p].add(slot['jour'])
                
                self.charge_tuteur[tuteur] += 1
                self.charge_cojury[best_cj] += 1
            else:
                unassigned.append(etu)
            
        return planning, unassigned

# --- INTERFACE ---

with st.sidebar:
    st.header("🧭 Navigation")
    steps = {1: "1. Étudiants", 2: "2. Paramètres", 3: "3. Dates", 4: "4. Import Dispos", 5: "5. Génération"}
    sel = st.radio("Aller à :", list(steps.keys()), format_func=lambda x: steps[x], index=st.session_state.etape -1)
    if sel != st.session_state.etape: st.session_state.etape = sel; st.rerun()
    st.divider()
    st.write(f"Étudiants : {len(st.session_state.etudiants)}")
    st.write(f"Dispos Tuteurs : {len(st.session_state.disponibilites)}")

if st.session_state.etape == 1:
    st.title("1. Import des Étudiants")
    f = st.file_uploader("Fichier CSV Étudiants", type=['csv', 'xlsx'])
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
    st.info("Pour le CSV fourni : Mettre 50 min.")
    if st.button("Suivant"): st.session_state.etape = 3; st.rerun()

elif st.session_state.etape == 3:
    st.title("3. Dates")
    st.info("Vérifiez bien l'année et les jours pour coller au CSV de disponibilités.")
    nb = st.number_input("Nb Jours", 1, 5, max(3, len(st.session_state.dates)))
    ds = []
    cols = st.columns(4)
    for i in range(nb):
        d_def = st.session_state.dates[i] if i < len(st.session_state.dates) else datetime(2026, 1, 26).date() + timedelta(days=i)
        ds.append(cols[i%4].date_input(f"Jour {i+1}", d_def))
    st.session_state.dates = ds
    
    st.subheader("Co-jurys supplémentaires")
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
    
    f = st.file_uploader("Fichier Disponibilités", type=['csv'])
    if f:
        tuteurs_propres = [e['Tuteur'] for e in st.session_state.etudiants if e['Tuteur']]
        dispos, treated, logs = importer_disponibilites(f, tuteurs_propres, st.session_state.co_jurys, mapping_config)
        if dispos:
            st.session_state.disponibilites = dispos
            st.success(f"✅ {len(dispos)} profils importés.")
            with st.expander("Voir détails"):
                for l in logs: st.write(l)
        else:
            st.error("Aucune disponibilité valide.")
            for l in logs: st.error(l)
    if st.button("Suivant"): st.session_state.etape = 5; st.rerun()

elif st.session_state.etape == 5:
    st.title("5. Génération avec Recherche Globale")
    st.markdown("L'algorithme explore désormais toutes les combinaisons (Créneau + Co-jury) pour trouver une solution.")
    
    if st.button("Lancer la planification", type="primary"):
        eng = SchedulerEngine(
            st.session_state.etudiants, st.session_state.dates,
            st.session_state.nb_salles, st.session_state.duree,
            st.session_state.disponibilites, st.session_state.co_jurys
        )
        plan, fail = eng.solve()
        st.session_state.planning = plan
        st.session_state.failed = fail
        st.session_state.stats = (eng.charge_tuteur, eng.charge_cojury)
        
    if st.session_state.planning:
        st.success(f"Planifié : {len(st.session_state.planning)} / Échecs : {len(st.session_state.failed)}")
        
        # --- STATS ---
        if 'stats' in st.session_state:
            charge_t, charge_c = st.session_state.stats
            stats_data = []
            for t in set(list(charge_t.keys()) + list(charge_c.keys())):
                stats_data.append({
                    "Enseignant": t,
                    "Tuteur": charge_t[t],
                    "Co-Jury": charge_c[t],
                    "Total": charge_t[t] + charge_c[t],
                    "Delta": charge_t[t] - charge_c[t]
                })
            df_stats = pd.DataFrame(stats_data).sort_values("Enseignant")
            with st.expander("📊 Statistiques", expanded=True):
                st.dataframe(df_stats, use_container_width=True)

        df = pd.DataFrame(st.session_state.planning)
        tab1, tab2 = st.tabs(["Tableau", "Gantt Enseignant"])
        with tab1: st.dataframe(df)
        with tab2:
            if not df.empty:
                gantt_data = []
                for item in st.session_state.planning:
                    gantt_data.append({
                        "Enseignant": item['Tuteur'], "Rôle": "Tuteur Principal", "Étudiant": item['Étudiant'],
                        "Avec": f"co-jury: {item['Co-jury']}", "Début": item['Début'], "Fin": item['Fin'], "Jour": item['Jour'],
                        "TimeStart": datetime(2000, 1, 1, item['Début'].hour, item['Début'].minute),
                        "TimeEnd": datetime(2000, 1, 1, item['Fin'].hour, item['Fin'].minute)
                    })
                    gantt_data.append({
                        "Enseignant": item['Co-jury'], "Rôle": "Co-jury", "Étudiant": item['Étudiant'],
                        "Avec": f"tuteur: {item['Tuteur']}", "Début": item['Début'], "Fin": item['Fin'], "Jour": item['Jour'],
                        "TimeStart": datetime(2000, 1, 1, item['Début'].hour, item['Début'].minute),
                        "TimeEnd": datetime(2000, 1, 1, item['Fin'].hour, item['Fin'].minute)
                    })
                df_g = pd.DataFrame(gantt_data).sort_values("Enseignant")
                fig = px.timeline(df_g, x_start="TimeStart", x_end="TimeEnd", y="Enseignant", color="Rôle", facet_col="Jour",
                                  hover_data=["Étudiant", "Avec"], text="Étudiant",
                                  color_discrete_map={"Tuteur Principal": "#1f77b4", "Co-jury": "#ff7f0e"},
                                  height=600+(len(df_g['Enseignant'].unique())*25))
                fig.update_xaxes(tickformat="%H:%M"); fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
        
        csv = df.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("Télécharger CSV", csv, "planning.csv", "text/csv")
        
        if st.session_state.failed:
            st.error("Étudiants non placés :")
            st.dataframe(pd.DataFrame(st.session_state.failed))
