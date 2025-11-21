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
st.set_page_config(page_title="Planification Soutenances v10 (Fix)", layout="wide", page_icon="🛠️")

# --- STYLES ---
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
    val_str = str(val).strip()
    # Nettoyage des sauts de ligne qui cassent l'affichage
    return val_str.replace("\n", " ").replace("\r", "")

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    text = text.upper()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return text

def lire_csv_robuste(uploaded_file):
    """Lecture renforcée pour gérer les ; dans les descriptions."""
    uploaded_file.seek(0)
    content = uploaded_file.getvalue()
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for enc in encodings:
        try:
            decoded = content.decode(enc)
            # On essaie de lire avec le moteur python qui gère mieux les guillemets "..." autour des textes avec ;
            return pd.read_csv(StringIO(decoded), sep=';', engine='python', quotechar='"'), None
        except: 
            # Fallback sur la virgule
            try:
                return pd.read_csv(StringIO(decoded), sep=',', engine='python'), None
            except: continue
            
    return None, "Lecture impossible. Vérifiez que les champs textes contenant des ';' sont bien entre guillemets."

# --- IMPORTERS ---
def importer_etudiants(uploaded_file):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], error
    
    # Mapping EXACT basé sur votre fichier
    col_map = {}
    
    # On cherche d'abord les colonnes exactes de votre fichier
    targets = {
        'nom': ['NOM', 'Nom'],
        'prenom': ['PRENOM', 'Prénom'],
        'tuteur': ['Enseignant référent (NOM Prénom)', 'Enseignant référent'],
        'pays': ['Service d’accueil – Pays', 'Pays']
    }
    
    # 1. Recherche Exacte
    for key, candidates in targets.items():
        for cand in candidates:
            if cand in df.columns:
                col_map[key] = cand
                break
    
    # 2. Fallback Fuzzy (si échec exact)
    raw_cols = list(df.columns)
    cols_norm = {normalize_text(c): c for c in raw_cols}
    
    if 'nom' not in col_map:
        for cn, cr in cols_norm.items(): 
            if "NOM" in cn and "PRENOM" not in cn and "REFERENT" not in cn: col_map['nom'] = cr; break
    if 'prenom' not in col_map:
        for cn, cr in cols_norm.items(): 
            if "PRENOM" in cn: col_map['prenom'] = cr; break
    if 'tuteur' not in col_map:
        for cn, cr in cols_norm.items(): 
            if ("REFERENT" in cn or "TUTEUR" in cn) and "ENTREPRISE" not in cn: col_map['tuteur'] = cr; break
    if 'pays' not in col_map:
        for cn, cr in cols_norm.items(): 
            if "PAYS" in cn: col_map['pays'] = cr; break

    missing = [k for k in ['nom', 'tuteur'] if k not in col_map]
    if missing: return [], f"Colonnes manquantes: {missing}. Colonnes lues: {raw_cols}"

    etudiants = []
    for _, row in df.iterrows():
        n = clean_str(row.get(col_map.get('nom')))
        p = clean_str(row.get(col_map.get('prenom'), ''))
        t = clean_str(row.get(col_map.get('tuteur')))
        y = clean_str(row.get(col_map.get('pays'), ''))
        
        # Correction noms inversés (Ex: Tuteur = "PERRIN Emmanuel" vs Dispo = "EMMANUEL PERRIN")
        # On garde tel quel ici, le fuzzy matching de l'import dispo fera le lien
        if n and t:
            etudiants.append({"Prénom": p, "Nom": n, "Pays": y, "Tuteur": t})
            
    return etudiants, None

def importer_disponibilites(uploaded_file, tuteurs_connus, co_jurys_connus, horaires_config):
    df, error = lire_csv_robuste(uploaded_file)
    if error: return [], [], [error]
    
    personnes_reconnues = {p for p in (tuteurs_connus + co_jurys_connus) if p and str(p).lower() != 'nan'}
    
    date_cols_map = {} 
    for col in df.columns:
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2})", str(col))
        if match:
            d_csv, h_csv = match.group(1), match.group(2)
            for j_app, c_list in horaires_config.items():
                if d_csv in j_app:
                    for c in c_list:
                        if c.startswith(h_csv): date_cols_map[col] = f"{j_app} | {c}"; break
    
    if not date_cols_map: return {}, [], ["Pas de colonnes dates valides (Vérifiez l'année)."]

    dispos_data = {}
    treated = set()
    logs = []
    col_nom = df.columns[0]
    
    for _, row in df.iterrows():
        nom_brut = clean_str(row[col_nom])
        if not nom_brut: continue
        
        best_match, best_score = None, 0
        for p in personnes_reconnues:
            # Token Sort Ratio: "Dupont Jean" == "Jean Dupont" (100%)
            score = fuzz.token_sort_ratio(nom_brut.lower(), p.lower())
            if score > best_score: best_score, best_match = score, p
        
        if best_score >= 70: # Seuil de tolérance
            final_name = best_match
            if final_name not in dispos_data: dispos_data[final_name] = {}
            for col_csv, key_app in date_cols_map.items():
                val = row.get(col_csv, 0)
                try:
                    dispos_data[final_name][key_app] = bool(int(float(val))) if pd.notna(val) else False
                except: pass
            treated.add(final_name)
        else:
            logs.append(f"Ignoré: {nom_brut} (Match: {best_match} {best_score}%)")
            
    return dispos_data, list(treated), logs

# --- MOTEUR DE PLANIFICATION ---

class SchedulerEngine:
    def __init__(self, etudiants, dates, nb_salles, duree, dispos, co_jurys_pool, params):
        self.etudiants = etudiants
        self.nb_salles = nb_salles
        self.duree = duree
        self.dispos = dispos
        self.dates = dates
        self.co_jurys_pool = list(set(co_jurys_pool))
        self.params = params
        self.slots = self._generate_slots()
        
        self.target_cojury = defaultdict(int)
        for e in self.etudiants: self.target_cojury[e['Tuteur']] += 1
            
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
        # MODIFICATION IMPORTANTE: Si la personne n'est pas dans le fichier de dispo,
        # on considère qu'elle est DISPONIBLE PAR DÉFAUT (pour éviter de tout bloquer).
        # Sauf si l'option stricte est activée (paramètre à ajouter éventuellement).
        if person not in self.dispos: return True 
        return self.dispos[person].get(slot_key, False)

    def run_optimization(self):
        best_sol = None
        best_score = (-1, float('inf'))
        
        prog = st.progress(0)
        status = st.empty()
        
        for i in range(self.params['n_iterations']):
            prog.progress((i+1)/self.params['n_iterations'])
            status.text(f"Simulation {i+1}/{self.params['n_iterations']}...")
            
            plan, fail, charges = self._solve_single_run()
            nb_placed = len(plan)
            
            # Calcul déséquilibre dette (somme des carrés pour pénaliser les gros écarts)
            imbalance = sum(abs(c['tuteur'] - c['cojury']) for c in charges.values())
            
            if nb_placed > best_score[0]:
                best_score = (nb_placed, imbalance)
                best_sol = (plan, fail, charges)
            elif nb_placed == best_score[0] and imbalance < best_score[1]:
                best_score = (nb_placed, imbalance)
                best_sol = (plan, fail, charges)
        
        prog.empty(); status.empty()
        return best_sol

    def _solve_single_run(self):
        planning = []
        unassigned = []
        occupied_slots = set()
        busy_jurys_at_slot = defaultdict(set)
        
        charge_tuteur = defaultdict(int)
        charge_cojury = defaultdict(int)
        jury_occupied_times = defaultdict(set)
        jury_occupied_days = defaultdict(set)
        
        # Tri des étudiants
        student_queue = []
        for etu in self.etudiants:
            tut = etu['Tuteur']
            # Si tuteur inconnu dans dispo, on met 100 (très dispo) pour ne pas le prioriser
            nb_dispos = sum(1 for v in self.dispos.get(tut, {}).values() if v) if tut in self.dispos else 100
            student_queue.append((nb_dispos + random.uniform(0,2), etu))
        student_queue.sort(key=lambda x: x[0])
        
        for _, etu in student_queue:
            tuteur = etu['Tuteur']
            best_move = None
            best_score = -float('inf')
            
            # Mélange des salles
            slots_shuffled = self.slots.copy()
            random.shuffle(slots_shuffled)
            
            # Pré-filtre slots valides pour Tuteur
            valid_slots = []
            for slot in slots_shuffled:
                if slot['id'] in occupied_slots: continue
                if tuteur in busy_jurys_at_slot[slot['key']]: continue
                if not self.is_available(tuteur, slot['key']): continue
                valid_slots.append(slot)
            
            if not valid_slots:
                unassigned.append(etu)
                continue
                
            for slot in valid_slots:
                # Score Tuteur
                t_score = 0
                t_prev = slot['start'] - timedelta(minutes=self.duree)
                t_next = slot['end']
                
                if t_prev in jury_occupied_times[tuteur]: t_score += self.params['w_contiguity']
                if t_next in jury_occupied_times[tuteur]: t_score += self.params['w_contiguity']
                if slot['jour'] in jury_occupied_days[tuteur]: t_score += self.params['w_day']
                
                # Chercher Co-jury
                for cj in self.all_possible_jurys:
                    if cj == tuteur: continue
                    if cj in busy_jurys_at_slot[slot['key']]: continue
                    if not self.is_available(cj, slot['key']): continue
                    
                    cj_score = 0
                    if t_prev in jury_occupied_times[cj]: cj_score += self.params['w_contiguity']
                    if t_next in jury_occupied_times[cj]: cj_score += self.params['w_contiguity']
                    if slot['jour'] in jury_occupied_days[cj]: cj_score += self.params['w_day']
                    
                    dette = self.target_cojury[cj] - charge_cojury[cj]
                    bal_score = dette * self.params['w_balance']
                    
                    rnd = random.uniform(0, self.params['w_random'])
                    
                    total = t_score + cj_score + bal_score + rnd
                    if total > best_score:
                        best_score = total
                        best_move = (slot, cj)
            
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
                    jury_occupied_times[p].add(slot['start'])
                    jury_occupied_days[p].add(slot['jour'])
                charge_tuteur[tuteur] += 1
                charge_cojury[best_cj] += 1
            else:
                unassigned.append(etu)
                
        final_charges = defaultdict(lambda: {'tuteur':0, 'cojury':0})
        for p,v in charge_tuteur.items(): final_charges[p]['tuteur'] = v
        for p,v in charge_cojury.items(): final_charges[p]['cojury'] = v
        return planning, unassigned, final_charges

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
    f = st.file_uploader("Fichier CSV (avec colonnes NOM, PRENOM, TUTEUR...)", type=['csv', 'xlsx'])
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
    st.info("Sélectionnez les jours du CSV (ex: 26, 27, 29 Janvier 2026).")
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
    eng = SchedulerEngine([], st.session_state.dates, 1, st.session_state.duree, {}, [], {})
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
            
            # Warning pour les tuteurs sans dispo
            missing_tuteurs = [t for t in tuteurs_propres if t not in dispos]
            if missing_tuteurs:
                st.warning(f"⚠️ {len(missing_tuteurs)} Tuteurs sans disponibilités détectés. Ils seront considérés 'Toujours Disponibles'.")
                with st.expander("Voir les tuteurs sans dispo"):
                    st.write(missing_tuteurs)
            
            with st.expander("Voir logs import"):
                for l in logs: st.write(l)
        else:
            st.error("Aucune disponibilité valide.")
            for l in logs: st.error(l)
    if st.button("Suivant"): st.session_state.etape = 5; st.rerun()

elif st.session_state.etape == 5:
    st.title("5. Génération")
    
    with st.expander("🎛️ Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        n_iter = c1.slider("Itérations", 10, 200, 50)
        w_rand = c2.slider("Exploration (Aléatoire)", 0, 500, 100)
        c3, c4 = st.columns(2)
        w_cont = c3.slider("Poids Contiguïté", 0, 5000, 2000)
        w_bal = c4.slider("Poids Équilibre Dette", 0, 2000, 500)
    
    if st.button("Lancer", type="primary"):
        params = {"n_iterations": n_iter, "w_random": w_rand, "w_contiguity": w_cont, "w_balance": w_bal, "w_day": 100}
        eng = SchedulerEngine(
            st.session_state.etudiants, st.session_state.dates,
            st.session_state.nb_salles, st.session_state.duree,
            st.session_state.disponibilites, st.session_state.co_jurys,
            params
        )
        plan, fail, charges = eng.run_optimization()
        st.session_state.planning = plan
        st.session_state.failed = fail
        st.session_state.stats_charges = charges
        
    if st.session_state.planning:
        st.success(f"Résultat : {len(st.session_state.planning)} placés, {len(st.session_state.failed)} échecs.")
        
        if 'stats_charges' in st.session_state:
            charges = st.session_state.stats_charges
            data = []
            all_p = set(charges.keys())
            for e in st.session_state.etudiants: all_p.add(e['Tuteur'])
            for p in all_p:
                c_t = charges[p]['tuteur']
                c_c = charges[p]['cojury']
                data.append({"Enseignant": p, "Tuteur": c_t, "Co-Jury": c_c, "Delta": c_t - c_c})
            st.dataframe(pd.DataFrame(data).sort_values("Enseignant"), use_container_width=True)

        df = pd.DataFrame(st.session_state.planning)
        
        tab1, tab2 = st.tabs(["Tableau", "Gantt"])
        with tab1: st.dataframe(df)
        with tab2:
            if not df.empty:
                gantt = []
                for x in st.session_state.planning:
                    for role, p in [("Tuteur", x['Tuteur']), ("Co-Jury", x['Co-jury'])]:
                        gantt.append({
                            "Enseignant": p, "Role": role, "Etudiant": x['Étudiant'], "Jour": x['Jour'],
                            "Start": datetime(2000,1,1,x['Début'].hour, x['Début'].minute),
                            "End": datetime(2000,1,1,x['Fin'].hour, x['Fin'].minute)
                        })
                df_g = pd.DataFrame(gantt).sort_values("Enseignant")
                fig = px.timeline(df_g, x_start="Start", x_end="End", y="Enseignant", color="Role", facet_col="Jour", text="Etudiant", height=800)
                fig.update_xaxes(tickformat="%H:%M"); fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                
        if st.session_state.failed:
            st.error("Non placés :")
            st.dataframe(pd.DataFrame(st.session_state.failed))
