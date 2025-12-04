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
import math
import copy

# --- CONFIGURATION ---
st.set_page_config(page_title="Planification Soutenances v21 (Recuit + Confort)", layout="wide", page_icon="🎓")

# --- STYLES ---
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    </style>
""", unsafe_allow_html=True)

# --- STATE ---
DEFAULT_STATE = {
    "etape": 1, "etudiants": [], "co_jurys": [], "dates": [],
    "disponibilites": {}, "filieres": {}, "planning": [], "nb_salles": 2,
    "duree": 50, "failed": [], "stats_charges": {}
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state: st.session_state[key] = value

# --- HELPER: Extraction Nom ---
def extract_nom_only(fullname):
    if not isinstance(fullname, str) or not fullname: return ""
    parts = fullname.strip().split()
    upper_parts = [p for p in parts if p.isupper() and len(p) > 1]
    if upper_parts: return " ".join(upper_parts)
    return parts[0] if parts else ""

# --- EXPORT EXCEL ---
def generate_excel_planning(planning_data, nb_salles):
    output = BytesIO()
    if not planning_data: return output
    df = pd.DataFrame(planning_data)
    slots_matin = ["08:00", "08:50", "09:40", "10:30", "11:20", "12:10"]
    slots_aprem = ["14:00", "14:50", "15:40", "16:30", "17:20"]
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        fmt_header = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D9E1F2', 'border': 1})
        fmt_pause = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#FFF2CC', 'border': 1})
        fmt_cell = workbook.add_format({'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        fmt_time = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_room = workbook.add_format({'bold': True, 'align': 'center', 'font_size': 14, 'border': 1})

        try:
            unique_days = sorted(df['Jour'].unique(), key=lambda x: datetime.strptime(x.split(" ")[1], "%d/%m/%Y"))
        except:
            unique_days = sorted(df['Jour'].unique())

        for jour in unique_days:
            sheet_name = re.sub(r'[\\/*?:\[\]]', "", str(jour))[:31]
            worksheet = workbook.add_worksheet(sheet_name)
            df_jour = df[df['Jour'] == jour]
            salles_theoriques = [f"Salle {i}" for i in range(1, nb_salles + 1)]
            col_offset = 0
            
            for salle in salles_theoriques:
                worksheet.merge_range(0, col_offset, 0, col_offset + 2, salle, fmt_room)
                worksheet.write(1, col_offset, "Heure", fmt_header)
                worksheet.write(1, col_offset + 1, "Etudiant", fmt_header)
                worksheet.write(1, col_offset + 2, "Jury + Co-jury", fmt_header)
                worksheet.set_column(col_offset, col_offset, 8)
                worksheet.set_column(col_offset + 1, col_offset + 1, 30)
                worksheet.set_column(col_offset + 2, col_offset + 2, 30)
                
                row_idx = 2
                for slot in slots_matin:
                    match = df_jour[(df_jour['Salle'] == salle) & (df_jour['Heure'].str.startswith(slot))]
                    worksheet.write(row_idx, col_offset, slot, fmt_time)
                    if not match.empty:
                        data = match.iloc[0]
                        etu_txt = f"{data['Étudiant']} ({data['Pays']})" if data['Pays'] else data['Étudiant']
                        nom_tuteur = extract_nom_only(data['Tuteur'])
                        nom_cojury = extract_nom_only(data['Co-jury'])
                        jury_txt = f"{nom_tuteur} + {nom_cojury}"
                        worksheet.write(row_idx, col_offset + 1, etu_txt, fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, jury_txt, fmt_cell)
                    else:
                        worksheet.write(row_idx, col_offset + 1, "", fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, "", fmt_cell)
                    row_idx += 1
                
                worksheet.merge_range(row_idx, col_offset, row_idx, col_offset + 2, "PAUSE", fmt_pause)
                row_idx += 1
                
                for slot in slots_aprem:
                    match = df_jour[(df_jour['Salle'] == salle) & (df_jour['Heure'].str.startswith(slot))]
                    worksheet.write(row_idx, col_offset, slot, fmt_time)
                    if not match.empty:
                        data = match.iloc[0]
                        etu_txt = f"{data['Étudiant']} ({data['Pays']})" if data['Pays'] else data['Étudiant']
                        nom_tuteur = extract_nom_only(data['Tuteur'])
                        nom_cojury = extract_nom_only(data['Co-jury'])
                        jury_txt = f"{nom_tuteur} + {nom_cojury}"
                        worksheet.write(row_idx, col_offset + 1, etu_txt, fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, jury_txt, fmt_cell)
                    else:
                        worksheet.write(row_idx, col_offset + 1, "", fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, "", fmt_cell)
                    row_idx += 1
                col_offset += 4
    return output.getvalue()

# --- IMPORTERS ---
def clean_str(val):
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '']: return ""
    val = str(val).replace('\n', ' ').replace('\r', '').strip()
    return " ".join(val.split())

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    text = text.upper().strip()
    text = "".join(text.split())
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return text

def lire_fichier_robuste(uploaded_file):
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(uploaded_file), None
        uploaded_file.seek(0)
        content = uploaded_file.getvalue()
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';'] 
        for enc in encodings:
            try:
                decoded = content.decode(enc)
                for sep in separators:
                    try:
                        df = pd.read_csv(StringIO(decoded), sep=sep, engine='python', quotechar='"', on_bad_lines='skip')
                        if len(df.columns) > 1: return df, None
                    except: continue
            except: continue
        return None, "Impossible de lire le fichier."
    except Exception as e: return None, f"Erreur technique : {str(e)}"

def importer_etudiants(uploaded_file):
    df, error = lire_fichier_robuste(uploaded_file)
    if error: return [], error
    df.columns = [str(c).strip().replace('\xa0', ' ') for c in df.columns]
    col_map = {}
    targets = {'nom': ['NOM','Nom'], 'prenom': ['PRENOM','Prénom'], 'tuteur': ['Enseignant référent (NOM Prénom)','Enseignant référent','Tuteur'], 'pays': ['Service d’accueil – Pays','Pays']}
    for key, candidates in targets.items():
        for cand in candidates:
            if cand in df.columns: col_map[key] = cand; break
    raw_cols = list(df.columns); cols_norm = {normalize_text(c): c for c in raw_cols}
    if 'nom' not in col_map:
        for n, r in cols_norm.items(): 
            if "NOM" in n and "PRENOM" not in n and "REFERENT" not in n and "ACCUEIL" not in n: col_map['nom'] = r; break
    if 'tuteur' not in col_map:
        for n, r in cols_norm.items():
            if ("REFERENT" in n or "ENSEIGNANT" in n) and "ENTREPRISE" not in n: col_map['tuteur'] = r; break
    missing = [k for k in ['nom', 'tuteur'] if k not in col_map]
    if missing: return [], f"Colonnes introuvables : {missing}. Colonnes : {raw_cols}"
    etudiants = []
    for _, row in df.iterrows():
        n = clean_str(row.get(col_map.get('nom')))
        p = clean_str(row.get(col_map.get('prenom'), ''))
        t = clean_str(row.get(col_map.get('tuteur')))
        y = clean_str(row.get(col_map.get('pays'), ''))
        if len(t) > 60 or len(n) > 60: continue 
        if n and t and t.lower() != 'nan': etudiants.append({"Prénom": p, "Nom": n, "Pays": y, "Tuteur": t})
    return etudiants, None

def importer_disponibilites(uploaded_file, tuteurs_connus, co_jurys_connus, horaires_config):
    df, error = lire_fichier_robuste(uploaded_file)
    if error: return [], [], [], [error]
    
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
    
    if not date_cols_map: return {}, [], {}, ["Pas de colonnes dates valides."]

    col_filiere = None
    for c in df.columns:
        if "FILIERE" in str(c).upper() or "DEPARTEMENT" in str(c).upper():
            col_filiere = c
            break

    dispos_data = {}; treated = set(); logs = []; filieres_data = {}
    col_nom = df.columns[0]
    
    for _, row in df.iterrows():
        nom_brut = clean_str(row[col_nom])
        if not nom_brut: continue
        best_match, best_score = None, 0
        for p in personnes_reconnues:
            score = fuzz.token_sort_ratio(nom_brut.lower(), p.lower())
            if score > best_score: best_score, best_match = score, p
            
        if best_score >= 60:
            final_name = best_match
            if final_name not in dispos_data: dispos_data[final_name] = {}
            if col_filiere:
                filiere_val = clean_str(row.get(col_filiere, "")).upper()
                if filiere_val: filieres_data[final_name] = filiere_val
            for col_csv, key_app in date_cols_map.items():
                val = row.get(col_csv, 0)
                try:
                    is_open = bool(int(float(val))) if pd.notna(val) else False
                    dispos_data[final_name][key_app] = is_open
                except: pass
            treated.add(final_name)
        else: logs.append(f"Ignoré: {nom_brut}")
        
    return dispos_data, list(treated), filieres_data, logs

# --- ALGO: RECUIT SIMULÉ (Simulated Annealing) ---
class AnnealingScheduler:
    def __init__(self, etudiants, dates, nb_salles, duree, dispos, filieres, params):
        self.etudiants = etudiants
        self.nb_salles = nb_salles
        self.duree = duree
        self.dispos = dispos
        self.filieres = filieres
        self.dates = dates
        self.params = params
        
        # Génération des créneaux
        self.slots = self._generate_slots()
        self.slots_map = {s['id']: s for s in self.slots}
        
        # Cibles strictes (Parité)
        self.target_cojury = defaultdict(int)
        for e in self.etudiants: 
            self.target_cojury[e['Tuteur']] += 1
            
        # Jurys potentiels (seulement ceux qui ont une cible > 0)
        self.active_jurys = list(self.target_cojury.keys())
        
        # Pré-calcul des disponibilités
        self.tutor_valid_slots = defaultdict(list)
        for p in self.active_jurys:
            for s in self.slots:
                if self.is_available(p, s['key']):
                    self.tutor_valid_slots[p].append(s['id'])
                    
        # Variables d'état
        self.solution = []
        self.unassigned = []

    def _generate_slots(self):
        slots = []; slot_id = 0
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
                            slots.append({"id": slot_id, "key": key, "jour": d_str, "heure": h_str, "salle": f"Salle {s}", "start": curr, "end": fin})
                            slot_id += 1
                        curr = fin
                except: continue
        return slots

    def is_available(self, person, slot_key):
        if person not in self.dispos: return True 
        return self.dispos[person].get(slot_key, False)

    def initial_solution_greedy(self):
        """Construction rapide d'une solution valide (mais moche)"""
        self.solution = []
        self.unassigned = []
        occupied = set()
        
        # On essaie de grouper par tuteur dès le début si possible (petit boost)
        etudiants_sorted = sorted(range(len(self.etudiants)), key=lambda x: self.etudiants[x]['Tuteur'])
        
        for idx in etudiants_sorted:
            etu = self.etudiants[idx]
            tut = etu['Tuteur']
            
            # Créneaux valides pour le tuteur non occupés
            possibles = [sid for sid in self.tutor_valid_slots[tut] if sid not in occupied]
            
            if possibles:
                # On prend le premier dispo (tendance à grouper chronologiquement si slots triés)
                sid = possibles[0] 
                occupied.add(sid)
                
                # Co-jury au hasard valide
                cands = []
                f_tut = self.filieres.get(tut)
                for cj in self.active_jurys:
                    if cj == tut: continue
                    if f_tut and self.filieres.get(cj) and f_tut != self.filieres.get(cj): continue
                    if self.is_available(cj, self.slots_map[sid]['key']):
                        cands.append(cj)
                
                cj = random.choice(cands) if cands else random.choice([x for x in self.active_jurys if x != tut])
                
                self.solution.append({
                    "idx": idx,
                    "slot": sid,
                    "tuteur": tut,
                    "cojury": cj
                })
            else:
                self.unassigned.append(idx)

    def calculate_cost(self, current_sol):
        """Calcul complet de l'énergie (plus c'est bas, mieux c'est)"""
        cost = 0
        
        # 1. Non assignés (Interdit)
        cost += len(self.unassigned) * 1_000_000
        
        # Structure de données pour analyse temporelle
        # person_schedule[nom] = [(start_datetime, slot_id, salle)]
        person_schedule = defaultdict(list)
        
        # Compteurs pour parité
        cojury_counts = defaultdict(int)
        
        # Compteurs conflits (doublons horaires)
        slot_occupancy = defaultdict(int) # slot_id -> count
        person_occupancy = defaultdict(list) # nom -> [time_key]
        
        for s in current_sol:
            cojury_counts[s['cojury']] += 1
            slot_info = self.slots_map[s['slot']]
            
            # Enregistrement pour analyse temporelle
            entry = (slot_info['start'], s['slot'], slot_info['salle'])
            person_schedule[s['tuteur']].append(entry)
            person_schedule[s['cojury']].append(entry)
            
            # Check dispo immédiat
            if not self.is_available(s['cojury'], slot_info['key']):
                cost += 5_000 # Cojury pas libre
                
            slot_occupancy[s['slot']] += 1
            person_occupancy[s['tuteur']].append(slot_info['key'])
            person_occupancy[s['cojury']].append(slot_info['key'])

        # 2. Parité (Critique)
        parity_error = 0
        for p in self.active_jurys:
            diff = abs(cojury_counts[p] - self.target_cojury[p])
            parity_error += diff
        cost += parity_error * 10_000
        
        # 3. Conflits Physiques (Ubiquité ou Salle occupée doublement)
        phys_conflicts = 0
        for pid, count in slot_occupancy.items():
            if count > 1: phys_conflicts += (count - 1)
        
        for p, times in person_occupancy.items():
            if len(times) != len(set(times)):
                phys_conflicts += (len(times) - len(set(times)))
                
        cost += phys_conflicts * 50_000

        # 4. Confort (Trous & Salles) -> C'est ici que l'amélioration se joue
        # On veut minimiser les trous et les changements de salle
        
        for p, entries in person_schedule.items():
            if not entries: continue
            # Tri chronologique
            entries.sort(key=lambda x: x[0])
            
            for i in range(len(entries) - 1):
                t1, sid1, room1 = entries[i]
                t2, sid2, room2 = entries[i+1]
                
                # Écart en minutes
                delta_min = (t2 - (t1 + timedelta(minutes=self.duree))).total_seconds() / 60
                
                # --- A. PÉNALITÉ TROUS (GAPS) ---
                if delta_min == 0:
                    # Contigu : OK
                    # --- B. PÉNALITÉ SALLE (ROOM SWAP) ---
                    # Si c'est collé mais qu'on change de salle, c'est pénible
                    if room1 != room2:
                        cost += 500 # Pénalité changement salle
                
                elif delta_min < 0:
                    # Chevauchement (déjà puni par phys_conflicts, mais on rajoute pour guider)
                    cost += 1000
                
                else:
                    # Il y a un trou
                    if delta_min > 90: 
                        # Grand trou (Pause déj ou trou matin/soir)
                        # On tolère plus facilement, mais on préfère éviter si possible
                        # ex: 8h-9h puis 16h-17h -> delta grand -> punition moyenne
                        cost += 100 # Coût fixe pour "revenir plus tard"
                    else:
                        # Petit trou (ex: 10h-11h, trou, 12h-13h)
                        # C'est le plus chiant : attendre 1h pour rien
                        cost += delta_min * 10 # 10 pts par minute d'attente
        
        return cost

    def run_annealing(self):
        # Init
        self.initial_solution_greedy()
        
        current_sol = copy.deepcopy(self.solution)
        current_cost = self.calculate_cost(current_sol)
        
        best_sol = copy.deepcopy(current_sol)
        best_cost = current_cost
        
        # Paramètres
        T = 2000.0
        alpha = 0.98 # Refroidissement lent pour bien explorer
        steps = self.params['n_iterations'] * 150 # Beaucoup d'itérations
        
        prog = st.progress(0)
        status = st.empty()
        
        for i in range(steps):
            if i % 100 == 0:
                prog.progress(min(1.0, i/steps))
                status.text(f"Optimisation ({i}/{steps})... Coût: {int(best_cost)}")
            
            # --- MUTATION ---
            candidate = copy.deepcopy(current_sol)
            if not candidate: break
            
            move_type = random.random()
            idx_mod = random.randint(0, len(candidate)-1)
            elem = candidate[idx_mod]
            
            # Type 1: Changer Co-Jury (Pour la Parité) - 40%
            if move_type < 0.4:
                others = [p for p in self.active_jurys if p != elem['tuteur']]
                if others:
                    new_cj = random.choice(others)
                    f_tut = self.filieres.get(elem['tuteur'])
                    f_cj = self.filieres.get(new_cj)
                    if not f_tut or not f_cj or f_tut == f_cj:
                         candidate[idx_mod]['cojury'] = new_cj
            
            # Type 2: Déplacer dans un slot vide (Pour boucher les trous) - 30%
            elif move_type < 0.7:
                all_valid = self.tutor_valid_slots[elem['tuteur']]
                if all_valid:
                    # On privilégie un slot qui est proche d'un autre slot du tuteur
                    # (Heuristique locale pour accélérer la convergence)
                    new_slot = random.choice(all_valid)
                    candidate[idx_mod]['slot'] = new_slot

            # Type 3: Swap de créneau avec un autre (Pour réorganiser) - 30%
            else:
                target_idx = random.randint(0, len(candidate)-1)
                if target_idx != idx_mod:
                    target = candidate[target_idx]
                    s1, s2 = elem['slot'], target['slot']
                    candidate[idx_mod]['slot'] = s2
                    candidate[target_idx]['slot'] = s1
            
            # --- ACCEPTATION ---
            new_cost = self.calculate_cost(candidate)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_sol = candidate
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_sol = copy.deepcopy(current_sol)
                    best_cost = current_cost
            
            T *= alpha
        
        prog.empty(); status.empty()
        
        # Reconstruction Résultat
        final_planning = []
        charges_t = defaultdict(int)
        charges_c = defaultdict(int)
        
        for item in best_sol:
            etu = self.etudiants[item['idx']]
            s = self.slots_map[item['slot']]
            final_planning.append({
                "Étudiant": f"{etu['Prénom']} {etu['Nom']}", 
                "Pays": etu['Pays'], 
                "Tuteur": item['tuteur'], 
                "Co-jury": item['cojury'], 
                "Jour": s['jour'], 
                "Heure": s['heure'], 
                "Salle": s['salle'], 
                "Début": s['start'], 
                "Fin": s['end']
            })
            charges_t[item['tuteur']] += 1
            charges_c[item['cojury']] += 1
            
        final_charges = {}
        all_p = set(charges_t.keys()) | set(charges_c.keys()) | set(self.active_jurys)
        for p in all_p:
            final_charges[p] = {'tuteur': charges_t[p], 'cojury': charges_c[p]}
            
        return final_planning, self.unassigned, final_charges

# --- UI ---
with st.sidebar:
    st.header("🧭 Navigation")
    steps = {1: "1. Étudiants", 2: "2. Paramètres", 3: "3. Dates", 4: "4. Import Dispos", 5: "5. Génération"}
    sel = st.radio("Aller à :", list(steps.keys()), format_func=lambda x: steps[x], index=st.session_state.etape -1)
    if sel != st.session_state.etape: st.session_state.etape = sel; st.rerun()
    st.divider()
    st.info("Algorithme v3 : Parité Stricte + Compactage Temporel + Stabilité Salle")

if st.session_state.etape == 1:
    st.title("1. Import des Étudiants")
    f = st.file_uploader("Fichier Étudiants (Excel/CSV)", type=['xlsx', 'csv'])
    if f:
        data, msg = importer_etudiants(f)
        if not data: st.error(msg)
        else:
            st.session_state.etudiants = data
            if msg: st.warning(msg)
            st.success(f"{len(data)} étudiants importés.")
            st.dataframe(pd.DataFrame(data).head())
            if st.button("Suivant"): st.session_state.etape = 2; st.rerun()

elif st.session_state.etape == 2:
    st.title("2. Paramètres")
    c1, c2 = st.columns(2)
    st.session_state.nb_salles = c1.number_input("Salles", 1, 10, st.session_state.nb_salles)
    st.session_state.duree = c2.number_input("Durée (min)", 30, 120, st.session_state.duree)
    if st.button("Suivant"): st.session_state.etape = 3; st.rerun()

elif st.session_state.etape == 3:
    st.title("3. Dates")
    nb = st.number_input("Nb Jours", 1, 5, max(3, len(st.session_state.dates)))
    ds = []; cols = st.columns(4)
    for i in range(nb):
        d_def = st.session_state.dates[i] if i < len(st.session_state.dates) else datetime(2026, 1, 26).date() + timedelta(days=i)
        ds.append(cols[i%4].date_input(f"Jour {i+1}", d_def))
    st.session_state.dates = ds
    if st.button("Suivant"): st.session_state.etape = 4; st.rerun()

elif st.session_state.etape == 4:
    st.title("4. Import Disponibilités")
    st.info("Le fichier doit contenir une colonne nommée 'FILIERE'.")
    eng_dummy = AnnealingScheduler([], st.session_state.dates, 1, st.session_state.duree, {}, {}, {})
    mapping_config = defaultdict(list)
    for s in eng_dummy.slots: k = s['key'].split(" | "); mapping_config[k[0]].append(k[1])
    
    f = st.file_uploader("Fichier Disponibilités", type=['xlsx', 'csv'])
    if f:
        tuteurs_propres = [e['Tuteur'] for e in st.session_state.etudiants if e['Tuteur']]
        dispos, treated, filieres, logs = importer_disponibilites(f, tuteurs_propres, st.session_state.co_jurys, mapping_config)
        if dispos:
            st.session_state.disponibilites = dispos
            st.session_state.filieres = filieres
            st.success(f"✅ {len(dispos)} enseignants importés.")
        else: st.error("Erreur import.")
    if st.button("Suivant"): st.session_state.etape = 5; st.rerun()

elif st.session_state.etape == 5:
    st.title("5. Génération Optimisée")
    
    st.write("Cet algorithme va chercher une solution qui minimise (dans l'ordre) :")
    st.write("1. Les étudiants non placés (Priorité absolue)")
    st.write("2. Les écarts de parité (Strict)")
    st.write("3. Les trous dans l'emploi du temps (Confort)")
    st.write("4. Les changements de salle intempestifs (Confort)")
    
    n_iter = st.slider("Puissance de calcul (Temps vs Qualité)", 50, 500, 200)
    
    if st.button("Lancer l'Optimisation", type="primary"):
        params = {"n_iterations": n_iter}
        eng = AnnealingScheduler(
            st.session_state.etudiants, st.session_state.dates, st.session_state.nb_salles, st.session_state.duree, 
            st.session_state.disponibilites, st.session_state.filieres, params
        )
        plan, fail, charges = eng.run_annealing()
        st.session_state.planning = plan
        st.session_state.failed = fail
        st.session_state.stats_charges = charges
        
    if st.session_state.planning:
        st.divider()
        st.header("Résultats")
        
        # Stats
        charges = st.session_state.stats_charges
        total_delta = sum(abs(v['tuteur'] - v['cojury']) for v in charges.values())
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Étudiants Placés", f"{len(st.session_state.planning)} / {len(st.session_state.etudiants)}")
        c2.metric("Écart Parité Total", total_delta, delta_color="inverse")
        c3.metric("Non placés", len(st.session_state.failed), delta_color="inverse")

        # Excel
        excel_data = generate_excel_planning(st.session_state.planning, st.session_state.nb_salles)
        st.download_button("📥 Télécharger Planning (.xlsx)", excel_data, "Planning_Soutenances.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Visu
        df = pd.DataFrame(st.session_state.planning)
        if not df.empty:
            tab1, tab2 = st.tabs(["Tableau", "Gantt (Visuel)"])
            with tab1: 
                st.dataframe(df)
                data_stats = []
                for p, vals in charges.items():
                    data_stats.append({"Enseignant": p, "Tuteur": vals['tuteur'], "Co-Jury": vals['cojury'], "Delta": vals['tuteur']-vals['cojury']})
                st.write("Détail Parité :"); st.dataframe(pd.DataFrame(data_stats))
                
            with tab2:
                gantt = []
                for x in st.session_state.planning:
                    for role, p in [("Tuteur", x['Tuteur']), ("Co-Jury", x['Co-jury'])]:
                        gantt.append({"Enseignant": p, "Role": role, "Etudiant": x['Étudiant'], "Jour": x['Jour'], "Start": datetime(2000,1,1,x['Début'].hour, x['Début'].minute), "End": datetime(2000,1,1,x['Fin'].hour, x['Fin'].minute), "Salle": x['Salle']})
                df_g = pd.DataFrame(gantt).sort_values(["Enseignant", "Start"])
                fig = px.timeline(df_g, x_start="Start", x_end="End", y="Enseignant", color="Salle", facet_col="Jour", text="Etudiant", height=800)
                fig.update_xaxes(tickformat="%H:%M"); fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Les couleurs représentent les salles. Idéalement, une ligne 'Enseignant' ne devrait pas changer de couleur souvent sur une même journée.")
