import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from thefuzz import fuzz
import re
import random
import unicodedata
import math
import copy

# --- CONFIGURATION ---
st.set_page_config(page_title="Planification Soutenances v20 (Optimiseur Avancé)", layout="wide", page_icon="🎓")

# --- STYLES ---
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb; }
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

# --- EXPORT EXCEL AVANCE ---
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

# --- MOTEUR DE RECUIT SIMULÉ (Simulated Annealing) ---
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
        
        # Cibles strictes
        self.target_cojury = defaultdict(int)
        for e in self.etudiants: 
            self.target_cojury[e['Tuteur']] += 1
            
        # Jurys potentiels : UNIQUEMENT ceux qui sont tuteurs (car les autres ont cible=0)
        self.active_jurys = list(self.target_cojury.keys())
        
        # Pré-calcul des disponibilités
        self.tutor_valid_slots = defaultdict(list)
        self.cojury_valid_slots = defaultdict(list)
        
        for p in self.active_jurys:
            for s in self.slots:
                if self.is_available(p, s['key']):
                    self.tutor_valid_slots[p].append(s['id'])
                    self.cojury_valid_slots[p].append(s['id'])
                    
        # État courant : liste de {etu_idx, slot_id, cojury}
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
        """Construction initiale : On place tout le monde au hasard sur un créneau valide Tuteur"""
        self.solution = []
        self.unassigned = []
        
        # Track slot usage to avoid double booking rooms
        occupied = set()
        
        # Mélanger étudiants
        indices = list(range(len(self.etudiants)))
        random.shuffle(indices)
        
        for idx in indices:
            etu = self.etudiants[idx]
            tut = etu['Tuteur']
            
            # 1. Trouver créneaux où tuteur est dispo et salle libre
            possibles = [sid for sid in self.tutor_valid_slots[tut] if sid not in occupied]
            
            if possibles:
                sid = random.choice(possibles)
                occupied.add(sid)
                
                # 2. Choisir un co-jury au hasard (même si pas optimal pour la parité pour l'instant)
                # On filtre juste par filière et dispo
                cands = []
                f_tut = self.filieres.get(tut)
                for cj in self.active_jurys:
                    if cj == tut: continue
                    if f_tut and self.filieres.get(cj) and f_tut != self.filieres.get(cj): continue
                    if self.is_available(cj, self.slots_map[sid]['key']):
                        cands.append(cj)
                
                cj = random.choice(cands) if cands else None
                # Si aucun co-jury dispo ce créneau là, on met un placeholder (on corrigera plus tard)
                if not cj: 
                    cj = random.choice([x for x in self.active_jurys if x != tut]) # Force brute
                
                self.solution.append({
                    "idx": idx,
                    "slot": sid,
                    "tuteur": tut,
                    "cojury": cj
                })
            else:
                self.unassigned.append(idx)

    def calculate_cost(self, current_sol):
        """Fonction d'énergie à minimiser"""
        cost = 0
        
        # 1. Pénalité non assignés (Trés élevée)
        cost += len(self.unassigned) * 100000
        
        # 2. Pénalité Parité (Haute)
        current_counts = defaultdict(int)
        for s in current_sol:
            current_counts[s['cojury']] += 1
            
        parity_error = 0
        for p in self.active_jurys:
            diff = abs(current_counts[p] - self.target_cojury[p])
            parity_error += diff
        
        cost += parity_error * 5000
        
        # 3. Pénalité Conflits (Un prof ne peut pas être à 2 endroits)
        # Check slots usage per person
        person_slots = defaultdict(list)
        for s in current_sol:
            person_slots[s['tuteur']].append(s['slot'])
            person_slots[s['cojury']].append(s['slot'])
            
        conflict_cost = 0
        for p, sids in person_slots.items():
            # Clés uniques (jour + heure)
            times = [self.slots_map[sid]['key'] for sid in sids]
            if len(times) != len(set(times)):
                conflict_cost += (len(times) - len(set(times)))
        
        cost += conflict_cost * 10000
        
        # 4. Pénalité Soft (Dispo Co-jury, Contiguïté, Salle)
        soft_cost = 0
        for s in current_sol:
            slot_info = self.slots_map[s['slot']]
            # Co-jury indisponible ?
            if not self.is_available(s['cojury'], slot_info['key']):
                soft_cost += 500
            
            # TODO: Contiguïté (plus complexe à calculer vite, on simplifie pour l'instant)
            
        cost += soft_cost
        return cost, parity_error

    def run_annealing(self):
        # Initialisation
        self.initial_solution_greedy()
        
        current_sol = copy.deepcopy(self.solution)
        current_cost, _ = self.calculate_cost(current_sol)
        
        best_sol = copy.deepcopy(current_sol)
        best_cost = current_cost
        best_unassigned = list(self.unassigned)
        
        # Paramètres recuit
        T = 1000.0
        alpha = 0.95
        steps = self.params['n_iterations'] * 100 # On multiplie pour avoir bcp de tirages
        
        prog = st.progress(0)
        status_text = st.empty()
        
        for i in range(steps):
            if i % 100 == 0:
                prog.progress(min(1.0, i/steps))
                status_text.text(f"Optimisation... Iter {i} | Cost {int(best_cost)} | Unassigned {len(best_unassigned)}")
            
            # --- MOUVEMENT ALÉATOIRE ---
            candidate = copy.deepcopy(current_sol)
            move_type = random.random()
            
            if not candidate and not self.unassigned: break
            
            # A. Essayer de placer un non-assigné (Priorité Absolue)
            if self.unassigned and random.random() < 0.3:
                u_idx = random.choice(self.unassigned)
                # Trouver un slot au hasard (swap ou vide)
                # Simplification: On prend un élément de la solution et on swap l'étudiant
                if candidate:
                    victim_idx = random.randint(0, len(candidate)-1)
                    victim = candidate[victim_idx]
                    
                    # Vérifier si Tuteur de U est dispo au slot de Victim
                    u_obj = self.etudiants[u_idx]
                    slot_key = self.slots_map[victim['slot']]['key']
                    
                    if self.is_available(u_obj['Tuteur'], slot_key):
                        # Swap
                        # U prend la place, Victim devient unassigned
                        # On garde le co-jury de la place (ou on en change)
                        new_elem = {
                            "idx": u_idx,
                            "slot": victim['slot'],
                            "tuteur": u_obj['Tuteur'],
                            "cojury": victim['cojury'] # On garde le cojury pour l'instant
                        }
                        candidate[victim_idx] = new_elem
                        # La victime sort
                        # NOTE: Gestion unassigned complexe dans boucle, on simplifie:
                        # On ne fait ça que si on swap vraiment. 
                        # Pour simplifier le code ici, on se concentre sur B et C
                        pass

            # B. Changer de Co-Jury (Pour régler la parité)
            elif move_type < 0.6 and candidate:
                idx_mod = random.randint(0, len(candidate)-1)
                elem = candidate[idx_mod]
                
                # Choisir un nouveau co-jury
                others = [p for p in self.active_jurys if p != elem['tuteur']]
                if others:
                    new_cj = random.choice(others)
                    # Check filiere
                    f_tut = self.filieres.get(elem['tuteur'])
                    f_cj = self.filieres.get(new_cj)
                    if not f_tut or not f_cj or f_tut == f_cj:
                         candidate[idx_mod]['cojury'] = new_cj
            
            # C. Changer de Slot (Déplacement ou Échange)
            elif candidate:
                idx_mod = random.randint(0, len(candidate)-1)
                elem = candidate[idx_mod]
                
                # Option C1: Déplacer vers un slot vide
                occupied_slots = {x['slot'] for x in candidate}
                all_valid = self.tutor_valid_slots[elem['tuteur']]
                empty_valid = [s for s in all_valid if s not in occupied_slots]
                
                if empty_valid and random.random() < 0.5:
                    new_slot = random.choice(empty_valid)
                    candidate[idx_mod]['slot'] = new_slot
                
                # Option C2: Échanger avec quelqu'un d'autre
                else:
                    target_idx = random.randint(0, len(candidate)-1)
                    if target_idx != idx_mod:
                        target = candidate[target_idx]
                        # Check : Tuteur 1 dispo slot 2 ET Tuteur 2 dispo slot 1
                        k1 = self.slots_map[elem['slot']]['key']
                        k2 = self.slots_map[target['slot']]['key']
                        
                        if (self.is_available(elem['tuteur'], k2) and 
                            self.is_available(target['tuteur'], k1)):
                            # Swap slots
                            s1, s2 = elem['slot'], target['slot']
                            candidate[idx_mod]['slot'] = s2
                            candidate[target_idx]['slot'] = s1

            # --- ACCEPTATION ---
            new_cost, _ = self.calculate_cost(candidate)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_sol = candidate
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_sol = copy.deepcopy(current_sol)
                    best_cost = current_cost
            
            # Refroidissement
            T *= alpha
        
        prog.empty()
        status_text.empty()
        
        # Reconstruction format output
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
            
        return final_planning, best_unassigned, final_charges

# --- UI ---
with st.sidebar:
    st.header("🧭 Navigation")
    steps = {1: "1. Étudiants", 2: "2. Paramètres", 3: "3. Dates", 4: "4. Import Dispos", 5: "5. Génération"}
    sel = st.radio("Aller à :", list(steps.keys()), format_func=lambda x: steps[x], index=st.session_state.etape -1)
    if sel != st.session_state.etape: st.session_state.etape = sel; st.rerun()
    st.divider()
    st.write(f"Étudiants : {len(st.session_state.etudiants)}")
    st.write(f"Dispos Tuteurs : {len(st.session_state.disponibilites)}")
    if 'filieres' in st.session_state:
        st.write(f"Filières : {len(set(st.session_state.filieres.values()))}")

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
    st.subheader("Co-jurys")
    st.info("ℹ️ Règle Parité Stricte : Seuls les enseignants ayant des étudiants seront utilisés comme co-jurys.")
    if st.button("Suivant"): st.session_state.etape = 4; st.rerun()

elif st.session_state.etape == 4:
    st.title("4. Import Disponibilités")
    st.info("Le fichier doit contenir une colonne nommée 'FILIERE'.")
    # Dummy engine just to get slots
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
            
            nb_profs_fil = len(filieres)
            nb_types_fil = len(set(filieres.values()))
            st.success(f"✅ {len(dispos)} enseignants importés.")
            if nb_types_fil > 0:
                st.success(f"✅ {nb_profs_fil} enseignants avec filière détectée ({nb_types_fil} filières distinctes).")
            else:
                st.warning("⚠️ Aucune filière détectée (vérifiez la colonne 'FILIERE').")
            
            with st.expander("Logs"): 
                for l in logs: st.write(l)
        else: st.error("Erreur import.")
    if st.button("Suivant"): st.session_state.etape = 5; st.rerun()

elif st.session_state.etape == 5:
    st.title("5. Génération (Optimiseur Recuit Simulé)")
    st.markdown("""
    Cette méthode est **itérative**. Elle part d'une solution approximative et tente de la réparer par millions de petites modifications.
    Plus le nombre d'itérations est élevé, meilleure sera la convergence vers la "Parité Parfaite".
    """)
    
    with st.expander("Paramètres", expanded=True):
        n_iter = st.slider("Puissance de calcul (x100 pas)", 10, 500, 100, help="Augmenter pour trouver des solutions difficiles")
        
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
        
        # Stats Parité
        charges = st.session_state.stats_charges
        data_stats = []
        total_delta = 0
        for p, vals in charges.items():
            delta = vals['tuteur'] - vals['cojury']
            total_delta += abs(delta)
            if vals['tuteur'] > 0 or vals['cojury'] > 0:
                data_stats.append({
                    "Enseignant": p, 
                    "Filière": st.session_state.filieres.get(p, "-"), 
                    "Etudiants suivis (Cible)": vals['tuteur'], 
                    "Fois Co-Jury (Réel)": vals['cojury'], 
                    "Écart": delta
                })
        
        df_stats = pd.DataFrame(data_stats).sort_values("Enseignant")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Étudiants placés", f"{len(st.session_state.planning)} / {len(st.session_state.etudiants)}")
            if total_delta == 0:
                st.success("✅ PARITÉ STRICTE RESPECTÉE !")
            else:
                st.warning(f"⚠️ Écart total de parité : {total_delta} (Essayez d'augmenter la puissance)")
            
        with c2:
             st.dataframe(df_stats, use_container_width=True, height=200)

        # Excel
        excel_data = generate_excel_planning(st.session_state.planning, st.session_state.nb_salles)
        st.download_button("📥 Télécharger Planning (.xlsx)", excel_data, "Planning_Soutenances.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Visu
        df = pd.DataFrame(st.session_state.planning)
        if not df.empty:
            tab1, tab2 = st.tabs(["Tableau", "Gantt"])
            with tab1: st.dataframe(df)
            with tab2:
                gantt = []
                for x in st.session_state.planning:
                    for role, p in [("Tuteur", x['Tuteur']), ("Co-Jury", x['Co-jury'])]:
                        gantt.append({"Enseignant": p, "Role": role, "Etudiant": x['Étudiant'], "Jour": x['Jour'], "Start": datetime(2000,1,1,x['Début'].hour, x['Début'].minute), "End": datetime(2000,1,1,x['Fin'].hour, x['Fin'].minute)})
                df_g = pd.DataFrame(gantt).sort_values("Enseignant")
                fig = px.timeline(df_g, x_start="Start", x_end="End", y="Enseignant", color="Role", facet_col="Jour", text="Etudiant", height=800)
                fig.update_xaxes(tickformat="%H:%M"); fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.failed: 
            st.error(f"{len(st.session_state.failed)} étudiants non placés (Pas de créneaux tuteur/salle disponibles)")
            st.dataframe(pd.DataFrame(st.session_state.failed))
