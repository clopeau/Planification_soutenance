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
st.set_page_config(page_title="Planification Soutenances v15 (Filières)", layout="wide", page_icon="🎓")

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
    "duree": 50, "failed": []
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state: st.session_state[key] = value

# --- EXPORT EXCEL AVANCE ---
def generate_excel_planning(planning_data, nb_salles):
    """Génère un fichier Excel formaté selon la demande (Feuille/Jour, Colonne/Salle)."""
    output = BytesIO()
    if not planning_data:
        return output
        
    df = pd.DataFrame(planning_data)
    
    # Créneaux théoriques (pour afficher même les trous)
    slots_matin = ["08:00", "08:50", "09:40", "10:30", "11:20", "12:10"]
    slots_aprem = ["14:00", "14:50", "15:40", "16:30", "17:20"]
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- STYLES ---
        fmt_header = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D9E1F2', 'border': 1})
        fmt_pause = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#FFF2CC', 'border': 1})
        fmt_cell = workbook.add_format({'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        fmt_time = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_room = workbook.add_format({'bold': True, 'align': 'center', 'font_size': 14, 'border': 1})

        # Trier les jours
        try:
            unique_days = sorted(df['Jour'].unique(), key=lambda x: datetime.strptime(x.split(" ")[1], "%d/%m/%Y"))
        except:
            unique_days = sorted(df['Jour'].unique())

        for jour in unique_days:
            # Nom de l'onglet (max 31 chars)
            sheet_name = re.sub(r'[\\/*?:\[\]]', "", str(jour))[:31]
            worksheet = workbook.add_worksheet(sheet_name)
            
            df_jour = df[df['Jour'] == jour]
            salles_theoriques = [f"Salle {i}" for i in range(1, nb_salles + 1)]
            
            col_offset = 0
            
            for salle in salles_theoriques:
                # Entête Salle
                worksheet.merge_range(0, col_offset, 0, col_offset + 2, salle, fmt_room)
                
                # Sous-titres
                worksheet.write(1, col_offset, "Heure", fmt_header)
                worksheet.write(1, col_offset + 1, "Etudiant", fmt_header)
                worksheet.write(1, col_offset + 2, "Jury + Co-jury", fmt_header)
                
                # Largeur colonnes
                worksheet.set_column(col_offset, col_offset, 8)     # Heure
                worksheet.set_column(col_offset + 1, col_offset + 1, 30) # Etudiant
                worksheet.set_column(col_offset + 2, col_offset + 2, 30) # Jury
                
                row_idx = 2
                
                # MATIN
                for slot in slots_matin:
                    match = df_jour[(df_jour['Salle'] == salle) & (df_jour['Heure'].str.startswith(slot))]
                    worksheet.write(row_idx, col_offset, slot, fmt_time)
                    if not match.empty:
                        data = match.iloc[0]
                        etu_txt = f"{data['Étudiant']} ({data['Pays']})" if data['Pays'] else data['Étudiant']
                        jury_txt = f"{data['Tuteur'].split(' ')[0]} + {data['Co-jury'].split(' ')[0]}"
                        worksheet.write(row_idx, col_offset + 1, etu_txt, fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, jury_txt, fmt_cell)
                    else:
                        worksheet.write(row_idx, col_offset + 1, "", fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, "", fmt_cell)
                    row_idx += 1
                
                # PAUSE
                worksheet.merge_range(row_idx, col_offset, row_idx, col_offset + 2, "PAUSE", fmt_pause)
                row_idx += 1
                
                # APRES-MIDI
                for slot in slots_aprem:
                    match = df_jour[(df_jour['Salle'] == salle) & (df_jour['Heure'].str.startswith(slot))]
                    worksheet.write(row_idx, col_offset, slot, fmt_time)
                    if not match.empty:
                        data = match.iloc[0]
                        etu_txt = f"{data['Étudiant']} ({data['Pays']})" if data['Pays'] else data['Étudiant']
                        jury_txt = f"{data['Tuteur'].split(' ')[0]} + {data['Co-jury'].split(' ')[0]}"
                        worksheet.write(row_idx, col_offset + 1, etu_txt, fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, jury_txt, fmt_cell)
                    else:
                        worksheet.write(row_idx, col_offset + 1, "", fmt_cell)
                        worksheet.write(row_idx, col_offset + 2, "", fmt_cell)
                    row_idx += 1
                
                col_offset += 4

    return output.getvalue()

# --- HELPERS & IMPORTERS ---
def clean_str(val):
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '']: return ""
    val = str(val).replace('\n', ' ').replace('\r', '').strip()
    return " ".join(val.split())

def normalize_text(text):
    if not isinstance(text, str): return str(text)
    text = text.upper().strip()
    text = "".join(text.split()) # remove all spaces
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
    if error: return [], [], [], [error] # Modifié pour retourner filieres
    
    personnes_reconnues = {p for p in (tuteurs_connus + co_jurys_connus) if p and str(p).lower() != 'nan'}
    
    # Mapping des dates
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

    # --- MODIFICATION: Recherche de la colonne FILIERE ---
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
            
            # Extraction Filière
            if col_filiere:
                filiere_val = clean_str(row.get(col_filiere, "")).upper()
                if filiere_val:
                    filieres_data[final_name] = filiere_val

            # Extraction Dates
            for col_csv, key_app in date_cols_map.items():
                val = row.get(col_csv, 0)
                try:
                    is_open = bool(int(float(val))) if pd.notna(val) else False
                    dispos_data[final_name][key_app] = is_open
                except: pass
            treated.add(final_name)
        else: logs.append(f"Ignoré: {nom_brut}")
        
    return dispos_data, list(treated), filieres_data, logs

# --- MOTEUR ---
class SchedulerEngine:
    def __init__(self, etudiants, dates, nb_salles, duree, dispos, filieres, co_jurys_pool, params):
        self.etudiants = etudiants
        self.nb_salles = nb_salles
        self.duree = duree
        self.dispos = dispos
        self.filieres = filieres # Stockage des filières
        self.dates = dates
        self.co_jurys_pool = list(set(co_jurys_pool))
        self.params = params
        self.slots = self._generate_slots()
        self.target_cojury = defaultdict(int)
        for e in self.etudiants: self.target_cojury[e['Tuteur']] += 1
        self.tuteurs_actifs = list(set(e['Tuteur'] for e in etudiants if e['Tuteur']))
        self.all_possible_jurys = list(set(self.co_jurys_pool + self.tuteurs_actifs))

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

    def run_optimization(self):
        best_sol = None; best_score = (-1, float('inf'))
        prog = st.progress(0); status = st.empty()
        for i in range(self.params['n_iterations']):
            prog.progress((i+1)/self.params['n_iterations'])
            plan, fail, charges = self._solve_single_run()
            nb = len(plan); imb = sum(abs(c['tuteur']-c['cojury']) for c in charges.values())
            if nb > best_score[0]: best_score = (nb, imb); best_sol = (plan, fail, charges)
            elif nb == best_score[0] and imb < best_score[1]: best_score = (nb, imb); best_sol = (plan, fail, charges)
        prog.empty(); status.empty()
        return best_sol

    def _solve_single_run(self):
        planning = []; unassigned = []
        occupied_slots = set(); busy_jurys = defaultdict(set)
        charge_t = defaultdict(int); charge_c = defaultdict(int)
        jury_times = defaultdict(set); jury_days = defaultdict(set)
        
        # Tri aléatoire/pondéré des étudiants
        student_queue = []
        for etu in self.etudiants:
            tut = etu['Tuteur']
            nb = sum(1 for v in self.dispos.get(tut, {}).values() if v) if tut in self.dispos else 100
            student_queue.append((nb + random.uniform(0,2), etu))
        student_queue.sort(key=lambda x: x[0])
        
        for _, etu in student_queue:
            tuteur = etu['Tuteur']
            f_tut = self.filieres.get(tuteur) # Récupère la filière du tuteur
            
            best_move = None; best_score = -float('inf')
            slots_shuffled = self.slots.copy(); random.shuffle(slots_shuffled)
            valid_slots = []
            
            for slot in slots_shuffled:
                if slot['id'] in occupied_slots: continue
                if tuteur in busy_jurys[slot['key']]: continue
                if not self.is_available(tuteur, slot['key']): continue
                valid_slots.append(slot)
                
            if not valid_slots: unassigned.append(etu); continue
            
            for slot in valid_slots:
                t_score = 0
                t_prev = slot['start'] - timedelta(minutes=self.duree); t_next = slot['end']
                if t_prev in jury_times[tuteur]: t_score += self.params['w_contiguity']
                if t_next in jury_times[tuteur]: t_score += self.params['w_contiguity']
                if slot['jour'] in jury_days[tuteur]: t_score += self.params['w_day']
                
                for cj in self.all_possible_jurys:
                    if cj == tuteur: continue
                    
                    # --- CONTRAINTE FILIERE ---
                    f_cj = self.filieres.get(cj)
                    # Si les deux ont une filière définie et qu'elles sont différentes => on saute
                    if f_tut and f_cj and f_tut != f_cj:
                        continue
                    # --------------------------

                    if cj in busy_jurys[slot['key']]: continue
                    if not self.is_available(cj, slot['key']): continue
                    
                    cj_score = 0
                    if t_prev in jury_times[cj]: cj_score += self.params['w_contiguity']
                    if t_next in jury_times[cj]: cj_score += self.params['w_contiguity']
                    if slot['jour'] in jury_days[cj]: cj_score += self.params['w_day']
                    
                    bal_score = (self.target_cojury[cj] - charge_c[cj]) * self.params['w_balance']
                    total = t_score + cj_score + bal_score + random.uniform(0, self.params['w_random'])
                    
                    if total > best_score: best_score = total; best_move = (slot, cj)
            
            if best_move:
                slot, best_cj = best_move
                planning.append({"Étudiant": f"{etu['Prénom']} {etu['Nom']}", "Pays": etu['Pays'], "Tuteur": tuteur, "Co-jury": best_cj, "Jour": slot['jour'], "Heure": slot['heure'], "Salle": slot['salle'], "Début": slot['start'], "Fin": slot['end']})
                occupied_slots.add(slot['id'])
                busy_jurys[slot['key']].add(tuteur); busy_jurys[slot['key']].add(best_cj)
                for p in [tuteur, best_cj]: jury_times[p].add(slot['start']); jury_days[p].add(slot['jour'])
                charge_t[tuteur] += 1; charge_c[best_cj] += 1
            else: unassigned.append(etu)
            
        final_charges = defaultdict(lambda: {'tuteur':0, 'cojury':0})
        for p,v in charge_t.items(): final_charges[p]['tuteur'] = v
        for p,v in charge_c.items(): final_charges[p]['cojury'] = v
        return planning, unassigned, final_charges

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
        st.write(f"Filières détectées : {len(st.session_state.filieres)}")

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
    st.subheader("Co-jurys supplémentaires"); txt = st.text_input("Nom")
    if txt and txt not in st.session_state.co_jurys: st.session_state.co_jurys.append(txt)
    if st.session_state.co_jurys: st.write(st.session_state.co_jurys)
    if st.button("Suivant"): st.session_state.etape = 4; st.rerun()

elif st.session_state.etape == 4:
    st.title("4. Import Disponibilités")
    st.info("Le fichier doit contenir une colonne nommée 'FILIERE' pour activer la contrainte de filière.")
    eng = SchedulerEngine([], st.session_state.dates, 1, st.session_state.duree, {}, {}, [], {})
    mapping_config = defaultdict(list)
    for s in eng.slots: k = s['key'].split(" | "); mapping_config[k[0]].append(k[1])
    f = st.file_uploader("Fichier Disponibilités", type=['xlsx', 'csv'])
    if f:
        tuteurs_propres = [e['Tuteur'] for e in st.session_state.etudiants if e['Tuteur']]
        dispos, treated, filieres, logs = importer_disponibilites(f, tuteurs_propres, st.session_state.co_jurys, mapping_config)
        if dispos:
            st.session_state.disponibilites = dispos
            st.session_state.filieres = filieres
            st.success(f"✅ {len(dispos)} profils importés.")
            if filieres:
                st.success(f"✅ {len(filieres)} filières associées trouvées.")
            else:
                st.warning("⚠️ Aucune colonne 'FILIERE' trouvée. La contrainte ne sera pas appliquée.")
            
            with st.expander("Logs"): 
                for l in logs: st.write(l)
        else: st.error("Erreur import."); 
    if st.button("Suivant"): st.session_state.etape = 5; st.rerun()

elif st.session_state.etape == 5:
    st.title("5. Génération")
    with st.expander("Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        n_iter = c1.slider("Itérations", 10, 200, 50)
        w_rand = c2.slider("Exploration", 0, 500, 100)
        c3, c4 = st.columns(2)
        w_cont = c3.slider("Poids Contiguïté", 0, 5000, 2000)
        w_bal = c4.slider("Poids Équilibre", 0, 2000, 500)
    
    if st.button("Lancer", type="primary"):
        params = {"n_iterations": n_iter, "w_random": w_rand, "w_contiguity": w_cont, "w_balance": w_bal, "w_day": 100}
        # Ajout de self.filieres lors de l'appel
        eng = SchedulerEngine(
            st.session_state.etudiants, 
            st.session_state.dates, 
            st.session_state.nb_salles, 
            st.session_state.duree, 
            st.session_state.disponibilites, 
            st.session_state.filieres,  # <-- PASSAGE DES FILIERES
            st.session_state.co_jurys, 
            params
        )
        plan, fail, charges = eng.run_optimization()
        st.session_state.planning = plan; st.session_state.failed = fail; st.session_state.stats_charges = charges
        
    if st.session_state.planning:
        st.success(f"Placés : {len(st.session_state.planning)} | Échecs : {len(st.session_state.failed)}")
        if 'stats_charges' in st.session_state:
            charges = st.session_state.stats_charges; data = []
            all_p = set(charges.keys())
            for e in st.session_state.etudiants: all_p.add(e['Tuteur'])
            for p in all_p:
                c_t = charges[p]['tuteur']; c_c = charges[p]['cojury']
                fil = st.session_state.filieres.get(p, "-")
                data.append({"Enseignant": p, "Filière": fil, "Tuteur": c_t, "Co-Jury": c_c, "Delta": c_t-c_c})
            st.dataframe(pd.DataFrame(data).sort_values("Enseignant"), use_container_width=True)

        # BOUTON EXPORT EXCEL FORMATE
        excel_data = generate_excel_planning(st.session_state.planning, st.session_state.nb_salles)
        st.download_button(
            label="📥 Télécharger Planning Formaté (.xlsx)",
            data=excel_data,
            file_name="Planning_Soutenances.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        df = pd.DataFrame(st.session_state.planning)
        tab1, tab2 = st.tabs(["Tableau", "Gantt"])
        with tab1: st.dataframe(df)
        with tab2:
            if not df.empty:
                gantt = []
                for x in st.session_state.planning:
                    for role, p in [("Tuteur", x['Tuteur']), ("Co-Jury", x['Co-jury'])]:
                        gantt.append({"Enseignant": p, "Role": role, "Etudiant": x['Étudiant'], "Jour": x['Jour'], "Start": datetime(2000,1,1,x['Début'].hour, x['Début'].minute), "End": datetime(2000,1,1,x['Fin'].hour, x['Fin'].minute)})
                df_g = pd.DataFrame(gantt).sort_values("Enseignant")
                fig = px.timeline(df_g, x_start="Start", x_end="End", y="Enseignant", color="Role", facet_col="Jour", text="Etudiant", height=800)
                fig.update_xaxes(tickformat="%H:%M"); fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
        if st.session_state.failed: st.error("Non placés :"); st.dataframe(pd.DataFrame(st.session_state.failed))
