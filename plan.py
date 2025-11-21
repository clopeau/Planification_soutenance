import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from io import BytesIO, StringIO
from thefuzz import fuzz
import re

st.set_page_config(page_title="Planification Soutenances", layout="wide")
st.title("Planification Optimisée des Soutenances de Stage")

# Configuration des étapes
etapes = [
    "etudiants", "salles", "duree_soutenance", "co_jury",
    "dates", "disponibilites", "disponibilites_selection", "generation"
]

etapes_labels = {
    "etudiants": "Étape 1 : Étudiants",
    "salles": "Étape 2 : Salles",
    "duree_soutenance": "Étape 3 : Durée",
    "co_jury": "Étape 4 : Co-jurys",
    "dates": "Étape 5 : Dates",
    "disponibilites": "Étape 6 : Créneaux",
    "disponibilites_selection": "Étape 7 : Disponibilités",
    "generation": "Étape 8 : Planning"
}

def afficher_navigation():
    st.sidebar.markdown("### 🧭 Navigation")
    current_etape_index = 0
    if 'etape' in st.session_state and st.session_state.etape in etapes:
        current_etape_index = etapes.index(st.session_state.etape)
    
    etape_selectionnee = st.sidebar.selectbox(
        "Aller à une autre étape :",
        options=etapes,
        format_func=lambda x: etapes_labels.get(x, x),
        index=current_etape_index,
        key="navigation_selectbox"
    )
    if 'etape' not in st.session_state or etape_selectionnee != st.session_state.etape:
        st.session_state.etape = etape_selectionnee
        st.rerun()

# Initialisation des variables de session
if "etape" not in st.session_state: st.session_state.etape = "etudiants"
if "etudiants" not in st.session_state: st.session_state.etudiants = []
if "co_jurys" not in st.session_state: st.session_state.co_jurys = []
if "dates_soutenance" not in st.session_state: st.session_state.dates_soutenance = []
if "disponibilites" not in st.session_state: st.session_state.disponibilites = {}
if "planning_final" not in st.session_state: st.session_state.planning_final = []
if "nb_salles" not in st.session_state: st.session_state.nb_salles = 2
if "duree_soutenance" not in st.session_state: st.session_state.duree_soutenance = 50 
if "horaires_par_jour" not in st.session_state: st.session_state.horaires_par_jour = {}

@dataclass
class Individu:
    genes: List[int]
    fitness: float = 0.0
    soutenances_planifiees: int = 0
    conflits: int = 0

class AlgorithmeGenetique:
    def __init__(self, planificateur, taille_population=80, nb_generations=800,
                 taux_mutation=0.12, taux_croisement=0.8):
        self.planificateur = planificateur
        self.taille_population = taille_population
        self.nb_generations = nb_generations
        self.taux_mutation = taux_mutation
        self.taux_croisement = taux_croisement
        self.creneaux = planificateur.generer_creneaux_uniques()
        self.nb_etudiants = len(planificateur.etudiants) if planificateur.etudiants else 0
        self.creneaux_valides_par_etudiant = self._precalculer_creneaux_valides() if self.nb_etudiants > 0 else {}
        self.historique_fitness = []
        self.meilleure_solution = Individu(genes=[-1]*self.nb_etudiants) if self.nb_etudiants > 0 else Individu(genes=[])

    def _precalculer_creneaux_valides(self):
        creneaux_valides = {}
        if not self.planificateur.etudiants or not self.creneaux: return creneaux_valides
        for idx_etu, etudiant in enumerate(self.planificateur.etudiants):
            tuteur = etudiant["Tuteur"]
            creneaux_possibles = []
            for idx_creneau, creneau in enumerate(self.creneaux):
                if self.planificateur.est_disponible(tuteur, creneau['jour'], creneau['heure']):
                    co_jurys_disponibles = self.planificateur.trouver_co_jurys_disponibles(
                        tuteur, creneau['jour'], creneau['heure']
                    )
                    if co_jurys_disponibles:
                        creneaux_possibles.append(idx_creneau)
            creneaux_valides[idx_etu] = creneaux_possibles
        return creneaux_valides

    def generer_individu_intelligent(self) -> Individu:
        genes = [-1] * self.nb_etudiants
        if self.nb_etudiants == 0: return Individu(genes=genes)
        creneaux_occupes_par_id = set()
        jurys_occupes_par_moment = defaultdict(set) 
        ordre_etudiants = list(range(self.nb_etudiants))
        random.shuffle(ordre_etudiants)
        for idx_etu in ordre_etudiants:
            if idx_etu >= len(self.planificateur.etudiants): continue
            creneaux_possibles_etu = self.creneaux_valides_par_etudiant.get(idx_etu, [])
            shuffled_creneaux_possibles = creneaux_possibles_etu.copy()
            random.shuffle(shuffled_creneaux_possibles)
            for idx_creneau_cand in shuffled_creneaux_possibles:
                if idx_creneau_cand >= len(self.creneaux): continue
                creneau_cand = self.creneaux[idx_creneau_cand]
                if creneau_cand['id'] in creneaux_occupes_par_id: continue
                tuteur_etu = self.planificateur.etudiants[idx_etu]["Tuteur"]
                moment_cand = creneau_cand['moment']
                if tuteur_etu in jurys_occupes_par_moment[moment_cand]: continue
                co_jurys_dispos_pour_creneau = self.planificateur.trouver_co_jurys_disponibles(
                    tuteur_etu, creneau_cand['jour'], creneau_cand['heure']
                )
                co_jury_final_pour_gen = None
                random.shuffle(co_jurys_dispos_pour_creneau)
                for co_j_cand_gen in co_jurys_dispos_pour_creneau:
                    if co_j_cand_gen not in jurys_occupes_par_moment[moment_cand]:
                        co_jury_final_pour_gen = co_j_cand_gen
                        break
                if co_jury_final_pour_gen:
                    genes[idx_etu] = idx_creneau_cand
                    creneaux_occupes_par_id.add(creneau_cand['id'])
                    jurys_occupes_par_moment[moment_cand].add(tuteur_etu)
                    jurys_occupes_par_moment[moment_cand].add(co_jury_final_pour_gen)
                    break
        return Individu(genes=genes)

    def calculer_fitness_amelioree(self, individu: Individu) -> Individu:
        planning = self.decoder_individu(individu)
        nb_soutenances = len(planning)
        nb_total_etudiants = self.nb_etudiants 
        taux_planification = nb_soutenances / nb_total_etudiants if nb_total_etudiants > 0 else 0
        conflits_salle, conflits_jury_fitness = self._analyser_conflits_detailles(planning)
        total_conflits = conflits_salle + conflits_jury_fitness
        equilibrage = self._calculer_equilibrage_charge(planning)
        bonus_alternance = self._calculer_bonus_alternance(planning)
        roles_par_jury = defaultdict(lambda: {'tuteur': 0, 'cojury': 0})
        for soutenance in planning:
            roles_par_jury[soutenance['Tuteur']]['tuteur'] += 1
            roles_par_jury[soutenance['Co-jury']]['cojury'] += 1
        penalite_balance_roles = 0
        score_balance_roles = 0 
        personnes_eligibles_balance = set(self.planificateur.tuteurs_referents) & set(self.planificateur.co_jurys)
        contrainte_parite_roles_violee_strict = False
        for jury, counts in roles_par_jury.items():
            if jury in personnes_eligibles_balance:
                difference = abs(counts['tuteur'] - counts['cojury'])
                if difference > 0: penalite_balance_roles += (difference ** 2) * 20 
                if difference == 0: score_balance_roles += 100 
                elif difference == 1: score_balance_roles += 30
        fitness_base = (taux_planification * 912 + max(0, (nb_soutenances - (nb_total_etudiants * 0.75))) * 50 + equilibrage * 30 + bonus_alternance * 15 + score_balance_roles * 1.5 - (nb_total_etudiants - nb_soutenances) * 150 - penalite_balance_roles)
        fitness_finale = fitness_base
        if total_conflits > 0: fitness_finale = -500_000 - (total_conflits * 1000) 
        else:
            if nb_soutenances >= nb_total_etudiants * 0.9: fitness_finale += 2000 
        individu.fitness = fitness_finale
        individu.soutenances_planifiees = nb_soutenances
        individu.conflits = total_conflits
        return individu

    def _analyser_conflits_detailles(self, planning):
        conflits_salle = 0
        conflits_jury = 0
        creneaux_salle_utilises = {} 
        jurys_occupes_moment = defaultdict(set)
        for soutenance in planning:
            cle_salle_moment = f"{soutenance['Jour']}_{soutenance['Créneau']}_{soutenance['Salle']}"
            moment_sout = f"{soutenance['Jour']}_{soutenance['Créneau']}"
            if cle_salle_moment in creneaux_salle_utilises: conflits_salle += 1
            creneaux_salle_utilises[cle_salle_moment] = True
            tuteur = soutenance['Tuteur']; co_jury = soutenance['Co-jury']
            if tuteur in jurys_occupes_moment[moment_sout]: conflits_jury += 1
            else: jurys_occupes_moment[moment_sout].add(tuteur)
            if co_jury in jurys_occupes_moment[moment_sout]: conflits_jury += 1
            else: jurys_occupes_moment[moment_sout].add(co_jury)
        return conflits_salle, conflits_jury

    def _calculer_equilibrage_charge(self, planning):
        if not planning: return 0.0 
        charges = defaultdict(int)
        for soutenance in planning:
            charges[soutenance['Tuteur']] += 1
            charges[soutenance['Co-jury']] += 1
        if not charges or len(charges) <= 1 : return 10.0 
        return max(0.0, 10.0 - np.sqrt(np.var(np.array(list(charges.values()))))) 

    def _calculer_bonus_alternance(self, planning): 
        jurys_par_periode = {'matin': defaultdict(set), 'apres_midi': defaultdict(set)} 
        for soutenance in planning:
            periode = 'matin' if soutenance['Début'].hour < 13 else 'apres_midi'
            jurys_par_periode[periode][soutenance['Jour']].add(soutenance['Tuteur'])
            jurys_par_periode[periode][soutenance['Jour']].add(soutenance['Co-jury'])
        total_jurys_alternant = 0
        for jour in set(jurys_par_periode['matin'].keys()) | set(jurys_par_periode['apres_midi'].keys()):
            total_jurys_alternant += len(jurys_par_periode['matin'].get(jour, set()) & jurys_par_periode['apres_midi'].get(jour, set()))
        return float(total_jurys_alternant * 1.0)

    def croisement_intelligent(self, parent1: Individu, parent2: Individu) -> Tuple[Individu, Individu]:
        len_genes = len(parent1.genes)
        if len_genes == 0: return Individu(genes=[]), Individu(genes=[])
        enfant1_genes = [-1] * len_genes; enfant2_genes = [-1] * len_genes
        point = random.randint(1, len_genes - 1) if len_genes > 1 else 0
        for i in range(point):
            enfant1_genes[i] = parent1.genes[i]; enfant2_genes[i] = parent2.genes[i]
        creneaux_e1 = set(g for g in enfant1_genes[:point] if g != -1)
        creneaux_e2 = set(g for g in enfant2_genes[:point] if g != -1)
        for i in range(point, len_genes):
            g_p2 = parent2.genes[i]
            if g_p2 != -1 and g_p2 not in creneaux_e1 and (i in self.creneaux_valides_par_etudiant and g_p2 in self.creneaux_valides_par_etudiant[i]):
                enfant1_genes[i] = g_p2; creneaux_e1.add(g_p2)
            else:
                g_p1 = parent1.genes[i]
                if g_p1 != -1 and g_p1 not in creneaux_e1 and (i in self.creneaux_valides_par_etudiant and g_p1 in self.creneaux_valides_par_etudiant[i]):
                    enfant1_genes[i] = g_p1; creneaux_e1.add(g_p1)
            g_p1_e2 = parent1.genes[i]
            if g_p1_e2 != -1 and g_p1_e2 not in creneaux_e2 and (i in self.creneaux_valides_par_etudiant and g_p1_e2 in self.creneaux_valides_par_etudiant[i]):
                enfant2_genes[i] = g_p1_e2; creneaux_e2.add(g_p1_e2)
            else:
                g_p2_e2 = parent2.genes[i]
                if g_p2_e2 != -1 and g_p2_e2 not in creneaux_e2 and (i in self.creneaux_valides_par_etudiant and g_p2_e2 in self.creneaux_valides_par_etudiant[i]):
                    enfant2_genes[i] = g_p2_e2; creneaux_e2.add(g_p2_e2)
        return Individu(genes=enfant1_genes), Individu(genes=enfant2_genes)

    def mutation_adaptative(self, individu: Individu) -> Individu:
        if not individu.genes: return individu 
        for i in range(len(individu.genes)):
            if random.random() < self.taux_mutation:
                if i >= self.nb_etudiants: continue 
                creneaux_possibles = self.creneaux_valides_par_etudiant.get(i, [])
                if not creneaux_possibles: continue
                gene_actuel = individu.genes[i]
                others = set(individu.genes[j] for j in range(len(individu.genes)) if j != i and individu.genes[j] != -1)
                options = [c for c in creneaux_possibles if c not in others]
                if not options:
                    if gene_actuel != -1 and gene_actuel in others : individu.genes[i] = -1
                    continue
                if gene_actuel == -1: individu.genes[i] = random.choice(options)
                else:
                    nouveaux = [c for c in options if c != gene_actuel]
                    if nouveaux: individu.genes[i] = random.choice(nouveaux)
                    elif gene_actuel not in options : individu.genes[i] = -1
        return individu

    def evoluer(self):
        population = []
        if self.nb_etudiants == 0: return [], self._stats_vides()
        for _ in range(self.taille_population):
            population.append(self.calculer_fitness_amelioree(self.generer_individu_intelligent()))
        if not population: return [], self._stats_vides()
        self.meilleure_solution = max(population, key=lambda x: x.fitness)
        stagnation = 0
        progress_bar = st.sidebar.progress(0)
        for generation in range(self.nb_generations):
            progress_bar.progress((generation + 1) / self.nb_generations)
            nouvelle_population = []
            population_triee = sorted(population, key=lambda x: x.fitness, reverse=True)
            elite_size = max(1, int(self.taille_population * 0.1)) 
            nouvelle_population.extend(population_triee[:elite_size])
            while len(nouvelle_population) < self.taille_population:
                if random.random() < self.taux_croisement and len(population) >= 2 :
                    parent1 = self.selection_tournament(population, k=5)
                    parent2 = self.selection_tournament(population, k=5)
                    enfant1, enfant2 = self.croisement_intelligent(parent1, parent2)
                    nouvelle_population.append(self.mutation_adaptative(enfant1))
                    if len(nouvelle_population) < self.taille_population: nouvelle_population.append(self.mutation_adaptative(enfant2))
                else: nouvelle_population.append(self.mutation_adaptative(random.choice(population)))
            population = [self.calculer_fitness_amelioree(ind) for ind in nouvelle_population[:self.taille_population]]
            meilleur = max(population, key=lambda x: x.fitness)
            if meilleur.fitness > self.meilleure_solution.fitness:
                self.meilleure_solution = meilleur; stagnation = 0
            else: stagnation += 1
            if stagnation > 30 and generation < self.nb_generations * 0.85:
                 nb_remplace = int(self.taille_population * 0.33)
                 population = sorted(population, key=lambda x: x.fitness, reverse=True)[:-nb_remplace] + [self.calculer_fitness_amelioree(self.generer_individu_intelligent()) for _ in range(nb_remplace)]
                 random.shuffle(population); stagnation = 0
            self.historique_fitness.append({'generation': generation+1, 'fitness_max': self.meilleure_solution.fitness, 'soutenances_max': self.meilleure_solution.soutenances_planifiees})
        progress_bar.empty()
        return self.decoder_individu(self.meilleure_solution), {
            'generations': self.nb_generations, 'fitness_finale': self.meilleure_solution.fitness,
            'soutenances_planifiees': self.meilleure_solution.soutenances_planifiees, 'conflits': self.meilleure_solution.conflits,
            'historique': self.historique_fitness
        }
    def _stats_vides(self): return {'generations': 0, 'fitness_finale': 0, 'soutenances_planifiees': 0, 'conflits': 0, 'historique': []}
    def selection_tournament(self, population, k=3): return max(random.sample(population, min(k, len(population))), key=lambda x: x.fitness)
    def decoder_individu(self, individu):
        planning = []
        jurys_occupes = defaultdict(set); creneaux_salles = set()
        for idx_etu, idx_creneau in enumerate(individu.genes):
            if idx_creneau == -1 or idx_creneau >= len(self.creneaux): continue
            etu = self.planificateur.etudiants[idx_etu]; creneau = self.creneaux[idx_creneau]
            tuteur = etu["Tuteur"]
            moment = creneau['moment']; id_c = creneau['id']
            if id_c in creneaux_salles or tuteur in jurys_occupes[moment]: continue
            co_jurys = self.planificateur.trouver_co_jurys_disponibles(tuteur, creneau['jour'], creneau['heure'])
            co_jury = next((cj for cj in co_jurys if cj not in jurys_occupes[moment]), None)
            if co_jury:
                planning.append({"Étudiant": f"{etu['Prénom']} {etu['Nom']}", "Pays": etu['Pays'], "Tuteur": tuteur, "Co-jury": co_jury, "Jour": creneau['jour'], "Créneau": creneau['heure'], "Salle": creneau['salle'], "Début": creneau['datetime_debut'], "Fin": creneau['datetime_fin']})
                creneaux_salles.add(id_c); jurys_occupes[moment].update([tuteur, co_jury])
        return planning

class PlanificationOptimiseeV2:
    def __init__(self, etudiants, co_jurys, dates, disponibilites, nb_salles, duree):
        self.etudiants = etudiants; self.co_jurys = co_jurys; self.dates = dates
        self.disponibilites = disponibilites; self.nb_salles = nb_salles; self.duree = duree
        self.tuteurs_referents = list(set([e["Tuteur"] for e in self.etudiants if "Tuteur" in e])) if self.etudiants else []
        self.tous_jurys = list(set(self.tuteurs_referents + self.co_jurys))
        self.charge_jurys_tuteur = {j: 0 for j in self.tous_jurys}; self.charge_jurys_cojury = {j: 0 for j in self.tous_jurys}; self.charge_jurys_total = {j: 0 for j in self.tous_jurys}

    def generer_creneaux_uniques(self): 
        creneaux = []; id_c = 0
        if not self.dates: return []
        for jour_obj in self.dates:
            jour_str = jour_obj.strftime("%A %d/%m/%Y")
            for p in [("08:00", "12:10"), ("14:00", "18:10")]: 
                try: start = datetime.combine(jour_obj, datetime.strptime(p[0], "%H:%M").time()); end = datetime.combine(jour_obj, datetime.strptime(p[1], "%H:%M").time())
                except ValueError: continue
                curr = start
                while curr + timedelta(minutes=self.duree) <= end:
                    fin = curr + timedelta(minutes=self.duree)
                    heure = f"{curr.strftime('%H:%M')} - {fin.strftime('%H:%M')}"
                    for s in range(1, self.nb_salles + 1):
                        creneaux.append({'id': id_c, 'jour': jour_str, 'heure': heure, 'salle': f"Salle {s}", 'datetime_debut': curr, 'datetime_fin': fin, 'moment': f"{jour_str}_{heure}"})
                        id_c += 1
                    curr = fin
        return creneaux

    def est_disponible(self, personne, jour, heure): return self.disponibilites.get(personne, {}).get(f"{jour} | {heure}", False)
    def trouver_co_jurys_disponibles(self, tuteur, jour, heure):
        cands = [j for j in self.tous_jurys if j != tuteur and self.est_disponible(j, jour, heure)]
        cands.sort(key=lambda x: (-(self.charge_jurys_tuteur.get(x, 0) - self.charge_jurys_cojury.get(x, 0)), self.charge_jurys_total.get(x, 0)))
        return cands

    def optimiser_planning_ameliore(self):
        creneaux = self.generer_creneaux_uniques(); planning = []; occ = set(); jurys_occ = defaultdict(set)
        etus = self.etudiants.copy(); random.shuffle(etus); fail = 0
        for etu in etus:
            tuteur = etu["Tuteur"]; placed = False
            random.shuffle(creneaux)
            for c in creneaux:
                if c['id'] in occ or tuteur in jurys_occ[c['moment']]: continue
                if not self.est_disponible(tuteur, c['jour'], c['heure']): continue
                co_jurys = self.trouver_co_jurys_disponibles(tuteur, c['jour'], c['heure'])
                cj = next((j for j in co_jurys if j not in jurys_occ[c['moment']]), None)
                if cj:
                    planning.append({"Étudiant": f"{etu['Prénom']} {etu['Nom']}", "Pays": etu['Pays'], "Tuteur": tuteur, "Co-jury": cj, "Jour": c['jour'], "Créneau": c['heure'], "Salle": c['salle'], "Début": c['datetime_debut'], "Fin": c['datetime_fin']})
                    occ.add(c['id']); jurys_occ[c['moment']].update([tuteur, cj]); placed = True; break
            if not placed: fail += 1
        return planning, fail

    def optimiser_avec_genetique(self, utiliser_genetique_ui=False, **params): 
        p_class, f_class = self.optimiser_planning_ameliore()
        if not utiliser_genetique_ui and (not self.etudiants or len(p_class)/len(self.etudiants) > 0.85): return p_class, f_class, None
        ag = AlgorithmeGenetique(self, **params)
        p_gen, stats = ag.evoluer()
        if len(p_gen) >= len(p_class): return p_gen, len(self.etudiants)-len(p_gen), stats
        return p_class, f_class, stats

    def verifier_conflits(self, planning): return [] 

# --- Fonctions d'import ---
def importer_etudiants_csv(uploaded_file):
    etudiants_list = []
    uploaded_file.seek(0)
    
    # 1. Tentative de lecture flexible
    try:
        # 'python' engine est plus robuste pour les séparateurs
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', engine='python', on_bad_lines='skip')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1', engine='python', on_bad_lines='skip')
    except Exception as e:
        return [], f"Erreur technique lecture CSV: {str(e)}"

    # 2. Nettoyage des noms de colonnes (espaces, majuscules)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 3. Recherche intelligente des colonnes (insensible à la casse)
    col_map = {}
    columns_upper = {c.upper(): c for c in df.columns}

    # Recherche Prénom
    for c_up, c_real in columns_upper.items():
        if "PRENOM" in c_up: col_map['Prénom'] = c_real; break
    
    # Recherche Nom (doit contenir NOM mais PAS Prénom)
    for c_up, c_real in columns_upper.items():
        if "NOM" in c_up and "PRENOM" not in c_up and "ENSEIGNANT" not in c_up: col_map['Nom'] = c_real; break
    
    # Recherche Pays (Pays ou Service d'accueil - Pays)
    for c_up, c_real in columns_upper.items():
        if "PAYS" in c_up and "SERVICE" in c_up: # Priorité Service accueil
            col_map['Pays'] = c_real; break
    if 'Pays' not in col_map: # Fallback
        for c_up, c_real in columns_upper.items():
            if c_up == "PAYS": col_map['Pays'] = c_real; break

    # Recherche Tuteur (Enseignant référent)
    for c_up, c_real in columns_upper.items():
        if "ENSEIGNANT" in c_up or "TUTEUR" in c_up: col_map['Tuteur'] = c_real; break

    # Vérification
    required = ['Prénom', 'Nom', 'Pays', 'Tuteur']
    missing = [r for r in required if r not in col_map]
    
    if missing:
        return [], f"Colonnes introuvables malgré analyse : {missing}. Colonnes lues : {list(df.columns)}"

    # 4. Extraction des données
    for _, row in df.iterrows():
        prenom = str(row.get(col_map['Prénom'], '')).strip()
        nom = str(row.get(col_map['Nom'], '')).strip()
        if prenom and nom and prenom.lower() != 'nan':
            etudiants_list.append({
                "Prénom": prenom,
                "Nom": nom,
                "Pays": str(row.get(col_map['Pays'], '')).strip(),
                "Tuteur": str(row.get(col_map['Tuteur'], '')).strip()
            })
            
    return etudiants_list, None

def importer_disponibilites_csv(uploaded_file, horaires_config, tuteurs, cojurys):
    msgs_s, msgs_e, msgs_w = [], [], []
    try: content = uploaded_file.getvalue(); df_csv = pd.read_csv(StringIO(content.decode('utf-8')), sep=';')
    except: df_csv = pd.read_csv(StringIO(content.decode('latin-1')), sep=';')
    
    csv_map = {}; recognized = set(tuteurs + cojurys); treated = set()
    
    for col in df_csv.columns[3:]:
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2})", str(col))
        if match:
            d_csv, h_csv = match.group(1), match.group(2)
            for j_app, c_list in horaires_config.items():
                if d_csv in j_app:
                    for c in c_list:
                        if c.startswith(h_csv): csv_map[col] = f"{j_app} | {c}"; break
    
    if not csv_map: return [], ["Aucune date correspondante trouvée. Vérifiez que les dates de l'étape 5 correspondent EXACTEMENT aux dates du CSV (Année 2026 ?)."], []

    for _, row in df_csv.iterrows():
        nom_brut = str(row[df_csv.columns[0]]).strip()
        if not nom_brut: continue
        best_match, score = None, 0
        for n in recognized:
            s = fuzz.token_sort_ratio(nom_brut.lower(), n.lower())
            if s > score: score, best_match = s, n
        
        if score >= 75:
            if best_match not in st.session_state.disponibilites: st.session_state.disponibilites[best_match] = {}
            for col_csv, key_app in csv_map.items():
                try: st.session_state.disponibilites[best_match][key_app] = bool(int(float(row[col_csv])))
                except: pass
            treated.add(best_match)
        else: msgs_w.append(f"Ignoré: {nom_brut} (Match max: {score}%)")
            
    if treated: msgs_s.append(f"{len(treated)} disponibilités importées.")
    else: msgs_e.append("Aucun enseignant reconnu.")
    return msgs_s, msgs_e, msgs_w

def importer_disponibilites_excel_simple_header(f, h, t, c): return [], ["Utilisez CSV"], []

# --- UI ---
st.sidebar.header("📥 Import Données")
src = st.sidebar.radio("Source Étudiants:", ("Excel", "CSV Ecole"))
if src == "Excel":
    f = st.sidebar.file_uploader("Excel", type=["xlsx"])
    if f: 
        d = pd.read_excel(f, sheet_name=None)
        if "etudiants" in d: st.session_state.etudiants = d["etudiants"].to_dict('records'); st.sidebar.success("OK")
        if "co_jurys" in d: st.session_state.co_jurys = d["co_jurys"]["Nom"].dropna().astype(str).tolist()
else:
    f = st.sidebar.file_uploader("CSV Etu", type=["csv"])
    if f: 
        l, e = importer_etudiants_csv(f)
        if e: st.sidebar.error(e)
        elif l: 
            st.session_state.etudiants = l
            st.sidebar.success(f"{len(l)} Etus")

if st.session_state.etape == "etudiants":
    afficher_navigation(); st.header("1. Étudiants")
    if st.session_state.etudiants: st.dataframe(pd.DataFrame(st.session_state.etudiants), hide_index=True)
    if st.button("Suivant"): st.session_state.etape = "salles"; st.rerun()

elif st.session_state.etape == "salles":
    afficher_navigation(); st.header("2. Salles")
    st.session_state.nb_salles = st.number_input("Nb Salles", 1, 10, st.session_state.nb_salles)
    if st.button("Suivant"): st.session_state.etape = "duree_soutenance"; st.rerun()

elif st.session_state.etape == "duree_soutenance":
    afficher_navigation(); st.header("3. Durée")
    st.session_state.duree_soutenance = st.number_input("Durée (min)", 30, 120, st.session_state.duree_soutenance)
    st.info("Pour le CSV fourni : Mettre 50 min.")
    if st.button("Suivant"): st.session_state.etape = "co_jury"; st.rerun()

elif st.session_state.etape == "co_jury":
    afficher_navigation(); st.header("4. Co-jurys")
    if st.session_state.co_jurys: st.write(st.session_state.co_jurys)
    else: st.info("Aucun co-jury spécifique ajouté (utilisation des tuteurs entre eux).")
    if st.button("Suivant"): st.session_state.etape = "dates"; st.rerun()

elif st.session_state.etape == "dates":
    afficher_navigation(); st.header("5. Dates")
    st.warning("ATTENTION : Vérifiez l'année ! (ex: 2026)")
    nb = st.number_input("Nb Jours", 1, 10, max(2, len(st.session_state.dates_soutenance)))
    d_ui = []
    cols = st.columns(4)
    for i in range(nb):
        val = st.session_state.dates_soutenance[i] if i < len(st.session_state.dates_soutenance) else datetime(2026, 1, 26).date() + timedelta(days=i)
        d_ui.append(cols[i%4].date_input(f"Jour {i+1}", val))
    if st.button("Valider"): st.session_state.dates_soutenance = d_ui; st.session_state.etape = "disponibilites"; st.rerun()

elif st.session_state.etape == "disponibilites":
    afficher_navigation(); st.header("6. Créneaux")
    if st.button("Générer les créneaux"):
        opt = PlanificationOptimiseeV2([],[],st.session_state.dates_soutenance,{},1,st.session_state.duree_soutenance)
        raw = opt.generer_creneaux_uniques()
        h = defaultdict(list)
        for c in raw: 
            if c['heure'] not in h[c['jour']]: h[c['jour']].append(c['heure'])
        for k in h: h[k].sort()
        st.session_state.horaires_par_jour = dict(h)
        st.success(f"Créneaux générés pour {len(h)} jours.")
        st.session_state.etape = "disponibilites_selection"; st.rerun()

elif st.session_state.etape == "disponibilites_selection":
    afficher_navigation(); st.header("7. Disponibilités")
    f = st.file_uploader("CSV Disponibilités", type=["csv"])
    if f:
        tuteurs = list(set([e["Tuteur"] for e in st.session_state.etudiants]))
        s, e, w = importer_disponibilites_csv(f, st.session_state.horaires_par_jour, tuteurs, st.session_state.co_jurys)
        for m in s: st.success(m)
        for m in e: st.error(m)
        with st.expander("Warnings"): 
            for m in w: st.write(m)
    
    if st.button("Suivant"): st.session_state.etape = "generation"; st.rerun()

elif st.session_state.etape == "generation":
    afficher_navigation(); st.header("8. Génération")
    
    st.subheader("🔍 Outil de Diagnostic")
    if st.button("Diagnostiquer les données"):
        st.write("---")
        # 1. Check Dates
        st.write("**1. Vérification des Dates :**")
        if not st.session_state.horaires_par_jour: st.error("Aucun créneau généré.")
        else: st.success(f"Jours générés : {list(st.session_state.horaires_par_jour.keys())}")
        
        # 2. Check Dispos
        st.write("**2. Vérification des Disponibilités :**")
        nb_dispos_true = 0
        jurys_avec_dispos = []
        for j, d in st.session_state.disponibilites.items():
            true_vals = [k for k, v in d.items() if v]
            if true_vals: 
                nb_dispos_true += len(true_vals)
                jurys_avec_dispos.append(j)
        
        st.write(f"Total créneaux marqués 'Disponibles' : {nb_dispos_true}")
        if nb_dispos_true == 0:
            st.error("❌ AUCUNE DISPONIBILITÉ 'VRAIE' TROUVÉE. Le planning échouera forcément.")
            st.info("Conseil : Vérifiez que les dates de l'étape 5 (Année 2026 ?) correspondent exactement aux colonnes de votre CSV.")
        else:
            st.success(f"✅ {len(jurys_avec_dispos)} jurys ont au moins 1 créneau disponible.")
            with st.expander("Voir qui est disponible"):
                st.write(jurys_avec_dispos)

    st.divider()
    if st.button("Lancer"):
        opt = PlanificationOptimiseeV2(st.session_state.etudiants, st.session_state.co_jurys, st.session_state.dates_soutenance, st.session_state.disponibilites, st.session_state.nb_salles, st.session_state.duree_soutenance)
        p, fail, stats = opt.optimiser_avec_genetique(True)
        st.session_state.planning_final = p
        if p:
            st.success(f"{len(p)} planifiés !")
            st.dataframe(pd.DataFrame(p))
        else:
            st.error("Echec total. Utilisez le bouton Diagnostic ci-dessus.")
