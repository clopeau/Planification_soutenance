C'est une erreur de ma part, je m'en excuse. Le message d'erreur NameError indique que la variable df n'est pas définie à cet endroit précis du code.

Dans la fonction importer_disponibilites_csv, j'ai nommé le tableau de données df_csv lors de la lecture, mais j'ai écrit par erreur df.iterrows() (le nom standard) au lieu de df_csv.iterrows() dans la boucle.

Voici le code corrigé. Vous pouvez remplacer tout le contenu de votre fichier app.py par celui-ci.

code
Python
download
content_copy
expand_less
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
if "etape" not in st.session_state:
    st.session_state.etape = "etudiants"
if "etudiants" not in st.session_state:
    st.session_state.etudiants = []
if "co_jurys" not in st.session_state:
    st.session_state.co_jurys = []
if "dates_soutenance" not in st.session_state:
    st.session_state.dates_soutenance = []
if "disponibilites" not in st.session_state:
    st.session_state.disponibilites = {}
if "planning_final" not in st.session_state:
    st.session_state.planning_final = []
if "nb_salles" not in st.session_state:
    st.session_state.nb_salles = 2
if "duree_soutenance" not in st.session_state:
    st.session_state.duree_soutenance = 50 
if "horaires_par_jour" not in st.session_state: 
    st.session_state.horaires_par_jour = {}


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
        if not self.planificateur.etudiants or not self.creneaux:
            return creneaux_valides
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
        if self.nb_etudiants == 0:
            return Individu(genes=genes)
            
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

                if creneau_cand['id'] in creneaux_occupes_par_id:
                    continue
                
                tuteur_etu = self.planificateur.etudiants[idx_etu]["Tuteur"]
                moment_cand = creneau_cand['moment']
                
                if tuteur_etu in jurys_occupes_par_moment[moment_cand]:
                    continue

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
        max_difference_toleree_strict = 0 

        for jury, counts in roles_par_jury.items():
            if jury in personnes_eligibles_balance:
                difference = abs(counts['tuteur'] - counts['cojury'])
                if difference > max_difference_toleree_strict: 
                    contrainte_parite_roles_violee_strict = True
                
                penalite_balance_roles += (difference ** 2) * 20 
                if difference == 0:
                    score_balance_roles += 100 
                elif difference == 1:
                    score_balance_roles += 30


        fitness_base = (
                taux_planification * 912 +
                max(0, (nb_soutenances - (nb_total_etudiants * 0.75))) * 50 + 
                equilibrage * 30 +  
                bonus_alternance * 15 + 
                score_balance_roles * 1.5 - 
                (nb_total_etudiants - nb_soutenances) * 150 - 
                penalite_balance_roles 
        )
        
        fitness_finale = fitness_base

        if total_conflits > 0:
            fitness_finale = -500_000 - (total_conflits * 1000) 
        elif contrainte_parite_roles_violee_strict: 
            fitness_finale = -1_000_000 - penalite_balance_roles
        else: 
            if nb_soutenances >= nb_total_etudiants * 0.9: 
                fitness_finale += 2000 

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

            if cle_salle_moment in creneaux_salle_utilises:
                conflits_salle += 1
            creneaux_salle_utilises[cle_salle_moment] = True

            tuteur = soutenance['Tuteur']
            co_jury = soutenance['Co-jury']

            if tuteur in jurys_occupes_moment[moment_sout]:
                conflits_jury += 1
            else:
                jurys_occupes_moment[moment_sout].add(tuteur)
            
            if co_jury in jurys_occupes_moment[moment_sout]:
                conflits_jury += 1
            else:
                jurys_occupes_moment[moment_sout].add(co_jury)
                
        return conflits_salle, conflits_jury

    def _calculer_equilibrage_charge(self, planning):
        if not planning: return 0.0 
        charges = defaultdict(int)
        for soutenance in planning:
            charges[soutenance['Tuteur']] += 1
            charges[soutenance['Co-jury']] += 1
        
        if not charges or len(charges) <= 1 : return 10.0 
        
        valeurs_charges = np.array(list(charges.values()))
        variance = np.var(valeurs_charges) 
        return max(0.0, 10.0 - np.sqrt(variance)) 

    def _calculer_bonus_alternance(self, planning): 
        jurys_par_periode = {'matin': defaultdict(set), 'apres_midi': defaultdict(set)} 
        for soutenance in planning:
            jour = soutenance['Jour']
            debut_heure = soutenance['Début'].hour
            periode = 'matin' if debut_heure < 13 else 'apres_midi'
            jurys_par_periode[periode][jour].add(soutenance['Tuteur'])
            jurys_par_periode[periode][jour].add(soutenance['Co-jury'])
        total_jurys_alternant = 0
        jours_concernes = set(jurys_par_periode['matin'].keys()) | set(jurys_par_periode['apres_midi'].keys())
        for jour_alternance in jours_concernes:
            jurys_matin_ce_jour = jurys_par_periode['matin'].get(jour_alternance, set())
            jurys_aprem_ce_jour = jurys_par_periode['apres_midi'].get(jour_alternance, set())
            jurys_alternant_ce_jour = jurys_matin_ce_jour & jurys_aprem_ce_jour
            total_jurys_alternant += len(jurys_alternant_ce_jour)
        return float(total_jurys_alternant * 1.0)

    def croisement_intelligent(self, parent1: Individu, parent2: Individu) -> Tuple[Individu, Individu]:
        len_genes = len(parent1.genes)
        if len_genes == 0: 
             return Individu(genes=[]), Individu(genes=[])

        enfant1_genes_final = [-1] * len_genes
        enfant2_genes_final = [-1] * len_genes
        
        point_croisement = random.randint(1, len_genes - 1) if len_genes > 1 else 0

        for i in range(point_croisement):
            enfant1_genes_final[i] = parent1.genes[i]
            enfant2_genes_final[i] = parent2.genes[i]
        
        creneaux_pris_e1 = set(g for g in enfant1_genes_final[:point_croisement] if g != -1)
        creneaux_pris_e2 = set(g for g in enfant2_genes_final[:point_croisement] if g != -1)

        for i in range(point_croisement, len_genes):
            # Enfant 1
            gene_candidat_p2 = parent2.genes[i]
            if gene_candidat_p2 != -1 and gene_candidat_p2 not in creneaux_pris_e1 and \
               (i in self.creneaux_valides_par_etudiant and gene_candidat_p2 in self.creneaux_valides_par_etudiant[i]):
                enfant1_genes_final[i] = gene_candidat_p2
                creneaux_pris_e1.add(gene_candidat_p2)
            else: 
                gene_candidat_p1 = parent1.genes[i]
                if gene_candidat_p1 != -1 and gene_candidat_p1 not in creneaux_pris_e1 and \
                   (i in self.creneaux_valides_par_etudiant and gene_candidat_p1 in self.creneaux_valides_par_etudiant[i]):
                    enfant1_genes_final[i] = gene_candidat_p1
                    creneaux_pris_e1.add(gene_candidat_p1)

            # Enfant 2
            gene_candidat_p1_e2 = parent1.genes[i]
            if gene_candidat_p1_e2 != -1 and gene_candidat_p1_e2 not in creneaux_pris_e2 and \
               (i in self.creneaux_valides_par_etudiant and gene_candidat_p1_e2 in self.creneaux_valides_par_etudiant[i]):
                enfant2_genes_final[i] = gene_candidat_p1_e2
                creneaux_pris_e2.add(gene_candidat_p1_e2)
            else: 
                gene_candidat_p2_e2 = parent2.genes[i]
                if gene_candidat_p2_e2 != -1 and gene_candidat_p2_e2 not in creneaux_pris_e2 and \
                   (i in self.creneaux_valides_par_etudiant and gene_candidat_p2_e2 in self.creneaux_valides_par_etudiant[i]):
                    enfant2_genes_final[i] = gene_candidat_p2_e2
                    creneaux_pris_e2.add(gene_candidat_p2_e2)
                    
        return Individu(genes=enfant1_genes_final), Individu(genes=enfant2_genes_final)


    def mutation_adaptative(self, individu: Individu) -> Individu:
        if not individu.genes: return individu 

        for i in range(len(individu.genes)):
            if random.random() < self.taux_mutation:
                if i >= self.nb_etudiants: continue 
                
                creneaux_possibles_pour_cet_etudiant = self.creneaux_valides_par_etudiant.get(i, [])
                if not creneaux_possibles_pour_cet_etudiant: continue

                gene_actuel_de_i = individu.genes[i]
                creneaux_utilises_par_autres = set(
                    individu.genes[j] for j in range(len(individu.genes)) if j != i and individu.genes[j] != -1
                )
                options_de_mutation_libres = [
                    c for c in creneaux_possibles_pour_cet_etudiant if c not in creneaux_utilises_par_autres
                ]

                if not options_de_mutation_libres:
                    if gene_actuel_de_i != -1 and gene_actuel_de_i in creneaux_utilises_par_autres :
                         individu.genes[i] = -1
                    continue

                if gene_actuel_de_i == -1: 
                    individu.genes[i] = random.choice(options_de_mutation_libres)
                else: 
                    nouveaux_creneaux_libres = [c for c in options_de_mutation_libres if c != gene_actuel_de_i]
                    if nouveaux_creneaux_libres:
                        individu.genes[i] = random.choice(nouveaux_creneaux_libres)
                    elif gene_actuel_de_i in options_de_mutation_libres : 
                        pass 
                    else: 
                        individu.genes[i] = -1
        return individu

    def evoluer(self) -> Tuple[List[Dict], Dict]:
        population = []
        if self.nb_etudiants == 0:
             st.sidebar.warning("AG: Aucun étudiant, arrêt.")
             return [], self._stats_vides()

        for _ in range(self.taille_population):
            individu = self.generer_individu_intelligent()
            population.append(self.calculer_fitness_amelioree(individu))

        if not population or not any(ind.genes != [-1]*self.nb_etudiants for ind in population) :
            st.sidebar.warning("AG: Population initiale difficile à construire.")
            self.meilleure_solution = Individu(genes=[-1]*self.nb_etudiants)

        if population:
            self.meilleure_solution = max(population, key=lambda x: x.fitness, default=self.meilleure_solution)
        else: 
            return [], self._stats_vides()

        stagnation = 0
        progress_bar = st.sidebar.progress(0)
        
        for generation in range(self.nb_generations):
            if not population: break
            progress_bar.progress((generation + 1) / self.nb_generations)

            nouvelle_population = []
            population_triee = sorted(population, key=lambda x: x.fitness, reverse=True)
            
            elite_size = max(1, int(self.taille_population * 0.1)) 
            nouvelle_population.extend(population_triee[:elite_size])

            while len(nouvelle_population) < self.taille_population:
                if random.random() < self.taux_croisement and len(population) >= 2 :
                    parent1 = self.selection_tournament(population, k=5)
                    parent2 = self.selection_tournament(population, k=5)
                    if parent1.genes and parent2.genes: 
                        enfant1, enfant2 = self.croisement_intelligent(parent1, parent2)
                        nouvelle_population.append(self.mutation_adaptative(enfant1))
                        if len(nouvelle_population) < self.taille_population:
                             nouvelle_population.append(self.mutation_adaptative(enfant2))
                else: 
                    if population:
                        individu_choisi = random.choice(population)
                        nouvelle_population.append(self.mutation_adaptative(individu_choisi))
                    else: break 
            
            nouvelle_population = nouvelle_population[:self.taille_population] 
            population = [self.calculer_fitness_amelioree(ind) for ind in nouvelle_population]

            if not population: break 

            meilleur_actuel_gen = max(population, key=lambda x: x.fitness, default=self.meilleure_solution)

            if meilleur_actuel_gen.fitness > self.meilleure_solution.fitness:
                self.meilleure_solution = meilleur_actuel_gen
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > max(30, self.nb_generations * 0.15) and generation < self.nb_generations * 0.85: 
                nb_a_remplacer = int(self.taille_population * 0.33) 
                population_meilleurs_conserves = sorted(population, key=lambda x: x.fitness, reverse=True)[:-nb_a_remplacer]
                nouveaux = [self.calculer_fitness_amelioree(self.generer_individu_intelligent()) for _ in range(nb_a_remplacer)]
                population = population_meilleurs_conserves + nouveaux
                random.shuffle(population)
                stagnation = 0

            fitness_moyenne_gen = np.mean([ind.fitness for ind in population]) if population else self.meilleure_solution.fitness
            self.historique_fitness.append({
                'generation': generation+1,
                'fitness_max': self.meilleure_solution.fitness, 
                'fitness_moyenne': fitness_moyenne_gen,
                'soutenances_max': self.meilleure_solution.soutenances_planifiees,
                'conflits_min': self.meilleure_solution.conflits
            })

        progress_bar.empty()
        planning_final = self.decoder_individu(self.meilleure_solution)
        taux_reussite_final = (self.meilleure_solution.soutenances_planifiees / self.nb_etudiants) if self.nb_etudiants > 0 else 0
        statistiques = {
            'generations': self.nb_generations,
            'fitness_finale': self.meilleure_solution.fitness,
            'soutenances_planifiees': self.meilleure_solution.soutenances_planifiees,
            'conflits': self.meilleure_solution.conflits,
            'taux_reussite': taux_reussite_final,
            'historique': self.historique_fitness
        }
        return planning_final, statistiques
    
    def _stats_vides(self):
        return {'generations': 0, 'fitness_finale': 0, 'soutenances_planifiees': 0, 
                'conflits': 0, 'taux_reussite': 0, 'historique': []}


    def selection_tournament(self, population: List[Individu], k=3) -> Individu:
        if not population:
            return Individu(genes=[-1]*self.nb_etudiants if self.nb_etudiants > 0 else [])
        k_valide = min(k, len(population))
        if k_valide == 0 : return population[0]
        participants = random.sample(population, k_valide)
        return max(participants, key=lambda x: x.fitness)

    def decoder_individu(self, individu: Individu) -> List[Dict]:
        planning = []
        if not individu or not individu.genes or not self.creneaux or not self.planificateur.etudiants:
            return planning

        jurys_occupes_decode_moment = defaultdict(set) 
        creneaux_salles_decode_ids = set()

        for idx_etu, idx_creneau_gene in enumerate(individu.genes):
            if idx_creneau_gene == -1 or idx_creneau_gene >= len(self.creneaux):
                continue
            if idx_etu >= len(self.planificateur.etudiants): continue
            
            etudiant_obj = self.planificateur.etudiants[idx_etu]
            creneau_obj_decode = self.creneaux[idx_creneau_gene]
            tuteur_principal = etudiant_obj["Tuteur"]
            
            moment_str_decode = creneau_obj_decode['moment']
            id_creneau_salle_decode = creneau_obj_decode['id']

            if id_creneau_salle_decode in creneaux_salles_decode_ids:
                continue 
            if tuteur_principal in jurys_occupes_decode_moment[moment_str_decode]:
                continue 

            co_jurys_possibles_decode = self.planificateur.trouver_co_jurys_disponibles(
                tuteur_principal, creneau_obj_decode['jour'], creneau_obj_decode['heure']
            )
            
            co_jury_final_choisi = None
            for cj_cand_decode in co_jurys_possibles_decode:
                if cj_cand_decode not in jurys_occupes_decode_moment[moment_str_decode]:
                    co_jury_final_choisi = cj_cand_decode
                    break
            
            if co_jury_final_choisi:
                planning.append({
                    "Étudiant": f"{etudiant_obj['Prénom']} {etudiant_obj['Nom']}",
                    "Pays": etudiant_obj['Pays'], "Tuteur": tuteur_principal, "Co-jury": co_jury_final_choisi,
                    "Jour": creneau_obj_decode['jour'], "Créneau": creneau_obj_decode['heure'], 
                    "Salle": creneau_obj_decode['salle'],
                    "Début": creneau_obj_decode['datetime_debut'], "Fin": creneau_obj_decode['datetime_fin']
                })
                creneaux_salles_decode_ids.add(id_creneau_salle_decode)
                jurys_occupes_decode_moment[moment_str_decode].add(tuteur_principal)
                jurys_occupes_decode_moment[moment_str_decode].add(co_jury_final_choisi)
        return planning


class PlanificationOptimiseeV2:
    def __init__(self, etudiants, co_jurys, dates, disponibilites, nb_salles, duree):
        self.etudiants = etudiants if etudiants else []
        self.co_jurys = co_jurys if co_jurys else []
        self.dates = dates if dates else []
        self.disponibilites = disponibilites if disponibilites else {}
        self.nb_salles = nb_salles
        self.duree = duree

        self.tuteurs_referents = list(set([e["Tuteur"] for e in self.etudiants if "Tuteur" in e])) if self.etudiants else []
        self.tous_jurys = list(set(self.tuteurs_referents + self.co_jurys))
        
        self.charge_jurys_tuteur = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_cojury = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_total = {jury: 0 for jury in self.tous_jurys}

    def generer_creneaux_uniques(self): 
        creneaux = []
        creneau_id = 0
        if not self.dates: return []

        for jour_obj in self.dates:
            jour_str_app = jour_obj.strftime("%A %d/%m/%Y")
            # Périodes alignées sur le CSV (Matin 8h-12h10, Aprem 14h-18h10)
            for periode in [("08:00", "12:10"), ("14:00", "18:10")]: 
                try:
                    debut_dt_obj = datetime.strptime(periode[0], "%H:%M").time()
                    fin_dt_obj = datetime.strptime(periode[1], "%H:%M").time()
                except ValueError: continue 

                current_dt = datetime.combine(jour_obj, debut_dt_obj)
                end_dt = datetime.combine(jour_obj, fin_dt_obj)

                while current_dt + timedelta(minutes=self.duree) <= end_dt:
                    fin_creneau_dt = current_dt + timedelta(minutes=self.duree)
                    heure_str_app = f"{current_dt.strftime('%H:%M')} - {fin_creneau_dt.strftime('%H:%M')}"
                    for salle_num in range(1, self.nb_salles + 1):
                        creneaux.append({
                            'id': creneau_id, 'jour': jour_str_app, 'heure': heure_str_app,
                            'salle': f"Salle {salle_num}", 'datetime_debut': current_dt,
                            'datetime_fin': fin_creneau_dt, 'moment': f"{jour_str_app}_{heure_str_app}"
                        })
                        creneau_id += 1
                    current_dt = fin_creneau_dt
        return creneaux

    def est_disponible(self, personne, jour_str_app, heure_str_app): 
        key = f"{jour_str_app} | {heure_str_app}"
        return self.disponibilites.get(personne, {}).get(key, False)

    def trouver_co_jurys_disponibles(self, tuteur_referent, jour_str_app, heure_str_app): 
        co_jurys_dispo = []
        for jury_cand in self.tous_jurys:
            if jury_cand != tuteur_referent and self.est_disponible(jury_cand, jour_str_app, heure_str_app):
                co_jurys_dispo.append(jury_cand)
        
        def sort_key_balance_cojury(jury_c):
            diff_roles = self.charge_jurys_tuteur.get(jury_c, 0) - self.charge_jurys_cojury.get(jury_c, 0)
            charge_tot = self.charge_jurys_total.get(jury_c, 0)
            return (-diff_roles, charge_tot)

        co_jurys_dispo.sort(key=sort_key_balance_cojury)
        return co_jurys_dispo

    def optimiser_planning_ameliore(self): 
        self.charge_jurys_tuteur = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_cojury = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_total = {jury: 0 for jury in self.tous_jurys}

        creneaux = self.generer_creneaux_uniques()
        planning = []
        creneaux_occupes_ids = set() 
        jurys_par_moment_app = defaultdict(set) 
        
        if not self.etudiants: return [], 0 
        
        etudiants_melanges = self.etudiants.copy()
        random.shuffle(etudiants_melanges)
        non_planifies_count = 0

        for etudiant_obj_classique in etudiants_melanges:
            tuteur_ref_classique = etudiant_obj_classique["Tuteur"]
            soutenance_planifiee_etu_classique = False
            creneaux_melanges_classique = creneaux.copy()
            random.shuffle(creneaux_melanges_classique)

            for creneau_obj_classique in creneaux_melanges_classique:
                if creneau_obj_classique['id'] in creneaux_occupes_ids: continue
                if not self.est_disponible(tuteur_ref_classique, creneau_obj_classique['jour'], creneau_obj_classique['heure']): continue
                if tuteur_ref_classique in jurys_par_moment_app[creneau_obj_classique['moment']]: continue

                co_jurys_possibles_classique = self.trouver_co_jurys_disponibles( 
                    tuteur_ref_classique, creneau_obj_classique['jour'], creneau_obj_classique['heure']
                )
                co_jury_choisi_classique = None
                for cj_cand_classique in co_jurys_possibles_classique:
                    if cj_cand_classique not in jurys_par_moment_app[creneau_obj_classique['moment']]:
                        co_jury_choisi_classique = cj_cand_classique
                        break
                
                if co_jury_choisi_classique:
                    planning.append({
                        "Étudiant": f"{etudiant_obj_classique['Prénom']} {etudiant_obj_classique['Nom']}",
                        "Pays": etudiant_obj_classique['Pays'], "Tuteur": tuteur_ref_classique, "Co-jury": co_jury_choisi_classique,
                        "Jour": creneau_obj_classique['jour'], "Créneau": creneau_obj_classique['heure'], "Salle": creneau_obj_classique['salle'],
                        "Début": creneau_obj_classique['datetime_debut'], "Fin": creneau_obj_classique['datetime_fin']
                    })
                    creneaux_occupes_ids.add(creneau_obj_classique['id'])
                    jurys_par_moment_app[creneau_obj_classique['moment']].add(tuteur_ref_classique)
                    jurys_par_moment_app[creneau_obj_classique['moment']].add(co_jury_choisi_classique)
                    
                    self.charge_jurys_tuteur[tuteur_ref_classique] += 1
                    self.charge_jurys_cojury[co_jury_choisi_classique] += 1
                    self.charge_jurys_total[tuteur_ref_classique] +=1
                    self.charge_jurys_total[co_jury_choisi_classique] +=1
                    
                    soutenance_planifiee_etu_classique = True
                    break
            
            if not soutenance_planifiee_etu_classique:
                non_planifies_count += 1
        
        return planning, non_planifies_count

    def optimiser_avec_genetique(self, utiliser_genetique_ui=False, **params_genetique_ui): 
        planning_classique, non_planifies_classique = self.optimiser_planning_ameliore()
        nb_etudiants_total = len(self.etudiants) if self.etudiants else 0
        taux_reussite_classique = (len(planning_classique) / nb_etudiants_total) if nb_etudiants_total > 0 else 0.0

        run_ag = False
        if utiliser_genetique_ui:
            run_ag = True
        elif nb_etudiants_total > 0 and taux_reussite_classique < 0.85 and planning_classique :
            run_ag = True
        elif nb_etudiants_total > 0 and not planning_classique:
            run_ag = True

        if run_ag:
            st.info("🧬 Lancement de l'optimisation génétique...")
            config_ag = { 
                'taille_population': 80, 'nb_generations': 300, 
                'taux_mutation': 0.15, 'taux_croisement': 0.85, 
                **params_genetique_ui 
            }
            config_ag['taille_population'] = max(20, config_ag['taille_population']) 
            config_ag['nb_generations'] = max(20, config_ag['nb_generations'])     

            if not self.etudiants:
                st.warning("AG non lancé : aucun étudiant à planifier.")
                return planning_classique, non_planifies_classique, None

            ag_instance = AlgorithmeGenetique(self, **config_ag)
            planning_genetique, stats_ag = ag_instance.evoluer()
            
            if stats_ag is None: stats_ag = ag_instance._stats_vides()
            stats_ag['amelioration_valeur'] = 0 

            classique_score_comparaison = len(planning_classique)
            genetique_score_comparaison = len(planning_genetique)

            if genetique_score_comparaison > classique_score_comparaison:
                st.success(f"✅ AG a amélioré le nombre de soutenances: {len(planning_genetique)} vs {len(planning_classique)} (classique)")
                stats_ag['amelioration_valeur'] = len(planning_genetique) - len(planning_classique)
                return planning_genetique, nb_etudiants_total - len(planning_genetique), stats_ag
            elif genetique_score_comparaison == classique_score_comparaison and stats_ag.get('fitness_finale', -float('inf')) > -100000 : 
                 st.info(f"AG a planifié autant ({len(planning_genetique)}). Résultat de l'AG conservé (Fitness: {stats_ag.get('fitness_finale', 0.0):.0f}).")
                 return planning_genetique, nb_etudiants_total - len(planning_genetique), stats_ag
            else:
                st.info(f"ℹ️ AG n'a pas amélioré ({len(planning_genetique)} planifiées). Résultat classique ({len(planning_classique)}) conservé.")
                return planning_classique, non_planifies_classique, stats_ag 
        
        return planning_classique, non_planifies_classique, None


    def verifier_conflits(self, planning): 
        conflits_messages = []
        creneaux_salles_occupes = defaultdict(list) 
        jurys_moments_occupes = defaultdict(list)   

        for idx, soutenance in enumerate(planning):
            cle_moment_salle = f"{soutenance['Jour']}_{soutenance['Créneau']}_{soutenance['Salle']}"
            cle_moment = f"{soutenance['Jour']}_{soutenance['Créneau']}"

            creneaux_salles_occupes[cle_moment_salle].append(soutenance['Étudiant'])
            jurys_moments_occupes[cle_moment].extend([
                (soutenance['Tuteur'], soutenance['Étudiant']),
                (soutenance['Co-jury'], soutenance['Étudiant'])
            ])

        for moment_salle, etudiants_conflit in creneaux_salles_occupes.items():
            if len(etudiants_conflit) > 1:
                conflits_messages.append(f"Salle: {moment_salle} surbookée ({', '.join(etudiants_conflit)})")

        for moment, jurys_affectes_avec_etu in jurys_moments_occupes.items():
            compteur_jurys_moment = defaultdict(list) 
            for jury, etudiant_associe in jurys_affectes_avec_etu:
                compteur_jurys_moment[jury].append(etudiant_associe)
            
            for jury, etudiants_pour_jury in compteur_jurys_moment.items():
                if len(etudiants_pour_jury) > 1:
                    etudiants_str = " et ".join(f"'{etu}'" for etu in etudiants_pour_jury)
                    conflits_messages.append(f"Jury: {jury} à {moment} pour {etudiants_str}")
        return conflits_messages

# --- Fonction d'importation ETUDIANTS (CSV) ---
def importer_etudiants_csv(uploaded_file):
    etudiants_list = []
    try:
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1', on_bad_lines='skip')
    except Exception as e:
        return [], f"Erreur lecture CSV: {e}"

    # Nettoyage des colonnes vides (ex: ;;;; à la fin)
    df.dropna(how='all', inplace=True)
    
    required_cols_map = {
        'PRENOM': 'Prénom',
        'NOM': 'Nom',
        'Pays': 'Pays',
        'Enseignant référent (NOM Prénom)': 'Tuteur'
    }
    
    # Vérification sommaire des colonnes
    missing = [c for c in required_cols_map.keys() if c not in df.columns]
    if missing:
        return [], f"Colonnes manquantes dans le CSV: {missing}"

    for _, row in df.iterrows():
        # Extraction et nettoyage
        prenom = str(row.get('PRENOM', '')).strip()
        nom = str(row.get('NOM', '')).strip()
        pays = str(row.get('Pays', '')).strip()
        tuteur = str(row.get('Enseignant référent (NOM Prénom)', '')).strip()
        
        if not prenom or not nom: continue # Ignorer lignes vides

        etudiants_list.append({
            "Prénom": prenom,
            "Nom": nom,
            "Pays": pays,
            "Tuteur": tuteur
        })
        
    return etudiants_list, None


# --- Fonctions d'importation des disponibilités (MODIFIÉE POUR CSV) ---
def importer_disponibilites_csv(uploaded_file, 
                                horaires_par_jour_app_config: Dict[str, List[str]], 
                                tous_tuteurs_app: List[str], 
                                co_jurys_app: List[str],
                                score_matching_seuil=75):
    messages_succes, messages_erreur, messages_warning = [], [], []
    personnes_traitees_import = set()
    personnes_reconnues_app_set = set(tous_tuteurs_app + co_jurys_app)
    
    if not uploaded_file: return [], ["Aucun fichier CSV fourni."], []
    
    content = uploaded_file.getvalue()
    df_csv = None
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            stringio = StringIO(content.decode(enc))
            df_csv = pd.read_csv(stringio, sep=';', header=0, skipinitialspace=True)
            break
        except UnicodeDecodeError:
            continue
            
    if df_csv is None or df_csv.empty: 
        return [], ["Fichier CSV vide ou encodage non supporté."], []

    csv_col_to_app_key = {}
    
    for col_name in df_csv.columns[3:]:
        col_str = str(col_name).strip()
        if not col_str or "Unnamed" in col_str: continue
        
        match = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2})", col_str)
        
        if match:
            date_csv = match.group(1)
            heure_debut_csv = match.group(2)
            
            found_key = None
            for jour_app, creneaux_app in horaires_par_jour_app_config.items():
                if date_csv in jour_app:
                    for creneau in creneaux_app:
                        if creneau.startswith(heure_debut_csv):
                            found_key = f"{jour_app} | {creneau}"
                            break
                if found_key: break
            
            if found_key:
                csv_col_to_app_key[col_name] = found_key

    if not csv_col_to_app_key:
        return [], ["Aucune colonne de date reconnue. Vérifiez que les dates (ex: 26/01/2026) correspondent à celles définies à l'étape 5."], []

    col_nom = df_csv.columns[0] 

    # CORRECTION: Utilisation explicite de df_csv
    for _, row in df_csv.iterrows():
        nom_csv_brut = str(row[col_nom]).strip()
        if not nom_csv_brut or pd.isna(row[col_nom]): continue

        best_match, h_score = None, 0
        for nom_app in personnes_reconnues_app_set:
            # Token Sort Ratio gère l'inversion Nom/Prénom (Emmanuel PERRIN vs PERRIN Emmanuel)
            score = fuzz.token_sort_ratio(nom_csv_brut.lower(), nom_app.lower())
            if score > h_score: h_score, best_match = score, nom_app
        
        nom_final = None
        if h_score >= score_matching_seuil:
            nom_final = best_match
        else:
            messages_warning.append(f"Ignoré: '{nom_csv_brut}' (max match: {h_score}% avec '{best_match}')")
            continue

        personnes_traitees_import.add(nom_final)
        if nom_final not in st.session_state.disponibilites: 
            st.session_state.disponibilites[nom_final] = {}

        for col_csv, app_key in csv_col_to_app_key.items():
            val = row[col_csv]
            try:
                if pd.isna(val): is_dispo = False
                else: is_dispo = bool(int(float(val)))
                st.session_state.disponibilites[nom_final][app_key] = is_dispo
            except ValueError:
                pass 

    if personnes_traitees_import: 
        messages_succes.append(f"Import CSV réussi pour {len(personnes_traitees_import)} jurys.")
    else:
        messages_erreur.append("Aucun jury importé.")

    return messages_succes, messages_erreur, messages_warning

def importer_disponibilites_excel_simple_header(uploaded_file, horaires_par_jour, tuteurs, cojurys, score_matching_seuil=75):
    return [], ["Utilisez le format CSV pour ce fichier spécifique."], []

# --- Interface utilisateur ---

st.sidebar.header("📥 Import Données de Base")
source_option = st.sidebar.radio("Source Étudiants :", ("Excel (Ancien format)", "CSV (Format Ecole)"))

if source_option == "Excel (Ancien format)":
    excel_file_base = st.sidebar.file_uploader("Fichier Excel .xlsx", type=["xlsx"], key="excel_base_uploader_key")
    if excel_file_base:
        try:
            excel_data_base = pd.read_excel(excel_file_base, sheet_name=None)
            if "etudiants" in excel_data_base:
                etu_df = excel_data_base["etudiants"]
                req_cols = {"Nom", "Prénom", "Pays", "Tuteur"}
                if req_cols.issubset(etu_df.columns):
                    st.session_state.etudiants = etu_df[list(req_cols)].to_dict(orient="records")
                    st.sidebar.success(f"{len(st.session_state.etudiants)} étudiants importés.")
            if "co_jurys" in excel_data_base:
                cj_df = excel_data_base["co_jurys"]
                if "Nom" in cj_df.columns:
                    st.session_state.co_jurys = cj_df["Nom"].dropna().astype(str).tolist()
                    st.sidebar.success(f"{len(st.session_state.co_jurys)} co-jurys importés.")
        except Exception as e_imp_base: st.sidebar.error(f"Erreur Excel base: {e_imp_base}")

else: # Import CSV Ecole
    csv_file_etu = st.sidebar.file_uploader("Fichier Étudiants .csv", type=["csv"], key="csv_etu_uploader_key")
    if csv_file_etu:
        etu_list, err = importer_etudiants_csv(csv_file_etu)
        if err:
            st.sidebar.error(err)
        elif etu_list:
            st.session_state.etudiants = etu_list
            st.sidebar.success(f"{len(etu_list)} étudiants importés depuis le CSV.")
            
            # Extraction automatique des co-jurys potentiels si besoin (non présents dans ce CSV spécifique pour la soutenance)
            # Mais on peut vider la liste pour éviter les confusions avec des imports précédents
            # st.session_state.co_jurys = [] 


if st.session_state.etape == "etudiants":
    afficher_navigation()
    st.header(etapes_labels["etudiants"])
    with st.form("ajout_etudiant_form_key"):
        nom, prenom = st.text_input("Nom"), st.text_input("Prénom")
        pays, tuteur = st.text_input("Pays"), st.text_input("Tuteur")
        if st.form_submit_button("Ajouter étudiant") and all([nom, prenom, pays, tuteur]):
            st.session_state.etudiants.append({"Nom": nom, "Prénom": prenom, "Pays": pays, "Tuteur": tuteur})
            st.success(f"Étudiant {prenom} {nom} ajouté."); st.rerun()
    if st.session_state.etudiants:
        st.dataframe(pd.DataFrame(st.session_state.etudiants), use_container_width=True, hide_index=True)
    if st.button("Suivant > Salles", type="primary", key="etu_suivant_btn"):
        if st.session_state.etudiants: st.session_state.etape = "salles"; st.rerun()
        else: st.error("Ajoutez au moins un étudiant.")

elif st.session_state.etape == "salles":
    afficher_navigation()
    st.header(etapes_labels["salles"])
    val_nb_salles = st.session_state.get("nb_salles", 2)
    nb_salles_in = st.number_input("Nombre de salles", 1, 10, val_nb_salles, 1, key="nb_salles_in_key")
    if st.button("Valider > Durée", type="primary", key="salles_val_btn"):
        st.session_state.nb_salles = nb_salles_in
        st.session_state.etape = "duree_soutenance"; st.rerun()

elif st.session_state.etape == "duree_soutenance":
    afficher_navigation()
    st.header(etapes_labels["duree_soutenance"])
    val_duree = st.session_state.get("duree_soutenance", 50)
    duree_in = st.number_input("Durée soutenance (min)", 30, 120, val_duree, 10, key="duree_in_key")
    if st.button("Valider > Co-jurys", type="primary", key="duree_val_btn"):
        st.session_state.duree_soutenance = duree_in
        st.session_state.etape = "co_jury"; st.rerun()

elif st.session_state.etape == "co_jury":
    afficher_navigation()
    st.header(etapes_labels["co_jury"])
    with st.form("ajout_cojury_form_key"):
        nom_cj_in = st.text_input("Nom du co-jury")
        if st.form_submit_button("Ajouter co-jury") and nom_cj_in:
            if nom_cj_in not in st.session_state.co_jurys:
                st.session_state.co_jurys.append(nom_cj_in); st.success(f"Co-jury {nom_cj_in} ajouté."); st.rerun()
            else: st.warning("Co-jury déjà existant.")
    if st.session_state.co_jurys:
        st.subheader("Liste des co-jurys")
        for idx, cj in enumerate(st.session_state.co_jurys):
            c1, c2 = st.columns([0.8, 0.2])
            c1.write(f"👨‍🏫 {cj}")
            if c2.button("Suppr.", key=f"cj_del_{idx}_{cj[:5]}"):
                del st.session_state.co_jurys[idx]; st.rerun()
    if st.button("Suivant > Dates", type="primary", key="cojury_next_btn"):
        st.session_state.etape = "dates"; st.rerun()

elif st.session_state.etape == "dates":
    afficher_navigation()
    st.header(etapes_labels["dates"])
    st.info("⚠️ Important : Les dates sélectionnées ici doivent correspondre aux dates présentes dans votre fichier CSV de disponibilités (ex: 26/01/2026).")
    
    nb_jours_def = len(st.session_state.dates_soutenance) if st.session_state.dates_soutenance else 2
    nb_jours_sout_in = st.number_input("Nombre de jours de soutenances", 1, 10, nb_jours_def, key="nb_jours_in_key")
    dates_saisies_ui = []
    
    default_date = datetime(2026, 1, 26).date() 
    
    cols_dates = st.columns(min(nb_jours_sout_in, 4))
    for i in range(nb_jours_sout_in):
        if i < len(st.session_state.dates_soutenance):
            val_d = st.session_state.dates_soutenance[i]
        else:
            val_d = default_date + timedelta(days=i if i < 2 else i+1) 
        
        with cols_dates[i % 4]:
            d = st.date_input(f"Date Jour {i+1}", value=val_d, key=f"date_sout_in_{i}")
            dates_saisies_ui.append(d)
            
    if st.button("Valider > Créneaux", type="primary", key="dates_val_btn"):
        st.session_state.dates_soutenance = dates_saisies_ui
        st.session_state.etape = "disponibilites"; st.rerun()

elif st.session_state.etape == "disponibilites":
    afficher_navigation()
    st.header(etapes_labels["disponibilites"])
    if st.session_state.dates_soutenance and st.session_state.duree_soutenance:
        horaires_par_jour_etape6 = {}
        planif_temp_etape6 = PlanificationOptimiseeV2([],[], st.session_state.dates_soutenance, {}, 1, st.session_state.duree_soutenance)
        creneaux_uniques_etape6 = planif_temp_etape6.generer_creneaux_uniques()
        
        for creneau_unique in creneaux_uniques_etape6:
            jour_str = creneau_unique['jour']
            heure_str = creneau_unique['heure']
            if jour_str not in horaires_par_jour_etape6:
                horaires_par_jour_etape6[jour_str] = []
            if heure_str not in horaires_par_jour_etape6[jour_str]:
                horaires_par_jour_etape6[jour_str].append(heure_str)
        
        for jour_key in horaires_par_jour_etape6:
            horaires_par_jour_etape6[jour_key].sort(key=lambda x: datetime.strptime(x.split(" - ")[0], "%H:%M"))

        st.session_state.horaires_par_jour = horaires_par_jour_etape6

        for jour_aff, slots_aff in st.session_state.horaires_par_jour.items():
            st.subheader(f"📅 {jour_aff}")
            if slots_aff:
                cols_aff = st.columns(min(len(slots_aff), 5)) 
                for i_s, slot_s in enumerate(slots_aff):
                    with cols_aff[i_s % 5]: st.info(f"🕒 {slot_s}")
            else: st.write("Aucun créneau généré.")
        if st.button("Suivant > Saisie Disponibilités", type="primary", key="creneaux_next_btn"):
            st.session_state.etape = "disponibilites_selection"; st.rerun()
    else: st.warning("Définissez dates et durée avant de générer les créneaux.")

elif st.session_state.etape == "disponibilites_selection":
    afficher_navigation()
    st.header(etapes_labels["disponibilites_selection"])

    st.subheader("⬇️ Importer les disponibilités (CSV)")
    st.markdown("**Format attendu :** CSV avec séparateur point-virgule (;). Colonnes : `NOM`, `FILIERE`, `Nb`, `Date HeureStart - HeureEnd`, ...")

    uploaded_file_dispo_ui_val = st.file_uploader("Choisir fichier CSV Disponibilités", type=["csv"], key="dispo_uploader_csv")

    if uploaded_file_dispo_ui_val is not None:
        horaires_ok = st.session_state.get("horaires_par_jour") and isinstance(st.session_state.horaires_par_jour, dict) and st.session_state.horaires_par_jour
        etudiants_ok = st.session_state.get("etudiants") and isinstance(st.session_state.etudiants, list) and st.session_state.etudiants

        if horaires_ok and etudiants_ok:
            tuteurs_app_list_imp = list(set([e["Tuteur"] for e in st.session_state.etudiants if "Tuteur" in e]))
            cojurys_app_list_imp = st.session_state.co_jurys if st.session_state.co_jurys else []
            
            with st.spinner("Import CSV en cours..."):
                s_msg, e_msg, w_msg = importer_disponibilites_csv(
                    uploaded_file_dispo_ui_val, st.session_state.horaires_par_jour, 
                    tuteurs_app_list_imp, cojurys_app_list_imp, score_matching_seuil=75
                )
            for m in s_msg: st.success(m)
            for m in e_msg: st.error(m)
            with st.expander("Voir les avertissements"):
                for m in w_msg: st.warning(m)
        else: st.error("Prérequis manquants (étapes créneaux/étudiants).")
    
    st.divider()
    st.subheader("✏️ Vérification / Modification")
    
    tous_tuteurs_disp_ui = list(set([e["Tuteur"] for e in st.session_state.etudiants if "Tuteur" in e])) if st.session_state.etudiants else []
    co_jurys_disp_ui = st.session_state.co_jurys if st.session_state.co_jurys else []
    personnes_disp_ui = sorted(list(set(tous_tuteurs_disp_ui + co_jurys_disp_ui)))

    if not personnes_disp_ui:
        st.info("Aucun jury défini.")
    else:
        for p_disp_init in personnes_disp_ui:
            if p_disp_init not in st.session_state.disponibilites: st.session_state.disponibilites[p_disp_init] = {}

        for personne_disp_loop in personnes_disp_ui:
            with st.expander(f"👨‍🏫 {personne_disp_loop}", expanded=False):
                dispos_p_actuelle = st.session_state.disponibilites.get(personne_disp_loop, {})
                for jour_disp_loop, creneaux_list_jour_disp in st.session_state.horaires_par_jour.items():
                    if not creneaux_list_jour_disp: continue 
                    st.write(f"**{jour_disp_loop}**")
                    
                    cols_disp_cb = st.columns(min(len(creneaux_list_jour_disp), 4))
                    for i_disp_cb, creneau_val_disp_cb in enumerate(creneaux_list_jour_disp):
                        with cols_disp_cb[i_disp_cb % 4]:
                            key_dispo_individual_cb = f"{jour_disp_loop} | {creneau_val_disp_cb}"
                            valeur_actuelle_cb_state = st.session_state.disponibilites[personne_disp_loop].get(key_dispo_individual_cb, False)
                            
                            if st.checkbox(creneau_val_disp_cb.split(" - ")[0], value=valeur_actuelle_cb_state, key=f"cb_{personne_disp_loop}_{i_disp_cb}_{jour_disp_loop}"):
                                st.session_state.disponibilites[personne_disp_loop][key_dispo_individual_cb] = True
                            else:
                                st.session_state.disponibilites[personne_disp_loop][key_dispo_individual_cb] = False

    if st.button("Suivant > Générer Planning", type="primary", key="dispo_sel_next_btn"):
        st.session_state.etape = "generation"; st.rerun()

elif st.session_state.etape == "generation":
    afficher_navigation()
    st.header(etapes_labels["generation"])
    
    utiliser_ag_ui = st.checkbox(
        "Utiliser l'algorithme génétique (recommandé)", value=True, 
        key="utiliser_ag_checkbox"
    )
    params_ag_config_from_ui = {}
    if utiliser_ag_ui:
        with st.expander("⚙️ Paramètres de l'algorithme génétique"):
            taille_pop_val = st.slider("Taille population AG", 20, 250, 80)
            nb_gen_val = st.slider("Nb générations AG", 20, 1000, 300)
            params_ag_config_from_ui = {
                'taille_population': taille_pop_val, 'nb_generations': nb_gen_val
            }
        
    if st.button("🚀 Lancer l'optimisation", type="primary", key="lancer_opti_final_btn"):
        if not st.session_state.etudiants: st.error("Aucun étudiant à planifier."); st.stop()
        if not st.session_state.dates_soutenance: st.error("Aucune date de soutenance définie."); st.stop()

        with st.spinner("Optimisation en cours..."):
            optimiseur = PlanificationOptimiseeV2(
                st.session_state.etudiants, st.session_state.co_jurys, st.session_state.dates_soutenance,
                st.session_state.disponibilites, st.session_state.nb_salles, st.session_state.duree_soutenance
            )
            planning_final_opti, non_planifies_opti, stats_ag_opti = optimiseur.optimiser_avec_genetique(
                utiliser_genetique_ui=utiliser_ag_ui, **params_ag_config_from_ui
            )
            st.session_state.planning_final = planning_final_opti

        if stats_ag_opti:
            st.success(f"Fitness Finale: {stats_ag_opti.get('fitness_finale', 0.0):.0f}")

        if st.session_state.planning_final:
            st.success(f"Planning généré! {len(st.session_state.planning_final)} soutenances planifiées.")
            if non_planifies_opti > 0: st.warning(f"⚠️ {non_planifies_opti} étudiant(s) non planifiés.")

            df_planning_final_ui = pd.DataFrame(st.session_state.planning_final)
            st.subheader("📋 Planning détaillé")
            st.dataframe(df_planning_final_ui, use_container_width=True, hide_index=True)

            csv_export_ui = df_planning_final_ui.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("Télécharger CSV", csv_export_ui, "planning_soutenances.csv", "text/csv")
        else:
            st.error("❌ Aucune soutenance n'a pu être planifiée.")

# Sidebar Résumé
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Résumé Actuel")
    st.write(f"**Étudiants :** {len(st.session_state.etudiants)}")
    st.write(f"**Co-jurys :** {len(st.session_state.co_jurys)}")
    if st.session_state.dates_soutenance:
        st.write(f"**Dates :** {len(st.session_state.dates_soutenance)} jour(s)")
