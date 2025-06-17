import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
# import networkx as nx # Non utilisé, peut être supprimé
from itertools import combinations
import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from io import BytesIO, StringIO # StringIO pour le CSV
from thefuzz import fuzz # Pour le rapprochement des noms

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
    # Vérifier si st.session_state.etape est valide avant de l'utiliser dans index
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

                # Vérifier si le créneau (salle incluse via son ID unique) est déjà pris
                if creneau_cand['id'] in creneaux_occupes_par_id: # L'ID du créneau est unique par salle/heure
                    continue
                
                tuteur_etu = self.planificateur.etudiants[idx_etu]["Tuteur"]
                moment_cand = creneau_cand['moment'] # Clé jour_heure
                
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
                    genes[idx_etu] = idx_creneau_cand # Assigner l'index du créneau (qui inclut la salle)
                    creneaux_occupes_par_id.add(creneau_cand['id']) # Marquer l'ID du créneau (salle+heure)
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
        
        # CONTRAINTE IMPERATIVE DE PARITE DES ROLES
        contrainte_parite_roles_violee_strict = False
        max_difference_toleree_strict = 0 # Doit être strictement égal pour la contrainte impérative

        for jury, counts in roles_par_jury.items():
            if jury in personnes_eligibles_balance:
                difference = abs(counts['tuteur'] - counts['cojury'])
                if difference > max_difference_toleree_strict: # Contrainte impérative
                    contrainte_parite_roles_violee_strict = True
                    # La pénalité sera appliquée globalement si cette variable est True
                
                # Pour la partie "soft" de la fitness (bonus/pénalité non bloquante)
                penalite_balance_roles += (difference ** 2) * 20 
                if difference == 0:
                    score_balance_roles += 100 
                elif difference == 1: # Si on tolérait un écart de 1 pour le score "soft"
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
        elif contrainte_parite_roles_violee_strict: # Appliqué seulement si pas d'autres conflits
            fitness_finale = -1_000_000 - penalite_balance_roles # Pénalité massive si parité impérative violée
        else: # Pas de conflits, pas de violation de parité impérative
            if nb_soutenances >= nb_total_etudiants * 0.9: 
                fitness_finale += 2000 

        individu.fitness = fitness_finale
        individu.soutenances_planifiees = nb_soutenances
        individu.conflits = total_conflits # Ce 'conflits' inclut salle et jury, pas la parité.
                                         # On pourrait ajouter un champ `parite_violee` à l'individu si besoin.
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
        if not planning: return 0.0 # Retourner un float pour la cohérence
        charges = defaultdict(int)
        for soutenance in planning:
            charges[soutenance['Tuteur']] += 1
            charges[soutenance['Co-jury']] += 1
        
        if not charges or len(charges) <= 1 : return 10.0 
        
        valeurs_charges = np.array(list(charges.values()))
        # Moyenne et variance sont déjà gérées par np.var si on passe ddof=0 (par défaut pour population)
        variance = np.var(valeurs_charges) 
        return max(0.0, 10.0 - np.sqrt(variance)) 

    def _calculer_bonus_alternance(self, planning): 
        bonus = 0.0 # Float
        # ... (logique inchangée mais retourne un float)
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

        enfant1_genes = genes_p1_copy = parent1.genes[:] # Copie pour manipulation
        enfant2_genes = genes_p2_copy = parent2.genes[:] # Copie pour manipulation
        
        point_croisement = random.randint(1, len_genes - 1) if len_genes > 1 else 0

        # Échange simple pour la première partie (plus robuste)
        for i in range(point_croisement):
            enfant1_genes[i] = parent2.genes[i] # Prend de P2
            enfant2_genes[i] = parent1.genes[i] # Prend de P1
        # La deuxième partie reste celle du parent original pour l'instant

        # Tenter de résoudre les conflits de créneaux dupliqués après le croisement initial
        # Cette partie est complexe à rendre parfaitement valide et peut être simplifiée
        # ou gérée par la mutation et la sélection.
        # Pour l'instant, on se contente du croisement simple, la fitness et la mutation aideront.

        # Une approche plus simple pour le croisement :
        # enfant1_genes = parent1.genes[:point_croisement] + parent2.genes[point_croisement:]
        # enfant2_genes = parent2.genes[:point_croisement] + parent1.genes[point_croisement:]
        # Puis, on pourrait avoir une étape de "réparation" pour les doublons de créneaux,
        # ou laisser la mutation/sélection s'en charger.
        # La version actuelle avec vérification de creneaux_valides_par_etudiant est déjà assez avancée.

        # Reprenons la logique de croisement intelligent avec vérification de validité et unicité
        enfant1_genes_final = [-1] * len_genes
        enfant2_genes_final = [-1] * len_genes

        # Partie 1 de parent1 pour enfant1, parent2 pour enfant2
        for i in range(point_croisement):
            enfant1_genes_final[i] = parent1.genes[i]
            enfant2_genes_final[i] = parent2.genes[i]
        
        creneaux_pris_e1 = set(g for g in enfant1_genes_final[:point_croisement] if g != -1)
        creneaux_pris_e2 = set(g for g in enfant2_genes_final[:point_croisement] if g != -1)

        # Partie 2 : essayer de prendre de l'autre parent si valide et non pris
        for i in range(point_croisement, len_genes):
            # Enfant 1 prend de Parent 2
            gene_candidat_p2 = parent2.genes[i]
            if gene_candidat_p2 != -1 and gene_candidat_p2 not in creneaux_pris_e1 and \
               (i in self.creneaux_valides_par_etudiant and gene_candidat_p2 in self.creneaux_valides_par_etudiant[i]):
                enfant1_genes_final[i] = gene_candidat_p2
                creneaux_pris_e1.add(gene_candidat_p2)
            else: # Sinon, essayer de garder de Parent 1
                gene_candidat_p1 = parent1.genes[i]
                if gene_candidat_p1 != -1 and gene_candidat_p1 not in creneaux_pris_e1 and \
                   (i in self.creneaux_valides_par_etudiant and gene_candidat_p1 in self.creneaux_valides_par_etudiant[i]):
                    enfant1_genes_final[i] = gene_candidat_p1
                    creneaux_pris_e1.add(gene_candidat_p1)

            # Enfant 2 prend de Parent 1
            gene_candidat_p1_e2 = parent1.genes[i]
            if gene_candidat_p1_e2 != -1 and gene_candidat_p1_e2 not in creneaux_pris_e2 and \
               (i in self.creneaux_valides_par_etudiant and gene_candidat_p1_e2 in self.creneaux_valides_par_etudiant[i]):
                enfant2_genes_final[i] = gene_candidat_p1_e2
                creneaux_pris_e2.add(gene_candidat_p1_e2)
            else: # Sinon, essayer de garder de Parent 2
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
                # S'assurer que l'étudiant i est valide
                if i >= self.nb_etudiants: continue 
                
                creneaux_possibles_pour_cet_etudiant = self.creneaux_valides_par_etudiant.get(i, [])
                if not creneaux_possibles_pour_cet_etudiant: continue

                gene_actuel_de_i = individu.genes[i]
                
                # Créneaux utilisés par les AUTRES étudiants
                creneaux_utilises_par_autres = set(
                    individu.genes[j] for j in range(len(individu.genes)) if j != i and individu.genes[j] != -1
                )

                # Créneaux valides pour cet étudiant ET non utilisés par d'autres
                options_de_mutation_libres = [
                    c for c in creneaux_possibles_pour_cet_etudiant if c not in creneaux_utilises_par_autres
                ]

                if not options_de_mutation_libres:
                    # Si l'étudiant est planifié mais son créneau cause un conflit (pris par un autre)
                    # ou si son créneau n'est plus valide pour lui (peu probable si bien généré)
                    # Alors on le déplanifie.
                    if gene_actuel_de_i != -1 and gene_actuel_de_i in creneaux_utilises_par_autres :
                         individu.genes[i] = -1
                    # Sinon (non planifié et pas d'option, ou planifié sans conflit et pas d'autre option), on ne change rien.
                    continue

                if gene_actuel_de_i == -1: # Était non planifié, on essaie de le planifier
                    individu.genes[i] = random.choice(options_de_mutation_libres)
                else: # Était planifié, on essaie de changer de créneau (si possible vers un autre)
                    # On préfère un NOUVEAU créneau libre s'il existe
                    nouveaux_creneaux_libres = [c for c in options_de_mutation_libres if c != gene_actuel_de_i]
                    if nouveaux_creneaux_libres:
                        individu.genes[i] = random.choice(nouveaux_creneaux_libres)
                    elif gene_actuel_de_i in options_de_mutation_libres : # Seule option libre est son créneau actuel
                        pass # On ne change pas
                    else: # Son créneau actuel n'est plus une option libre (conflit), donc déplanifier
                        individu.genes[i] = -1
        return individu

    def evoluer(self) -> Tuple[List[Dict], Dict]:
        # ... (Logique d'évolution, globalement inchangée mais s'assurer des gardes pour population vide)
        population = []
        if self.nb_etudiants == 0:
             st.sidebar.warning("AG: Aucun étudiant, arrêt.")
             return [], self._stats_vides()

        for _ in range(self.taille_population):
            individu = self.generer_individu_intelligent()
            population.append(self.calculer_fitness_amelioree(individu))

        if not population or not any(ind.genes != [-1]*self.nb_etudiants for ind in population) : # Si tous sont vides
            st.sidebar.warning("AG: Population initiale non viable.")
            self.meilleure_solution = Individu(genes=[-1]*self.nb_etudiants) # S'assurer qu'il y a une solution par défaut
            # return [], self._stats_vides() # On peut laisser continuer pour voir si la mutation aide.

        # S'assurer que meilleure_solution est initialisée même si la population est problématique
        if population:
            self.meilleure_solution = max(population, key=lambda x: x.fitness, default=self.meilleure_solution)
        else: # Si la population est vide après la première génération.
            st.sidebar.error("AG: Population vide après initialisation et évaluation.")
            return [], self._stats_vides()


        stagnation = 0
        for generation in range(self.nb_generations):
            if not population: # Double check
                st.sidebar.error(f"AG: Population vide à la gen {generation}, arrêt prématuré.")
                break

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
                    if population: # Pour random.choice
                        individu_choisi_pour_mutation_ou_copie = random.choice(population)
                        nouvelle_population.append(self.mutation_adaptative(individu_choisi_pour_mutation_ou_copie))
                    else: # Si la population s'est vidée, on ne peut rien faire
                        break 
            
            if not nouvelle_population: # Si la construction a échoué
                 st.sidebar.warning(f"AG: Nouvelle population vide à la gen {generation}.")
                 break # Sortir de la boucle des générations

            nouvelle_population = nouvelle_population[:self.taille_population] 
            population = [self.calculer_fitness_amelioree(ind) for ind in nouvelle_population]

            if not population: 
                st.sidebar.error(f"AG: Population vide après évaluation à la gen {generation}.")
                break 

            meilleur_actuel_gen = max(population, key=lambda x: x.fitness, default=self.meilleure_solution)

            if meilleur_actuel_gen.fitness > self.meilleure_solution.fitness:
                self.meilleure_solution = meilleur_actuel_gen
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > max(30, self.nb_generations * 0.15) and generation < self.nb_generations * 0.85: 
                nb_a_remplacer = int(self.taille_population * 0.33) 
                # Garder les meilleurs, remplacer une partie des moins bons par de nouveaux individus
                population_meilleurs_conserves = sorted(population, key=lambda x: x.fitness, reverse=True)[:-nb_a_remplacer]
                nouveaux_individus_pour_diversite = [self.calculer_fitness_amelioree(self.generer_individu_intelligent()) for _ in range(nb_a_remplacer)]
                population = population_meilleurs_conserves + nouveaux_individus_pour_diversite
                random.shuffle(population) # Important après modification
                stagnation = 0
                if self.nb_generations > 20 : 
                    st.sidebar.text(f"AG: Diversification à gén. {generation+1}")


            fitness_moyenne_gen = np.mean([ind.fitness for ind in population]) if population else self.meilleure_solution.fitness
            self.historique_fitness.append({
                'generation': generation+1,
                'fitness_max': self.meilleure_solution.fitness, 
                'fitness_moyenne': fitness_moyenne_gen,
                'soutenances_max': self.meilleure_solution.soutenances_planifiees,
                'conflits_min': self.meilleure_solution.conflits
            })
            if (generation+1) % max(1, (self.nb_generations // 10)) == 0 or generation == self.nb_generations -1 : 
                 st.sidebar.text(f"G:{generation+1} FMax:{self.meilleure_solution.fitness:.0f} S:{self.meilleure_solution.soutenances_planifiees} C:{self.meilleure_solution.conflits}")

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
    
    def _stats_vides(self): # Helper pour retourner des stats vides
        return {'generations': 0, 'fitness_finale': 0, 'soutenances_planifiees': 0, 
                'conflits': 0, 'taux_reussite': 0, 'historique': []}


    def selection_tournament(self, population: List[Individu], k=3) -> Individu:
        if not population:
            return Individu(genes=[-1]*self.nb_etudiants if self.nb_etudiants > 0 else [])
        
        k_valide = min(k, len(population))
        if k_valide == 0 : return population[0] # Devrait être impossible si population non vide

        participants = random.sample(population, k_valide)
        return max(participants, key=lambda x: x.fitness)

    def decoder_individu(self, individu: Individu) -> List[Dict]:
        planning = []
        if not individu or not individu.genes or not self.creneaux or not self.planificateur.etudiants: # Gardes
            return planning

        jurys_occupes_decode_moment = defaultdict(set) 
        creneaux_salles_decode_ids = set() # Utiliser l'ID unique du créneau (salle+heure)

        for idx_etu, idx_creneau_gene in enumerate(individu.genes):
            if idx_creneau_gene == -1 or idx_creneau_gene >= len(self.creneaux):
                continue
            if idx_etu >= len(self.planificateur.etudiants): continue
            
            etudiant_obj = self.planificateur.etudiants[idx_etu]
            creneau_obj_decode = self.creneaux[idx_creneau_gene]
            tuteur_principal = etudiant_obj["Tuteur"]
            
            moment_str_decode = creneau_obj_decode['moment'] # Jour_Heure
            id_creneau_salle_decode = creneau_obj_decode['id'] # ID unique pour Jour_Heure_Salle

            if id_creneau_salle_decode in creneaux_salles_decode_ids:
                continue 
            if tuteur_principal in jurys_occupes_decode_moment[moment_str_decode]:
                continue 

            co_jurys_possibles_decode = self.planificateur.trouver_co_jurys_disponibles(
                tuteur_principal, creneau_obj_decode['jour'], creneau_obj_decode['heure']
            ) # Cette méthode utilise déjà le tri par balance des rôles
            
            co_jury_final_choisi = None
            # random.shuffle(co_jurys_possibles_decode) # Le tri est déjà fait, on prend le premier dispo
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
            # Modification de la période du matin pour finir à 12h10 au lieu de 13h00
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
            creneaux_melanges_classique = creneaux.copy() # Re-mélanger pour chaque étudiant donne plus de chances
            random.shuffle(creneaux_melanges_classique)

            for creneau_obj_classique in creneaux_melanges_classique:
                if creneau_obj_classique['id'] in creneaux_occupes_ids: continue
                if not self.est_disponible(tuteur_ref_classique, creneau_obj_classique['jour'], creneau_obj_classique['heure']): continue
                if tuteur_ref_classique in jurys_par_moment_app[creneau_obj_classique['moment']]: continue

                co_jurys_possibles_classique = self.trouver_co_jurys_disponibles( 
                    tuteur_ref_classique, creneau_obj_classique['jour'], creneau_obj_classique['heure']
                )
                co_jury_choisi_classique = None
                for cj_cand_classique in co_jurys_possibles_classique: # Déjà trié par balance/charge
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
                # st.sidebar.warning(f"Classique: {etudiant_obj_classique['Prénom']} non planifié.") # Peut être trop verbeux
        
        return planning, non_planifies_count

    def optimiser_avec_genetique(self, utiliser_genetique_ui=False, **params_genetique_ui): 
        planning_classique, non_planifies_classique = self.optimiser_planning_ameliore()
        nb_etudiants_total = len(self.etudiants) if self.etudiants else 0
        taux_reussite_classique = (len(planning_classique) / nb_etudiants_total) if nb_etudiants_total > 0 else 0.0

        run_ag = False
        if utiliser_genetique_ui:
            run_ag = True
        elif nb_etudiants_total > 0 and taux_reussite_classique < 0.85 and planning_classique : # Lancer AG si <85% et si le classique a produit quelque chose
            run_ag = True
        elif nb_etudiants_total > 0 and not planning_classique: # Lancer AG si le classique n'a rien produit
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

            # Si le planificateur (self) n'a pas d'étudiants, l'AG ne peut pas fonctionner
            if not self.etudiants:
                st.warning("AG non lancé : aucun étudiant à planifier.")
                return planning_classique, non_planifies_classique, None

            ag_instance = AlgorithmeGenetique(self, **config_ag)
            planning_genetique, stats_ag = ag_instance.evoluer()
            
            # S'assurer que stats_ag est un dictionnaire même si evoluer retourne None par erreur
            if stats_ag is None: stats_ag = ag_instance._stats_vides() # Utiliser le helper
            stats_ag['amelioration_valeur'] = 0 

            # Comparaison basée sur le nombre de soutenances planifiées ET la fitness (si conflits/parité sont gérés par fitness)
            classique_score_comparaison = len(planning_classique) # Score simple pour le classique
            genetique_score_comparaison = len(planning_genetique) # Score simple pour l'AG

            # On pourrait aussi calculer une "fitness" pour le planning classique pour une meilleure comparaison
            # Mais pour l'instant, on se base sur le nombre de planifiés et la fitness de l'AG

            if genetique_score_comparaison > classique_score_comparaison:
                st.success(f"✅ AG a amélioré le nombre de soutenances: {len(planning_genetique)} vs {len(planning_classique)} (classique)")
                stats_ag['amelioration_valeur'] = len(planning_genetique) - len(planning_classique)
                return planning_genetique, nb_etudiants_total - len(planning_genetique), stats_ag
            elif genetique_score_comparaison == classique_score_comparaison and stats_ag.get('fitness_finale', -float('inf')) > -100000 : 
                 # Si même nombre, on peut regarder la fitness ou d'autres critères
                 # Pour l'instant, on favorise l'AG s'il a bien tourné
                 st.info(f"AG a planifié autant ({len(planning_genetique)}). Résultat de l'AG conservé (Fitness: {stats_ag.get('fitness_finale', 0.0):.0f}).")
                 return planning_genetique, nb_etudiants_total - len(planning_genetique), stats_ag
            else: # Classique est meilleur ou AG n'a pas bien tourné
                st.info(f"ℹ️ AG n'a pas amélioré ({len(planning_genetique)} planifiées, fitness {stats_ag.get('fitness_finale', 0.0):.0f}). Résultat classique ({len(planning_classique)}) conservé.")
                return planning_classique, non_planifies_classique, stats_ag 
        
        return planning_classique, non_planifies_classique, None


    def afficher_diagnostics(self, planning, tentatives_par_etudiant):
        # (Optionnel)
        pass

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


# --- Fonctions d'importation des disponibilités ---
def importer_disponibilites_excel_simple_header(
                                uploaded_file, 
                                horaires_par_jour_app_config: Dict[str, List[str]], 
                                tous_tuteurs_app: List[str], 
                                co_jurys_app: List[str],
                                score_matching_seuil=75): # J'ai remis 75 comme vous l'aviez
    messages_succes, messages_erreur, messages_warning = [], [], []
    personnes_traitees_import, personnes_reconnues_app_set = set(), set(tous_tuteurs_app + co_jurys_app)
    map_creneaux_app, cles_dispo_valides_app_set = {}, set()

    for jour_app_str, creneaux_list_app in horaires_par_jour_app_config.items():
        try: date_part_app = jour_app_str.split(" ")[1]
        except (IndexError, ValueError): continue
        for creneau_app_str in creneaux_list_app: 
            cles_dispo_valides_app_set.add(f"{jour_app_str} | {creneau_app_str}")
            try:
                h_debut, h_fin = [h.strip() for h in creneau_app_str.split(" - ")]
                map_creneaux_app[f"{date_part_app} {h_debut} à {h_fin}"] = f"{jour_app_str} | {creneau_app_str}"
            except ValueError: continue

    if not uploaded_file: return [], ["Aucun fichier Excel fourni."], []
    try:
        df_excel = pd.read_excel(uploaded_file, header=0, sheet_name=0)
        if df_excel.empty: return [], ["Fichier Excel vide."], []
        
        col_ens_nom = df_excel.columns[0]
        original_col_names = df_excel.columns.tolist()
        cleaned_col_map = {}

        for col_raw in original_col_names[2:]: 
            col_clean = str(col_raw).replace(" à\xa0 ", " à ").replace("\xa0", " ").strip(" .")
            parts = col_clean.split(" ")
            if len(parts) >= 4:
                try:
                    idx_a = parts.index("à")
                    if idx_a > 0 and idx_a + 1 < len(parts):
                        # S'assurer que parts[1] et parts[idx_a + 1] sont bien des heures
                        # Une vérification plus robuste pourrait utiliser regex ici
                        cle_excel_match = f"{parts[0]} {parts[1]} à {parts[idx_a + 1]}"
                        if cle_excel_match in map_creneaux_app:
                            cleaned_col_map[col_raw] = map_creneaux_app[cle_excel_match]
                except ValueError: pass # "à" non trouvé ou autre problème de format

        if not cleaned_col_map:
            err_msg = ["Aucun en-tête de créneau Excel mappé. Vérifiez formats."]
            if len(original_col_names) > 2: err_msg.append(f"Ex: '{original_col_names[2]}'")
            if map_creneaux_app: err_msg.append(f"Attendu (ex): '{list(map_creneaux_app.keys())[0]}'")
            return [], err_msg, []

        for _, row in df_excel.iterrows(): 
            nom_enseignant_fichier = str(row[col_ens_nom]).strip() 
            if not nom_enseignant_fichier: continue

            best_match_nom_app = None
            highest_score = 0
            nom_enseignant_fichier_lower = nom_enseignant_fichier.lower()

            for nom_app in personnes_reconnues_app_set:
                nom_app_lower = nom_app.lower()
                score = fuzz.token_sort_ratio(nom_enseignant_fichier_lower, nom_app_lower)
                if score > highest_score:
                    highest_score = score
                    best_match_nom_app = nom_app
            
            nom_enseignant_final_pour_app = None # Initialisation ici pour la portée
            if highest_score >= score_matching_seuil:
                nom_enseignant_final_pour_app = best_match_nom_app
                if nom_enseignant_final_pour_app.lower() != nom_enseignant_fichier_lower or highest_score < 100 :
                     messages_warning.append(
                         f"Fichier: '{nom_enseignant_fichier}' rapproché avec App: '{nom_enseignant_final_pour_app}' (score: {highest_score}%)"
                     )
            else:
                messages_warning.append(
                    f"Fichier: '{nom_enseignant_fichier}' non rapproché (meilleur score: {highest_score}% vs '{best_match_nom_app if best_match_nom_app else 'aucun'}'). Ignoré."
                )
                continue
            
            # Si on arrive ici, nom_enseignant_final_pour_app EST défini.
            personnes_traitees_import.add(nom_enseignant_final_pour_app) # Correction: Utiliser la variable correcte
            if nom_enseignant_final_pour_app not in st.session_state.disponibilites: # Correction: Utiliser la variable correcte
                st.session_state.disponibilites[nom_enseignant_final_pour_app] = {} # Correction: Utiliser la variable correcte
            
            for col_orig_excel, cle_app_match in cleaned_col_map.items():
                if col_orig_excel in row: # S'assurer que la colonne existe dans la ligne actuelle
                    val = row[col_orig_excel]
                    try:
                        if pd.isna(val): continue
                        # Correction: Utiliser la variable correcte
                        st.session_state.disponibilites[nom_enseignant_final_pour_app][cle_app_match] = bool(int(float(val)))
                    except ValueError: 
                        # Correction: Utiliser la variable correcte pour le nom de l'enseignant du fichier
                        messages_erreur.append(f"Val:'{val}' invalide pour '{nom_enseignant_fichier}' à '{col_orig_excel}'.")

        for p_nettoyage in personnes_traitees_import:
            if p_nettoyage in st.session_state.disponibilites:
                dispos_p = st.session_state.disponibilites[p_nettoyage]
                st.session_state.disponibilites[p_nettoyage] = {k:v for k,v in dispos_p.items() if k in cles_dispo_valides_app_set}

        if personnes_traitees_import: messages_succes.append(f"Dispos Excel importées pour {len(personnes_traitees_import)}.")
        elif not messages_erreur: messages_warning.append("Aucune personne Excel traitée.")
    except ImportError: messages_erreur.append("'openpyxl' requis. `pip install openpyxl`")
    except Exception as e: messages_erreur.append(f"Erreur import Excel: {str(e)}")
    return messages_succes, messages_erreur, messages_warning

def importer_disponibilites_csv(uploaded_file, 
                                horaires_par_jour_app_config: Dict[str, List[str]], 
                                tous_tuteurs_app: List[str], 
                                co_jurys_app: List[str],
                                score_matching_seuil=75):
    messages_succes, messages_erreur, messages_warning = [], [], []
    personnes_traitees_import, personnes_reconnues_app_set = set(), set(tous_tuteurs_app + co_jurys_app)
    map_creneaux_app, cles_dispo_valides_app_set = {}, set()

    # (Logique identique à la version Excel pour map_creneaux_app et cles_dispo_valides_app_set)
    for jour_app_str, creneaux_list_app in horaires_par_jour_app_config.items():
        try: date_part_app = jour_app_str.split(" ")[1]
        except (IndexError, ValueError): continue
        for creneau_app_str in creneaux_list_app: 
            cles_dispo_valides_app_set.add(f"{jour_app_str} | {creneau_app_str}")
            try:
                h_debut, h_fin = [h.strip() for h in creneau_app_str.split(" - ")]
                map_creneaux_app[f"{date_part_app} {h_debut} à {h_fin}"] = f"{jour_app_str} | {creneau_app_str}"
            except ValueError: continue
    
    if not uploaded_file: return [], ["Aucun fichier CSV fourni."], []
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df_csv = pd.read_csv(stringio, sep=';', header=0, skipinitialspace=True)
        if df_csv.empty: return [], ["Fichier CSV vide."], []

        col_ens_nom_csv = df_csv.columns[0]
        original_col_names_csv = df_csv.columns.tolist()
        cleaned_col_map_csv = {}

        for col_raw_csv in original_col_names_csv[2:]:
            col_clean_csv = str(col_raw_csv).replace(" à\xa0 ", " à ").replace("\xa0", " ").strip(" .")
            parts_csv = col_clean_csv.split(" ")
            if len(parts_csv) >= 4:
                try:
                    idx_a_csv = parts_csv.index("à")
                    if idx_a_csv > 0 and idx_a_csv + 1 < len(parts_csv):
                        cle_csv_match = f"{parts_csv[0]} {parts_csv[1]} à {parts_csv[idx_a_csv + 1]}"
                        if cle_csv_match in map_creneaux_app:
                            cleaned_col_map_csv[col_raw_csv] = map_creneaux_app[cle_csv_match]
                except ValueError: pass
        
        if not cleaned_col_map_csv:
            err_msg_csv = ["Aucun en-tête CSV mappé. Vérifiez formats."]
            if len(original_col_names_csv) > 2: err_msg_csv.append(f"Ex CSV non mappé: '{original_col_names_csv[2]}'")
            if map_creneaux_app: err_msg_csv.append(f"Attendu (ex): '{list(map_creneaux_app.keys())[0]}'")
            return [], err_msg_csv, []

        for _, row_csv in df_csv.iterrows():
            nom_ens_csv_row = str(row_csv[col_ens_nom_csv]).strip()
            if not nom_ens_csv_row: continue
            best_match_csv, h_score_csv = None, 0
            for nom_app_csv in personnes_reconnues_app_set:
                score_csv = fuzz.ratio(nom_ens_csv_row.lower(), nom_app_csv.lower())
                if score_csv > h_score_csv: h_score_csv, best_match_csv = score_csv, nom_app_csv
            
            nom_ens_final_csv = None
            if h_score_csv >= score_matching_seuil:
                nom_ens_final_csv = best_match_csv
                if nom_ens_final_csv != nom_ens_csv_row: messages_warning.append(f"CSV:'{nom_ens_csv_row}' -> App:'{nom_ens_final_csv}' ({h_score_csv}%)")
            else:
                messages_warning.append(f"CSV:'{nom_ens_csv_row}' non rapproché (max {h_score_csv}% vs '{best_match_csv}'). Ignoré.")
                continue

            personnes_traitees_import.add(nom_ens_final_csv)
            if nom_ens_final_csv not in st.session_state.disponibilites: st.session_state.disponibilites[nom_ens_final_csv] = {}

            for col_orig_csv, cle_app_match_csv in cleaned_col_map_csv.items():
                if col_orig_csv in row_csv: # S'assurer que la colonne existe dans la ligne (peut être manquant si CSV malformé)
                    val_csv = row_csv[col_orig_csv]
                    try:
                        if pd.isna(val_csv): continue
                        # Les valeurs CSV devraient être des int directement, mais float au cas où
                        st.session_state.disponibilites[nom_ens_final_csv][cle_app_match_csv] = bool(int(float(val_csv))) 
                    except ValueError: messages_erreur.append(f"Val CSV:'{val_csv}' invalide pour '{nom_ens_csv_row}' à '{col_orig_csv}'.")
        
        for p_nettoyage_csv in personnes_traitees_import: # Pruning
            if p_nettoyage_csv in st.session_state.disponibilites:
                dispos_p_csv = st.session_state.disponibilites[p_nettoyage_csv]
                st.session_state.disponibilites[p_nettoyage_csv] = {k:v for k,v in dispos_p_csv.items() if k in cles_dispo_valides_app_set}

        if personnes_traitees_import: messages_succes.append(f"Dispos CSV importées pour {len(personnes_traitees_import)}.")
        elif not messages_erreur: messages_warning.append("Aucune personne CSV traitée.")

    except UnicodeDecodeError: messages_erreur.append("Encodage CSV invalide. Utilisez UTF-8.")
    except Exception as e_csv: messages_erreur.append(f"Erreur import CSV: {str(e_csv)}")
    return messages_succes, messages_erreur, messages_warning

# --- Interface utilisateur ---

st.sidebar.header("📥 Import Données de Base")
excel_file_base = st.sidebar.file_uploader("Étudiants & Co-jurys (.xlsx)", type=["xlsx"], key="excel_base_uploader_key")
if excel_file_base:
    try:
        excel_data_base = pd.read_excel(excel_file_base, sheet_name=None)
        if "etudiants" in excel_data_base:
            etu_df = excel_data_base["etudiants"]
            req_cols = {"Nom", "Prénom", "Pays", "Tuteur"}
            if req_cols.issubset(etu_df.columns):
                st.session_state.etudiants = etu_df[list(req_cols)].to_dict(orient="records")
                st.sidebar.success(f"{len(st.session_state.etudiants)} étudiants importés.")
            else: st.sidebar.error("Feuille 'etudiants': colonnes requises manquantes.")
        if "co_jurys" in excel_data_base:
            cj_df = excel_data_base["co_jurys"]
            if "Nom" in cj_df.columns:
                st.session_state.co_jurys = cj_df["Nom"].dropna().astype(str).tolist()
                st.sidebar.success(f"{len(st.session_state.co_jurys)} co-jurys importés.")
            else: st.sidebar.error("Feuille 'co_jurys': colonne 'Nom' manquante.")
    except Exception as e_imp_base: st.sidebar.error(f"Erreur Excel base: {e_imp_base}")


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
            if c2.button("Suppr.", key=f"cj_del_{idx}_{cj[:5]}"): # Clé plus unique
                del st.session_state.co_jurys[idx]; st.rerun()
    if st.button("Suivant > Dates", type="primary", key="cojury_next_btn"):
        st.session_state.etape = "dates"; st.rerun()

elif st.session_state.etape == "dates":
    afficher_navigation()
    st.header(etapes_labels["dates"])
    nb_jours_def = len(st.session_state.dates_soutenance) if st.session_state.dates_soutenance else 2
    nb_jours_sout_in = st.number_input("Nombre de jours de soutenances", 1, 10, nb_jours_def, key="nb_jours_in_key")
    dates_saisies_ui = []
    for i in range(nb_jours_sout_in):
        date_def = st.session_state.dates_soutenance[i] if i < len(st.session_state.dates_soutenance) else datetime.now().date() + timedelta(days=i)
        dates_saisies_ui.append(st.date_input(f"Date Jour {i+1}", value=date_def, key=f"date_sout_in_{i}"))
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
            if heure_str not in horaires_par_jour_etape6[jour_str]: # Éviter doublons d'heures par jour
                horaires_par_jour_etape6[jour_str].append(heure_str)
        
        # Trier les créneaux par heure de début pour chaque jour
        for jour_key in horaires_par_jour_etape6:
            horaires_par_jour_etape6[jour_key].sort(key=lambda x: datetime.strptime(x.split(" - ")[0], "%H:%M"))

        st.session_state.horaires_par_jour = horaires_par_jour_etape6

        for jour_aff, slots_aff in st.session_state.horaires_par_jour.items():
            st.subheader(f"📅 {jour_aff}")
            if slots_aff:
                cols_aff = st.columns(min(len(slots_aff), 5)) 
                for i_s, slot_s in enumerate(slots_aff):
                    with cols_aff[i_s % 5]: st.info(f"🕒 {slot_s}")
            else: st.write("Aucun créneau pour ce jour avec la durée/périodes spécifiées.")
        if st.button("Suivant > Saisie Disponibilités", type="primary", key="creneaux_next_btn"):
            st.session_state.etape = "disponibilites_selection"; st.rerun()
    else: st.warning("Définissez dates et durée avant de générer les créneaux.")

elif st.session_state.etape == "disponibilites_selection":
    afficher_navigation()
    st.header(etapes_labels["disponibilites_selection"])

    st.subheader("⬇️ Importer les disponibilités")
    import_type = st.radio("Type de fichier à importer:", ('Excel (.xlsx)', 'CSV (.csv)'), index=0, key="import_type_radio_key", horizontal=True)
    
    uploader_key_suffix = "_excel" if import_type == 'Excel (.xlsx)' else "_csv"
    file_types_allowed = ["xlsx", "xls"] if import_type == 'Excel (.xlsx)' else ["csv"]
    
    if import_type == 'Excel (.xlsx)':
        st.markdown("<small>Structure Excel: 1ère feuille, 1ère ligne en-tête (`ENSEIGNANT | FILIERE | JJ/MM/AAAA HH:MM à HH:MM | ...`), puis données.</small>", unsafe_allow_html=True)
        importer_func_selected = importer_disponibilites_excel_simple_header
    else: # CSV
        st.markdown("<small>Structure CSV: délimiteur ';', UTF-8, 1ère ligne en-tête (`ENSEIGNANT;FILIERE;JJ/MM/AAAA HH:MM à HH:MM;...`), puis données.</small>", unsafe_allow_html=True)
        importer_func_selected = importer_disponibilites_csv

    uploaded_file_dispo_ui_key = f"dispo_uploader{uploader_key_suffix}"
    uploaded_file_dispo_ui_val = st.file_uploader(f"Choisir fichier {import_type}", type=file_types_allowed, key=uploaded_file_dispo_ui_key)

    if uploaded_file_dispo_ui_val is not None:
        horaires_ok = st.session_state.get("horaires_par_jour") and isinstance(st.session_state.horaires_par_jour, dict) and st.session_state.horaires_par_jour
        etudiants_ok = st.session_state.get("etudiants") and isinstance(st.session_state.etudiants, list) and st.session_state.etudiants
        cojurys_ok = st.session_state.get("co_jurys") and isinstance(st.session_state.co_jurys, list) # Peut être vide

        if horaires_ok and etudiants_ok: # Cojurys peuvent être vides
            tuteurs_app_list_imp = list(set([e["Tuteur"] for e in st.session_state.etudiants if "Tuteur" in e]))
            cojurys_app_list_imp = st.session_state.co_jurys if cojurys_ok else []
            
            with st.spinner(f"Import {import_type}..."):
                s_msg, e_msg, w_msg = importer_func_selected(
                    uploaded_file_dispo_ui_val, st.session_state.horaires_par_jour, 
                    tuteurs_app_list_imp, cojurys_app_list_imp, score_matching_seuil=75
                )
            for m in s_msg: st.success(m)
            for m in e_msg: st.error(m)
            for m in w_msg: st.warning(m)
            # Pour éviter le ré-upload automatique lors du prochain re-render, on met à None la variable qui tient le fichier
            # Ceci nécessite que la clé de l'uploader soit stable.
            # Cependant, Streamlit gère le FileUploader de manière à ce qu'il ne se recharge pas sans interaction.
            # On va donc juste laisser l'UI se mettre à jour.
        else: st.error("Prérequis manquants (étapes créneaux/étudiants).")
    
    st.divider()
    st.subheader("✏️ Saisie manuelle / Vérification")
    
    tous_tuteurs_disp_ui = list(set([e["Tuteur"] for e in st.session_state.etudiants if "Tuteur" in e])) if st.session_state.etudiants else []
    co_jurys_disp_ui = st.session_state.co_jurys if st.session_state.co_jurys else []
    personnes_disp_ui = sorted(list(set(tous_tuteurs_disp_ui + co_jurys_disp_ui)))

    if not personnes_disp_ui or not st.session_state.horaires_par_jour:
        st.info("Aucun jury ou créneau défini pour la saisie des disponibilités.")
    else:
        for p_disp_init in personnes_disp_ui: # Init dispo dict si besoin
            if p_disp_init not in st.session_state.disponibilites: st.session_state.disponibilites[p_disp_init] = {}

        for personne_disp_loop in personnes_disp_ui:
            st.markdown(f"#### 👨‍🏫 Dispos de {personne_disp_loop}")
            dispos_p_actuelle = st.session_state.disponibilites.get(personne_disp_loop, {})
            for jour_disp_loop, creneaux_list_jour_disp in st.session_state.horaires_par_jour.items():
                if not creneaux_list_jour_disp: continue 
                st.markdown(f"**{jour_disp_loop}**")
                
                toutes_coches_jour_actuel = all(dispos_p_actuelle.get(f"{jour_disp_loop} | {c_d_l}", False) for c_d_l in creneaux_list_jour_disp) if creneaux_list_jour_disp else False
                
                jour_key_cleaned = "".join(filter(str.isalnum, jour_disp_loop))
                all_sel_key_ui = f"all_sel_{personne_disp_loop}_{jour_key_cleaned}"
                
                # Checkbox "Toute la journée"
                # Sa valeur est True si toutes les sous-checkboxes sont True DANS LE STATE
                # L'interaction utilisateur est capturée par `all_selected_interaction_val`
                all_selected_interaction_val = st.checkbox("Toute la journée", value=toutes_coches_jour_actuel, key=all_sel_key_ui)

                # Si l'utilisateur VIENT DE COCHER "Toute la journée"
                if all_selected_interaction_val and not toutes_coches_jour_actuel:
                    for c_d_l_force in creneaux_list_jour_disp:
                        st.session_state.disponibilites[personne_disp_loop][f"{jour_disp_loop} | {c_d_l_force}"] = True
                    # Forcer un re-render pour que les cases individuelles reflètent le changement
                    st.experimental_rerun() 
                
                # Si l'utilisateur VIENT DE DECOCHER "Toute la journée"
                if not all_selected_interaction_val and toutes_coches_jour_actuel:
                    for c_d_l_force in creneaux_list_jour_disp:
                         st.session_state.disponibilites[personne_disp_loop][f"{jour_disp_loop} | {c_d_l_force}"] = False
                    st.experimental_rerun()


                cols_disp_cb = st.columns(min(len(creneaux_list_jour_disp), 4))
                for i_disp_cb, creneau_val_disp_cb in enumerate(creneaux_list_jour_disp):
                    with cols_disp_cb[i_disp_cb % 4]:
                        key_dispo_individual_cb = f"{jour_disp_loop} | {creneau_val_disp_cb}"
                        # Lire la valeur actuelle du state pour cette checkbox
                        valeur_actuelle_cb_state = st.session_state.disponibilites[personne_disp_loop].get(key_dispo_individual_cb, False)
                        
                        creneau_key_cleaned_cb = "".join(filter(str.isalnum, creneau_val_disp_cb))
                        individual_cb_ui_key = f"cb_{personne_disp_loop}_{jour_key_cleaned}_{i_disp_cb}_{creneau_key_cleaned_cb}"
                        
                        # La checkbox individuelle est désactivée si "Toute la journée" est effectivement cochée
                        is_disabled_ind_cb = all_selected_interaction_val 

                        checked_individual_val = st.checkbox(
                            creneau_val_disp_cb,
                            value=valeur_actuelle_cb_state, # Afficher la valeur du state
                            key=individual_cb_ui_key,
                            disabled=is_disabled_ind_cb 
                        )
                        
                        # Mettre à jour le state SEULEMENT si cette checkbox a été modifiée par l'utilisateur
                        # ET que "Toute la journée" n'est pas active (pour éviter conflit d'update)
                        if not is_disabled_ind_cb: # Si elle n'est pas désactivée
                            if checked_individual_val != valeur_actuelle_cb_state: # Et que sa valeur a changé
                                 st.session_state.disponibilites[personne_disp_loop][key_dispo_individual_cb] = checked_individual_val
                                 # Si ce changement affecte l'état de "Toute la journée", il faut un rerun pour la mettre à jour
                                 st.experimental_rerun()
                st.markdown("---")
            st.divider() 

    if st.button("Suivant > Générer Planning", type="primary", key="dispo_sel_next_btn"):
        st.session_state.etape = "generation"; st.rerun()

elif st.session_state.etape == "generation":
    afficher_navigation()
    st.header(etapes_labels["generation"])
    
    # Checkbox pour AG, cochée par défaut
    ag_params_key_suffix = "_gen_etape" # Pour rendre les clés des sliders uniques
    utiliser_ag_ui = st.checkbox(
        "Utiliser l'algorithme génétique (recommandé)", value=True, 
        help="L'AG est plus performant. Sera aussi utilisé si l'algo classique < 85% de réussite.",
        key=f"utiliser_ag_checkbox{ag_params_key_suffix}"
    )
    params_ag_config_from_ui = {}
    if utiliser_ag_ui:
        with st.expander("⚙️ Paramètres de l'algorithme génétique", expanded=st.session_state.get(f"ag_expander_open{ag_params_key_suffix}", False) ):
            st.session_state[f"ag_expander_open{ag_params_key_suffix}"] = True # Garder ouvert si cliqué
            taille_pop_val = st.slider("Taille population AG", 20, 250, 80, key=f"ag_pop{ag_params_key_suffix}")
            nb_gen_val = st.slider("Nb générations AG", 20, 1000, 300, key=f"ag_gen{ag_params_key_suffix}")
            taux_mut_val = st.slider("Taux mutation AG", 0.05, 0.50, 0.15, step=0.01, key=f"ag_mut{ag_params_key_suffix}")
            taux_crois_val = st.slider("Taux croisement AG", 0.50, 0.95, 0.85, step=0.01, key=f"ag_crois{ag_params_key_suffix}")
            params_ag_config_from_ui = {
                'taille_population': taille_pop_val, 'nb_generations': nb_gen_val, 
                'taux_mutation': taux_mut_val, 'taux_croisement': taux_crois_val
            }
        
    if st.button("🚀 Lancer l'optimisation", type="primary", key="lancer_opti_final_btn"):
        # Vérifications de base
        if not st.session_state.etudiants: st.error("Aucun étudiant à planifier."); st.stop()
        if not st.session_state.dates_soutenance: st.error("Aucune date de soutenance définie."); st.stop()
        if not st.session_state.horaires_par_jour: st.error("Créneaux non générés (Étape 6)."); st.stop()

        with st.spinner("Optimisation en cours... Cela peut prendre du temps..."):
            optimiseur = PlanificationOptimiseeV2(
                st.session_state.etudiants, st.session_state.co_jurys, st.session_state.dates_soutenance,
                st.session_state.disponibilites, st.session_state.nb_salles, st.session_state.duree_soutenance
            )
            planning_final_opti, non_planifies_opti, stats_ag_opti = optimiseur.optimiser_avec_genetique(
                utiliser_genetique_ui=utiliser_ag_ui, **params_ag_config_from_ui
            )
            st.session_state.planning_final = planning_final_opti

        if stats_ag_opti: # Si l'AG a tourné
            st.subheader("🧬 Statistiques de l'Algorithme Génétique")
            # ... (affichage des stats AG comme avant)
            c1,c2,c3 = st.columns(3)
            c1.metric("Générations", stats_ag_opti.get('generations', 'N/A'))
            c2.metric("Fitness Finale", f"{stats_ag_opti.get('fitness_finale', 0.0):.1f}")
            c3.metric("Conflits (AG)", stats_ag_opti.get('conflits', 'N/A'))
            if 'amelioration_valeur' in stats_ag_opti and stats_ag_opti['amelioration_valeur'] > 0:
                st.success(f"AG a ajouté {stats_ag_opti['amelioration_valeur']} soutenances.")
            
            hist_data = stats_ag_opti.get('historique')
            if hist_data:
                import plotly.graph_objects as go
                df_hist = pd.DataFrame(hist_data)
                if not df_hist.empty and 'generation' in df_hist.columns:
                    fig_evol = go.Figure()
                    if 'fitness_max' in df_hist.columns: fig_evol.add_trace(go.Scatter(x=df_hist['generation'], y=df_hist['fitness_max'], mode='lines', name='Fitness Max'))
                    if 'soutenances_max' in df_hist.columns: fig_evol.add_trace(go.Scatter(x=df_hist['generation'], y=df_hist['soutenances_max'], mode='lines', name='Soutenances Max', yaxis='y2'))
                    fig_evol.update_layout(title="Évolution AG", xaxis_title="Génération", yaxis_title="Fitness", yaxis2=dict(title="Soutenances", overlaying='y', side='right'), height=350)
                    st.plotly_chart(fig_evol, use_container_width=True)


        if st.session_state.planning_final:
            conflits_planning = optimiseur.verifier_conflits(st.session_state.planning_final)
            if conflits_planning:
                st.error("⚠️ Conflits détectés dans le planning final :"); [st.write(f"- {c}") for c in conflits_planning]
            else: st.success("✅ Aucun conflit de base (salle/jury simultané) détecté.")

            st.success(f"Planning généré! {len(st.session_state.planning_final)} soutenances planifiées.")
            if non_planifies_opti > 0: st.warning(f"⚠️ {non_planifies_opti} étudiant(s) non planifiés.")

            df_planning_final_ui = pd.DataFrame(st.session_state.planning_final)
            st.subheader("📋 Planning détaillé")
            st.dataframe(df_planning_final_ui.drop(['Début', 'Fin'], axis=1, errors='ignore'), use_container_width=True, hide_index=True)

            if not df_planning_final_ui.empty:
                st.subheader("📊 Visualisation Gantt")
                # ... (Gantt comme avant)
                df_planning_final_ui["Task"] = df_planning_final_ui["Étudiant"] + " (" + df_planning_final_ui["Salle"] + ")"
                fig_gantt_ui = px.timeline(
                    df_planning_final_ui, x_start="Début", x_end="Fin", y="Tuteur", color="Task",
                    title="Planning par tuteur", hover_data=["Étudiant", "Co-jury", "Salle", "Pays"]
                )
                fig_gantt_ui.update_yaxes(autorange="reversed"); fig_gantt_ui.update_layout(height=max(500, len(df_planning_final_ui['Tuteur'].unique())*40 + 100 ))
                st.plotly_chart(fig_gantt_ui, use_container_width=True)
                
                st.subheader("📥 Exportation")
                # ... (Export comme avant)
                csv_export_ui = df_planning_final_ui.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger CSV", csv_export_ui, "planning_soutenances.csv", "text/csv", key="dl_csv_btn")
                
                output_excel_ui = BytesIO()
                with pd.ExcelWriter(output_excel_ui, engine='openpyxl') as writer_excel_ui:
                    df_planning_final_ui.to_excel(writer_excel_ui, index=False, sheet_name='Planning')
                st.download_button("Télécharger Excel", output_excel_ui.getvalue(), "planning_soutenances.xlsx", 
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_excel_btn")
        else:
            st.error("❌ Aucune soutenance n'a pu être planifiée. Vérifiez les contraintes et disponibilités.")


# Sidebar Résumé
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Résumé Actuel")
    # ... (Résumé comme avant)
    st.write(f"**Étudiants :** {len(st.session_state.etudiants)}")
    st.write(f"**Co-jurys :** {len(st.session_state.co_jurys)}")
    st.write(f"**Salles :** {st.session_state.nb_salles}")
    st.write(f"**Durée :** {st.session_state.duree_soutenance} min")
    if st.session_state.dates_soutenance:
        st.write(f"**Dates :** {len(st.session_state.dates_soutenance)} jour(s)")
    st.markdown("---")
    st.markdown("""
    ### ℹ️ À propos
    Planification de soutenances.
    ---
    © 2024-2025 - Polytech 4A MAM
    """)
