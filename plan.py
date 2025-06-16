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
from io import BytesIO


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
    etape_selectionnee = st.sidebar.selectbox(
        "Aller à une autre étape :",
        options=etapes,
        format_func=lambda x: etapes_labels.get(x, x),
        index=etapes.index(st.session_state.etape) if st.session_state.etape in etapes else 0,
        key="navigation_selectbox"
    )
    if etape_selectionnee != st.session_state.etape:
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
if "horaires_par_jour" not in st.session_state: # Ajout pour éviter erreur si on navigue direct
    st.session_state.horaires_par_jour = {}

# ... (début du fichier identique) ...

@dataclass
class Individu:
    """Représente un individu (solution) dans l'algorithme génétique"""
    genes: List[int]  # Pour chaque étudiant, l'index du créneau assigné (-1 si non assigné)
    fitness: float = 0.0
    soutenances_planifiees: int = 0
    conflits: int = 0


class AlgorithmeGenetique: # Début de la classe
    def __init__(self, planificateur, taille_population=80, nb_generations=800,
                 taux_mutation=0.12, taux_croisement=0.8):
        self.planificateur = planificateur
        self.taille_population = taille_population
        self.nb_generations = nb_generations
        self.taux_mutation = taux_mutation
        self.taux_croisement = taux_croisement

        self.creneaux = planificateur.generer_creneaux_uniques()
        self.creneaux_valides_par_etudiant = self._precalculer_creneaux_valides()
        self.nb_etudiants = len(planificateur.etudiants) if planificateur.etudiants else 0 # S'assurer que planificateur.etudiants n'est pas None

        self.historique_fitness = []
        self.meilleure_solution = None

    def _precalculer_creneaux_valides(self):
        creneaux_valides = {}
        if not self.planificateur.etudiants: # Ajout de garde
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
        if self.nb_etudiants == 0: # Ajout de garde
            return Individu(genes=genes)
            
        creneaux_occupes = set()
        jurys_par_moment = {} # {moment_str: {jury1, jury2}}
        ordre_etudiants = list(range(self.nb_etudiants))
        random.shuffle(ordre_etudiants)

        for idx_etu in ordre_etudiants:
            # S'assurer que self.planificateur.etudiants[idx_etu] est accessible
            if idx_etu >= len(self.planificateur.etudiants): continue

            creneaux_possibles_etu = self.creneaux_valides_par_etudiant.get(idx_etu, [])
            creneaux_possibles = creneaux_possibles_etu.copy()
            random.shuffle(creneaux_possibles)

            for idx_creneau in creneaux_possibles:
                # S'assurer que self.creneaux[idx_creneau] est accessible
                if idx_creneau >= len(self.creneaux): continue
                creneau = self.creneaux[idx_creneau]

                if idx_creneau in creneaux_occupes:
                    continue
                
                tuteur = self.planificateur.etudiants[idx_etu]["Tuteur"]
                moment = creneau['moment']

                if moment not in jurys_par_moment:
                    jurys_par_moment[moment] = set()
                
                if tuteur in jurys_par_moment[moment]:
                    continue

                co_jurys_disponibles = self.planificateur.trouver_co_jurys_disponibles(
                    tuteur, creneau['jour'], creneau['heure']
                )
                co_jury_libre = None
                random.shuffle(co_jurys_disponibles) # Introduire de l'aléatoire dans le choix du co-jury
                for co_jury_cand in co_jurys_disponibles:
                    if co_jury_cand not in jurys_par_moment[moment]:
                        co_jury_libre = co_jury_cand
                        break
                
                if co_jury_libre:
                    genes[idx_etu] = idx_creneau
                    creneaux_occupes.add(idx_creneau)
                    jurys_par_moment[moment].add(tuteur)
                    jurys_par_moment[moment].add(co_jury_libre)
                    break
        return Individu(genes=genes)

    # LA FONCTION EST MAINTENANT CORRECTEMENT INDENTÉE DANS LA CLASSE
    def calculer_fitness_amelioree(self, individu: Individu) -> Individu:
        planning = self.decoder_individu(individu)

        nb_soutenances = len(planning)
        nb_total_etudiants = self.nb_etudiants # Utiliser self.nb_etudiants qui est initialisé
        taux_planification = nb_soutenances / nb_total_etudiants if nb_total_etudiants > 0 else 0

        conflits_salle, conflits_jury_fitness = self._analyser_conflits_detailles(planning)
        total_conflits = conflits_salle + conflits_jury_fitness

        equilibrage = self._calculer_equilibrage_charge(planning)
        bonus_alternance = self._calculer_bonus_alternance(planning)

        # NOUVEAU: Calculer la balance Tuteur/Co-jury
        roles_par_jury = defaultdict(lambda: {'tuteur': 0, 'cojury': 0})
        for soutenance in planning:
            roles_par_jury[soutenance['Tuteur']]['tuteur'] += 1
            roles_par_jury[soutenance['Co-jury']]['cojury'] += 1

        penalite_balance_roles = 0
        score_balance_roles = 0 
        
        # Personnes qui peuvent être à la fois tuteur et co-jury
        personnes_eligibles_balance = set(self.planificateur.tuteurs_referents) & set(self.planificateur.co_jurys)

        for jury, counts in roles_par_jury.items():
            if jury in personnes_eligibles_balance:
                difference = abs(counts['tuteur'] - counts['cojury'])
                # Pénalité proportionnelle au carré de la différence pour accentuer les grands écarts
                penalite_balance_roles += (difference ** 2) * 20 # Ajuster le poids (ex: 20)
                
                if difference == 0:
                    score_balance_roles += 100 # Fort bonus pour équilibre parfait
                elif difference == 1:
                    score_balance_roles += 30  # Bonus plus faible pour un écart de 1

        # Fonction de fitness (à maximiser)
        fitness = (
                taux_planification * 1000 +
                max(0, (nb_soutenances - (nb_total_etudiants * 0.75))) * 50 + # Bonus si on planifie > 75%
                equilibrage * 30 +  # Augmenter un peu le poids de l'équilibrage
                bonus_alternance * 15 + # Augmenter un peu le poids de l'alternance
                score_balance_roles * 1.5 - # Bonus pour la balance des rôles, ajuster poids
                total_conflits * 1000 -  # Pénalité massive pour conflits
                (nb_total_etudiants - nb_soutenances) * 150 - # Pénalité pour non-planifiés
                penalite_balance_roles # Pénalité pour déséquilibre des rôles
        )
        
        if total_conflits == 0 and nb_soutenances >= nb_total_etudiants * 0.9: # Si >90% planifié sans conflit
            fitness += 2000 # Bonus significatif

        individu.fitness = fitness
        individu.soutenances_planifiees = nb_soutenances
        individu.conflits = total_conflits
        return individu

    def _analyser_conflits_detailles(self, planning):
        conflits_salle = 0
        conflits_jury = 0
        creneaux_salle = {} # key: "jour_heure_salle", value: True
        jurys_par_moment = {} # key: "jour_heure", value: set of jurys

        for soutenance in planning:
            cle_salle = f"{soutenance['Jour']}_{soutenance['Créneau']}_{soutenance['Salle']}"
            moment = f"{soutenance['Jour']}_{soutenance['Créneau']}"

            if cle_salle in creneaux_salle:
                conflits_salle += 1
            creneaux_salle[cle_salle] = True

            if moment not in jurys_par_moment:
                jurys_par_moment[moment] = set()

            tuteur = soutenance['Tuteur']
            co_jury = soutenance['Co-jury']

            # Un tuteur ne peut pas être à deux endroits en même temps
            if tuteur in jurys_par_moment[moment]:
                conflits_jury += 1
            else:
                jurys_par_moment[moment].add(tuteur)
            
            # Un co-jury ne peut pas être à deux endroits en même temps
            # et ne peut pas être le tuteur de la même soutenance (géré par la logique de sélection)
            if co_jury in jurys_par_moment[moment]:
                # Si le co-jury est le même que le tuteur et qu'on a déjà compté le conflit pour le tuteur,
                # on pourrait compter double. Mais ici, on ajoute au set, donc pas de double comptage si tuteur=cojury
                # Cependant, la logique de sélection de co-jury devrait empêcher tuteur == co-jury.
                conflits_jury += 1
            else:
                jurys_par_moment[moment].add(co_jury)
                
        return conflits_salle, conflits_jury

    # ... (les autres méthodes de AlgorithmeGenetique : _calculer_equilibrage_charge, _calculer_bonus_alternance,
    #      croisement_intelligent, mutation_adaptative, evoluer, selection_tournament, decoder_individu
    #      restent indentées comme méthodes de cette classe) ...
    def _calculer_equilibrage_charge(self, planning): # Doit être une méthode
        if not planning: return 0
        charges = defaultdict(int)
        for soutenance in planning:
            charges[soutenance['Tuteur']] += 1
            charges[soutenance['Co-jury']] += 1
        
        if not charges or len(charges) <= 1 : return 10 # Score max si peu de jurys ou pas de planning
        
        valeurs_charges = list(charges.values())
        moyenne = np.mean(valeurs_charges)
        variance = np.var(valeurs_charges)
        # Score inversement proportionnel à la variance, borné
        # Plus la variance est faible, meilleur est le score. Max 10.
        return max(0, 10 - np.sqrt(variance)) # Utiliser racine carrée pour modérer l'impact de la variance

    def _calculer_bonus_alternance(self, planning): # Doit être une méthode
        bonus = 0
        jurys_par_periode = {'matin': defaultdict(set), 'apres_midi': defaultdict(set)} # {periode: {jour: {jury1, ...}}}
        
        for soutenance in planning:
            jour = soutenance['Jour']
            debut_heure = soutenance['Début'].hour
            periode = 'matin' if debut_heure < 13 else 'apres_midi' # 13h comme limite pour matin

            jurys_par_periode[periode][jour].add(soutenance['Tuteur'])
            jurys_par_periode[periode][jour].add(soutenance['Co-jury'])

        total_jurys_alternant = 0
        jours_concernes = set(jurys_par_periode['matin'].keys()) | set(jurys_par_periode['apres_midi'].keys())

        for jour_alternance in jours_concernes:
            jurys_matin_ce_jour = jurys_par_periode['matin'].get(jour_alternance, set())
            jurys_aprem_ce_jour = jurys_par_periode['apres_midi'].get(jour_alternance, set())
            jurys_alternant_ce_jour = jurys_matin_ce_jour & jurys_aprem_ce_jour
            total_jurys_alternant += len(jurys_alternant_ce_jour)
            
        # Bonus proportionnel au nombre de jurys qui ont des soutenances matin ET après-midi (sur différents jours potentiellement)
        # Ou un bonus par jour où il y a une bonne alternance.
        # Ici, on compte le nombre total d'instances de jury alternant sur un jour.
        return total_jurys_alternant * 1 # Ajuster le poids du bonus

    def croisement_intelligent(self, parent1: Individu, parent2: Individu) -> Tuple[Individu, Individu]: # Doit être une méthode
        len_genes = len(parent1.genes)
        if len_genes == 0: # Garde si les parents sont vides
             return Individu(genes=[]), Individu(genes=[])

        enfant1_genes = [-1] * len_genes
        enfant2_genes = [-1] * len_genes
        
        # S'assurer que le point de croisement est valide même si len_genes est petit
        point_croisement = random.randint(0, len_genes -1) if len_genes > 0 else 0


        # Copier la première partie
        for i in range(point_croisement):
            enfant1_genes[i] = parent1.genes[i]
            enfant2_genes[i] = parent2.genes[i]

        # Gérer les créneaux déjà utilisés dans la première partie des enfants
        creneaux_utilises_e1 = set(g for g in enfant1_genes[:point_croisement] if g != -1)
        creneaux_utilises_e2 = set(g for g in enfant2_genes[:point_croisement] if g != -1)

        # Pour la deuxième partie, essayer de prendre de l'autre parent en évitant les doublons de créneaux
        # et en s'assurant que l'étudiant à cet index peut bien prendre le créneau de l'autre parent.
        for i in range(point_croisement, len_genes):
            # Pour enfant1, envisager gene de parent2
            gene_p2 = parent2.genes[i]
            if gene_p2 != -1 and gene_p2 not in creneaux_utilises_e1 and \
               (i in self.creneaux_valides_par_etudiant and gene_p2 in self.creneaux_valides_par_etudiant[i]):
                enfant1_genes[i] = gene_p2
                creneaux_utilises_e1.add(gene_p2)
            else: # Sinon, essayer de garder le gene du parent1 s'il est valide et non utilisé
                gene_p1_part2 = parent1.genes[i]
                if gene_p1_part2 != -1 and gene_p1_part2 not in creneaux_utilises_e1 and \
                   (i in self.creneaux_valides_par_etudiant and gene_p1_part2 in self.creneaux_valides_par_etudiant[i]):
                    enfant1_genes[i] = gene_p1_part2
                    creneaux_utilises_e1.add(gene_p1_part2)


            # Pour enfant2, envisager gene de parent1
            gene_p1 = parent1.genes[i]
            if gene_p1 != -1 and gene_p1 not in creneaux_utilises_e2 and \
               (i in self.creneaux_valides_par_etudiant and gene_p1 in self.creneaux_valides_par_etudiant[i]):
                enfant2_genes[i] = gene_p1
                creneaux_utilises_e2.add(gene_p1)
            else: # Sinon, essayer de garder le gene du parent2 s'il est valide et non utilisé
                gene_p2_part2 = parent2.genes[i]
                if gene_p2_part2 != -1 and gene_p2_part2 not in creneaux_utilises_e2 and \
                   (i in self.creneaux_valides_par_etudiant and gene_p2_part2 in self.creneaux_valides_par_etudiant[i]):
                    enfant2_genes[i] = gene_p2_part2
                    creneaux_utilises_e2.add(gene_p2_part2)
                    
        return Individu(genes=enfant1_genes), Individu(genes=enfant2_genes)

    def mutation_adaptative(self, individu: Individu) -> Individu: # Doit être une méthode
        if not individu.genes: return individu # Garde

        for i in range(len(individu.genes)):
            if random.random() < self.taux_mutation:
                creneaux_possibles_etu = self.creneaux_valides_par_etudiant.get(i, [])
                if not creneaux_possibles_etu: continue

                gene_actuel = individu.genes[i]
                
                # Créer un set des gènes actuellement utilisés par d'autres étudiants
                autres_genes_utilises = set(individu.genes[j] for j in range(len(individu.genes)) if j != i and individu.genes[j] != -1)

                creneaux_libres_pour_mutation = [c for c in creneaux_possibles_etu if c not in autres_genes_utilises]

                if not creneaux_libres_pour_mutation: # Aucun créneau libre valide pour cet étudiant
                    if gene_actuel != -1 and gene_actuel not in autres_genes_utilises:
                        # Si l'étudiant est déjà planifié et son créneau est unique, on peut le laisser
                        # ou essayer de le déplanifier pour libérer son créneau (plus agressif)
                        # individu.genes[i] = -1 # Optionnel: déplanifier si pas d'autre option
                        pass 
                    else: # Si non planifié ou son créneau est déjà pris, et pas d'option, on ne fait rien
                         individu.genes[i] = -1 # Assurer qu'il est marqué comme non planifié s'il y a conflit

                    continue

                if gene_actuel == -1: # Si l'étudiant n'est pas planifié, essayer de le planifier
                    individu.genes[i] = random.choice(creneaux_libres_pour_mutation)
                else: # Sinon, essayer de changer vers un *autre* meilleur créneau libre
                    # Exclure le créneau actuel des choix pour forcer un changement s'il y a d'autres options
                    options_changement = [c for c in creneaux_libres_pour_mutation if c != gene_actuel]
                    if options_changement:
                        individu.genes[i] = random.choice(options_changement)
                    # Si la seule option libre est son créneau actuel, on le laisse.
                    elif gene_actuel in creneaux_libres_pour_mutation : # (ce qui signifie que creneaux_libres_pour_mutation ne contenait que gene_actuel)
                        individu.genes[i] = gene_actuel 
                    else: # Cas étrange, l'étudiant est planifié mais son créneau n'est pas dans les libres, conflit potentiel
                         individu.genes[i] = -1 # Déplanifier pour résoudre
        return individu

    def evoluer(self) -> Tuple[List[Dict], Dict]: # Doit être une méthode
        population = []
        if self.nb_etudiants == 0: # Si pas d'étudiants, pas de population
             st.warning("AG: Aucun étudiant à planifier, population vide.")
             return [], {'generations': 0, 'fitness_finale': 0, 'soutenances_planifiees': 0, 'conflits': 0, 'taux_reussite': 0, 'historique': []}

        for _ in range(self.taille_population):
            individu = self.generer_individu_intelligent()
            population.append(self.calculer_fitness_amelioree(individu))

        if not population :
            st.warning("AG: Impossible de générer une population initiale viable.")
            return [], {'generations': 0, 'fitness_finale': 0, 'soutenances_planifiees': 0, 'conflits': 0, 'taux_reussite': 0, 'historique': []}

        self.meilleure_solution = max(population, key=lambda x: x.fitness)
        stagnation = 0

        for generation in range(self.nb_generations):
            nouvelle_population = []
            population_triee = sorted(population, key=lambda x: x.fitness, reverse=True)
            
            elite_size = max(1, int(self.taille_population * 0.1)) # 10% d'élitisme, au moins 1
            nouvelle_population.extend(population_triee[:elite_size])

            # Remplissage de la nouvelle population
            while len(nouvelle_population) < self.taille_population:
                if random.random() < self.taux_croisement and len(population) >= 2 :
                    parent1 = self.selection_tournament(population, k=5)
                    parent2 = self.selection_tournament(population, k=5)
                    if parent1.genes and parent2.genes: # S'assurer que les parents ne sont pas vides
                        enfant1, enfant2 = self.croisement_intelligent(parent1, parent2)
                        nouvelle_population.append(self.mutation_adaptative(enfant1))
                        if len(nouvelle_population) < self.taille_population:
                             nouvelle_population.append(self.mutation_adaptative(enfant2))
                else: # Mutation ou immigration
                    if len(population) > 0:
                        individu_mute = self.mutation_adaptative(random.choice(population)) # Prendre un individu au hasard pour le muter
                        nouvelle_population.append(individu_mute)
                    else: # Si population de base est vide, générer un nouveau
                         nouvelle_population.append(self.generer_individu_intelligent())
            
            nouvelle_population = nouvelle_population[:self.taille_population] # S'assurer de la taille
            population = [self.calculer_fitness_amelioree(ind) for ind in nouvelle_population]

            if not population: # Si après évaluation, la population est vide (tous fitness inf?)
                st.warning(f"AG: Population vide à la génération {generation}.")
                break 

            meilleur_actuel_gen = max(population, key=lambda x: x.fitness, default=self.meilleure_solution)

            if meilleur_actuel_gen.fitness > self.meilleure_solution.fitness:
                self.meilleure_solution = meilleur_actuel_gen
                stagnation = 0
            else:
                stagnation += 1

            # Relance si stagnation (diversification)
            if stagnation > max(30, self.nb_generations * 0.1) and generation < self.nb_generations * 0.8: # Stagnation sur 10% des gen ou 30
                nb_a_remplacer = int(self.taille_population * 0.3) # Remplacer 30%
                population_moins_bons = sorted(population, key=lambda x: x.fitness)[:nb_a_remplacer]
                population_meilleurs = sorted(population, key=lambda x: x.fitness, reverse=True)[nb_a_remplacer:]
                
                nouveaux_individus_gen = [self.generer_individu_intelligent() for _ in range(nb_a_remplacer)]
                population = population_meilleurs + [self.calculer_fitness_amelioree(n_ind) for n_ind in nouveaux_individus_gen]
                random.shuffle(population)
                stagnation = 0
                if self.nb_generations > 20 : # Éviter log sur très peu de générations
                    st.sidebar.text(f"AG: Relance à la gén. {generation}")


            fitness_moyenne_gen = np.mean([ind.fitness for ind in population]) if population else 0
            self.historique_fitness.append({
                'generation': generation,
                'fitness_max': self.meilleure_solution.fitness, # Toujours le meilleur global
                'fitness_moyenne': fitness_moyenne_gen,
                'soutenances_max': self.meilleure_solution.soutenances_planifiees,
                'conflits_min': self.meilleure_solution.conflits
            })
            if generation % max(1, (self.nb_generations // 20)) == 0: 
                 st.sidebar.text(f"G:{generation} FitMax:{self.meilleure_solution.fitness:.0f} S:{self.meilleure_solution.soutenances_planifiees} C:{self.meilleure_solution.conflits}")

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

    def selection_tournament(self, population: List[Individu], k=3) -> Individu: # Doit être une méthode
        if not population:
            return Individu(genes=[]) # Cas où la population est vide
        
        # S'assurer que k n'est pas plus grand que la taille de la population
        k_valide = min(k, len(population))
        if k_valide == 0: # Si la population devient vide après le min(k, len)
             return population[0] if population else Individu(genes=[])


        participants = random.sample(population, k_valide)
        return max(participants, key=lambda x: x.fitness)

    def decoder_individu(self, individu: Individu) -> List[Dict]: # Doit être une méthode
        planning = []
        jurys_occupes_decode_moment = defaultdict(set) # {moment_str: {jury1, jury2}}
        creneaux_salles_decode_moment = defaultdict(set) # {moment_str_salle: True}

        if not individu.genes: return []

        # Trier les gènes par index de créneau pour essayer de placer les soutenances tôt d'abord ?
        # Ou traiter dans l'ordre des étudiants. L'ordre actuel est celui des étudiants.
        
        for idx_etu, idx_creneau_gene in enumerate(individu.genes):
            if idx_creneau_gene == -1 or idx_creneau_gene >= len(self.creneaux):
                continue
            
            # Vérifier si l'étudiant est valide
            if idx_etu >= len(self.planificateur.etudiants): continue
            etudiant_obj = self.planificateur.etudiants[idx_etu]
            creneau_obj_decode = self.creneaux[idx_creneau_gene]
            tuteur_principal = etudiant_obj["Tuteur"]
            
            moment_str_decode = creneau_obj_decode['moment']
            moment_salle_str_decode = f"{moment_str_decode}_{creneau_obj_decode['salle']}"

            # 1. Vérifier si la salle est déjà prise à ce moment précis
            if moment_salle_str_decode in creneaux_salles_decode_moment:
                continue # Conflit de salle DANS cet individu, ignorer cette soutenance

            # 2. Vérifier si le tuteur est déjà pris à ce moment (par une autre soutenance de cet individu)
            if tuteur_principal in jurys_occupes_decode_moment[moment_str_decode]:
                continue # Conflit de tuteur DANS cet individu, ignorer

            # 3. Trouver un co-jury disponible pour ce créneau et ce tuteur, et qui n'est pas déjà pris
            co_jurys_possibles_decode = self.planificateur.trouver_co_jurys_disponibles(
                tuteur_principal, creneau_obj_decode['jour'], creneau_obj_decode['heure']
            )
            
            co_jury_final_choisi = None
            random.shuffle(co_jurys_possibles_decode) # Introduire de l'aléatoire dans le choix du co-jury
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
                # Marquer ressources comme utilisées pour les prochaines soutenances de CET individu
                creneaux_salles_decode_moment[moment_salle_str_decode] = True
                jurys_occupes_decode_moment[moment_str_decode].add(tuteur_principal)
                jurys_occupes_decode_moment[moment_str_decode].add(co_jury_final_choisi)
        return planning

# FIN DE LA CLASSE AlgorithmeGenetique


class PlanificationOptimiseeV2:
    # ... (le reste de la classe PlanificationOptimiseeV2 et de l'UI Streamlit) ...
    # Assurez-vous que l'initialisation de charge_jurys_tuteur est bien présente
    def __init__(self, etudiants, co_jurys, dates, disponibilites, nb_salles, duree):
        self.etudiants = etudiants if etudiants else []
        self.co_jurys = co_jurys if co_jurys else []
        self.dates = dates if dates else []
        self.disponibilites = disponibilites if disponibilites else {}
        self.nb_salles = nb_salles
        self.duree = duree

        self.tuteurs_referents = list(set([e["Tuteur"] for e in self.etudiants])) if self.etudiants else []
        self.tous_jurys = list(set(self.tuteurs_referents + self.co_jurys))
        
        # Charges pour l'algorithme classique et la sélection de co-jury
        self.charge_jurys_tuteur = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_cojury = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_total = {jury: 0 for jury in self.tous_jurys} # Pour l'équilibrage général

    def generer_creneaux_uniques(self): # Doit être une méthode
        creneaux = []
        creneau_id = 0
        if not self.dates: return [] # Garde

        for jour_obj in self.dates:
            jour_str_app = jour_obj.strftime("%A %d/%m/%Y")
            for periode in [("08:00", "13:00"), ("14:00", "18:10")]: # Soir étendu
                try:
                    debut_dt_obj = datetime.strptime(periode[0], "%H:%M").time()
                    fin_dt_obj = datetime.strptime(periode[1], "%H:%M").time()
                except ValueError: continue # Ignorer période mal formatée

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

    def est_disponible(self, personne, jour_str_app, heure_str_app): # Doit être une méthode
        key = f"{jour_str_app} | {heure_str_app}"
        return self.disponibilites.get(personne, {}).get(key, False)

    def trouver_co_jurys_disponibles(self, tuteur_referent, jour_str_app, heure_str_app): # Doit être une méthode
        co_jurys_dispo = []
        for jury_cand in self.tous_jurys:
            if jury_cand != tuteur_referent and self.est_disponible(jury_cand, jour_str_app, heure_str_app):
                co_jurys_dispo.append(jury_cand)
        
        # Tri pour la balance des rôles et ensuite par charge totale
        def sort_key_balance_cojury(jury_c):
            # Prioriser ceux qui ont été plus tuteurs que co-jurys (différence positive)
            diff_roles = self.charge_jurys_tuteur.get(jury_c, 0) - self.charge_jurys_cojury.get(jury_c, 0)
            charge_tot = self.charge_jurys_total.get(jury_c, 0)
            # On veut trier par diff_roles décroissant (pour prendre ceux qui ont "besoin" d'être co-jury)
            # puis par charge_totale croissante
            return (-diff_roles, charge_tot)

        co_jurys_dispo.sort(key=sort_key_balance_cojury)
        return co_jurys_dispo

    def optimiser_planning_ameliore(self): # Doit être une méthode
        # Réinitialisation des compteurs de charge pour cet appel
        self.charge_jurys_tuteur = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_cojury = {jury: 0 for jury in self.tous_jurys}
        self.charge_jurys_total = {jury: 0 for jury in self.tous_jurys}

        creneaux = self.generer_creneaux_uniques()
        planning = []
        creneaux_occupes_ids = set() # Par ID de créneau (salle incluse)
        jurys_par_moment_app = defaultdict(set) # {moment_str: {jury1, jury2}}
        
        if not self.etudiants: return [], 0 # Pas d'étudiants
        
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

                co_jurys_possibles_classique = self.trouver_co_jurys_disponibles( # Utilise le tri par balance
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
                    
                    # Mise à jour des charges
                    self.charge_jurys_tuteur[tuteur_ref_classique] += 1
                    self.charge_jurys_cojury[co_jury_choisi_classique] += 1
                    self.charge_jurys_total[tuteur_ref_classique] +=1
                    self.charge_jurys_total[co_jury_choisi_classique] +=1
                    
                    soutenance_planifiee_etu_classique = True
                    break
            
            if not soutenance_planifiee_etu_classique:
                non_planifies_count += 1
                st.sidebar.warning(f"Classique: {etudiant_obj_classique['Prénom']} non planifié.")
        
        return planning, non_planifies_count

    def optimiser_avec_genetique(self, utiliser_genetique_ui=False, **params_genetique_ui): # Doit être une méthode
        planning_classique, non_planifies_classique = self.optimiser_planning_ameliore()
        nb_etudiants_total = len(self.etudiants) if self.etudiants else 0
        taux_reussite_classique = (len(planning_classique) / nb_etudiants_total) if nb_etudiants_total > 0 else 0

        if utiliser_genetique_ui or (nb_etudiants_total > 0 and taux_reussite_classique < 0.85): # Seuil légèrement augmenté
            st.info("🧬 Lancement de l'optimisation génétique...")
            config_ag = { # Défauts pour AG
                'taille_population': 80, 'nb_generations': 300, 
                'taux_mutation': 0.15, 'taux_croisement': 0.85, # Taux légèrement ajustés
                **params_genetique_ui # UI écrase les défauts
            }
            # S'assurer que les paramètres de l'AG sont valides
            config_ag['taille_population'] = max(10, config_ag['taille_population']) # Min pop size
            config_ag['nb_generations'] = max(10, config_ag['nb_generations'])     # Min generations


            ag_instance = AlgorithmeGenetique(self, **config_ag)
            planning_genetique, stats_ag = ag_instance.evoluer()
            stats_ag['amelioration_valeur'] = 0 # Init

            # Choisir le meilleur planning
            if len(planning_genetique) > len(planning_classique):
                st.success(f"✅ AG a amélioré: {len(planning_genetique)} vs {len(planning_classique)} (classique)")
                stats_ag['amelioration_valeur'] = len(planning_genetique) - len(planning_classique)
                return planning_genetique, nb_etudiants_total - len(planning_genetique), stats_ag
            elif len(planning_genetique) == len(planning_classique) and stats_ag.get('fitness_finale', -float('inf')) > -100000 : # Si AG a un score décent
                 # Comparer la fitness si le nombre de planifiés est le même (nécessiterait de calculer fitness du classique)
                 # Pour l'instant, on favorise l'AG s'il fait aussi bien et a des stats.
                 st.info(f"AG a planifié autant ({len(planning_genetique)}). On garde le résultat de l'AG.")
                 return planning_genetique, nb_etudiants_total - len(planning_genetique), stats_ag
            else:
                st.info("ℹ️ AG n'a pas amélioré. Résultat classique conservé.")
                return planning_classique, non_planifies_classique, stats_ag # Retourner stats_ag pour info
        
        return planning_classique, non_planifies_classique, None # AG non déclenché

    def afficher_diagnostics(self, planning, tentatives_par_etudiant): # Doit être une méthode
        # (Optionnel, peut être verbeux)
        if not st.session_state.get('show_diagnostics_expanded', False): return # Contrôlable par un expander/checkbox

        st.subheader("🔍 Diagnostics de l'optimisation")
        # ... (le reste de la logique d'affichage des diagnostics)
        pass

    def verifier_conflits(self, planning): # Doit être une méthode
        conflits_messages = []
        # ... (Logique de vérification des conflits, qui semble correcte) ...
        creneaux_salles_occupes = defaultdict(list) # moment_salle -> [etudiants]
        jurys_moments_occupes = defaultdict(list)   # moment -> [(jury, etudiant)]

        for idx, soutenance in enumerate(planning):
            cle_moment_salle = f"{soutenance['Jour']}_{soutenance['Créneau']}_{soutenance['Salle']}"
            cle_moment = f"{soutenance['Jour']}_{soutenance['Créneau']}"

            creneaux_salles_occupes[cle_moment_salle].append(soutenance['Étudiant'])
            # Stocker le nom du jury et l'étudiant associé pour des messages de conflit plus clairs
            jurys_moments_occupes[cle_moment].extend([
                (soutenance['Tuteur'], soutenance['Étudiant']),
                (soutenance['Co-jury'], soutenance['Étudiant'])
            ])

        for moment_salle, etudiants_conflit in creneaux_salles_occupes.items():
            if len(etudiants_conflit) > 1:
                conflits_messages.append(f"Salle: {moment_salle} surbookée ({', '.join(etudiants_conflit)})")

        for moment, jurys_affectes_avec_etu in jurys_moments_occupes.items():
            compteur_jurys_moment = defaultdict(list) # jury -> [etudiants_associés]
            for jury, etudiant_associe in jurys_affectes_avec_etu:
                compteur_jurys_moment[jury].append(etudiant_associe)
            
            for jury, etudiants_pour_jury in compteur_jurys_moment.items():
                if len(etudiants_pour_jury) > 1:
                    # Message plus clair: Le Jury X est assigné à EtudiantA ET EtudiantB au même moment
                    etudiants_str = " et ".join(f"'{etu}'" for etu in etudiants_pour_jury)
                    conflits_messages.append(f"Jury: {jury} à {moment} pour {etudiants_str}")
        return conflits_messages

# ... (le reste du fichier : importer_disponibilites_excel et toute l'UI Streamlit) ...
# (Aucun changement nécessaire dans l'UI pour cette modification de la fitness et de la sélection de co-jury)

def importer_disponibilites_excel(uploaded_file, horaires_par_jour_app_config, tous_tuteurs_app, co_jurys_app):
    messages_succes, messages_erreur, messages_warning = [], [], []
    personnes_traitees_import = set()
    personnes_reconnues_app_set = set(tous_tuteurs_app + co_jurys_app)
    cles_dispo_valides_app_set = set()
    for jour_app_cfg, creneaux_list_cfg in horaires_par_jour_app_config.items():
        for creneau_app_cfg in creneaux_list_cfg:
            cles_dispo_valides_app_set.add(f"{jour_app_cfg} | {creneau_app_cfg}")

    try:
        df_excel = pd.read_excel(uploaded_file, header=[0, 1], index_col=0, sheet_name=0)
        if 'FILIERE' in df_excel.columns.get_level_values(0):
            df_excel = df_excel.drop(columns='FILIERE', level=0)

        for nom_enseignant_excel_raw, row_data in df_excel.iterrows():
            nom_enseignant_clean = str(nom_enseignant_excel_raw).strip()
            if not nom_enseignant_clean: continue
            if nom_enseignant_clean not in personnes_reconnues_app_set:
                messages_warning.append(f"Enseignant Excel '{nom_enseignant_clean}' non reconnu. Ignoré.")
                continue
            
            personnes_traitees_import.add(nom_enseignant_clean)
            if nom_enseignant_clean not in st.session_state.disponibilites:
                st.session_state.disponibilites[nom_enseignant_clean] = {}
            
            # Optionnel : réinitialiser les dispos de la personne avant import
            # st.session_state.disponibilites[nom_enseignant_clean] = {}


            for multi_col_hdr, dispo_value in row_data.items():
                date_excel_raw, creneau_excel_raw = multi_col_hdr
                try:
                    date_obj_excel_parsed = pd.to_datetime(date_excel_raw).date() # Prend la partie date
                    jour_format_app_excel = date_obj_excel_parsed.strftime("%A %d/%m/%Y")
                except Exception: # Large exception pour parsing de date
                    messages_warning.append(f"Format date Excel '{date_excel_raw}' non reconnu. Ignoré.")
                    continue
                
                creneau_format_app_excel = str(creneau_excel_raw).replace(" Ä ", " - ").replace(" À ", " - ").strip()
                cle_dispo_excel_format_app = f"{jour_format_app_excel} | {creneau_format_app_excel}"

                if cle_dispo_excel_format_app not in cles_dispo_valides_app_set:
                    # messages_warning.append(f"Créneau Excel '{cle_dispo_excel_format_app}' non valide pour config app. Ignoré.")
                    continue
                try:
                    disponibilite_bool = bool(int(dispo_value))
                    st.session_state.disponibilites[nom_enseignant_clean][cle_dispo_excel_format_app] = disponibilite_bool
                except ValueError:
                    if not pd.isna(dispo_value):
                        messages_erreur.append(f"Valeur dispo '{dispo_value}' non valide pour '{nom_enseignant_clean}' "
                                               f"à '{cle_dispo_excel_format_app}'. Doit être 0 ou 1.")
        
        for personne_nettoyage in personnes_traitees_import:
            dispos_personne_actuelles = st.session_state.disponibilites.get(personne_nettoyage, {})
            cles_a_retirer = [k for k in dispos_personne_actuelles if k not in cles_dispo_valides_app_set]
            for k_retirer in cles_a_retirer:
                del st.session_state.disponibilites[personne_nettoyage][k_retirer]

        if personnes_traitees_import:
            messages_succes.append(f"Dispos importées pour {len(personnes_traitees_import)} personnes.")
        else:
            messages_warning.append("Aucune personne reconnue traitée depuis l'Excel.")
    except Exception as e_global:
        messages_erreur.append(f"Erreur import Excel: {e_global}")
    return messages_succes, messages_erreur, messages_warning

# --- Interface utilisateur ---

# Importation Excel
st.sidebar.header("📥 Importation Excel des Données de Base")
excel_file_base = st.sidebar.file_uploader("Importer étudiants et co-jurys", type=["xlsx"], key="excel_base_uploader")
if excel_file_base:
    try:
        excel_data_base = pd.read_excel(excel_file_base, sheet_name=None)
        if "etudiants" in excel_data_base:
            etu_df = excel_data_base["etudiants"]
            required_cols_etu = {"Nom", "Prénom", "Pays", "Tuteur"}
            if required_cols_etu.issubset(etu_df.columns):
                st.session_state.etudiants = etu_df[list(required_cols_etu)].to_dict(orient="records")
                st.sidebar.success(f"{len(st.session_state.etudiants)} étudiants importés.")
            else:
                st.sidebar.error("Feuille 'etudiants': colonnes Nom, Prénom, Pays, Tuteur manquantes.")
        if "co_jurys" in excel_data_base:
            cj_df = excel_data_base["co_jurys"]
            if "Nom" in cj_df.columns:
                st.session_state.co_jurys = cj_df["Nom"].dropna().astype(str).tolist()
                st.sidebar.success(f"{len(st.session_state.co_jurys)} co-jurys importés.")
            else:
                st.sidebar.error("Feuille 'co_jurys': colonne 'Nom' manquante.")
    except Exception as e_import_base:
        st.sidebar.error(f"Erreur lecture Excel base: {e_import_base}")


if st.session_state.etape == "etudiants":
    afficher_navigation()
    st.header(etapes_labels["etudiants"])
    with st.form("ajout_etudiant_form"):
        # ... (code inchangé pour ajout étudiant) ...
        nom, prenom = st.text_input("Nom"), st.text_input("Prénom")
        pays, tuteur = st.text_input("Pays"), st.text_input("Tuteur")
        if st.form_submit_button("Ajouter étudiant") and all([nom, prenom, pays, tuteur]):
            st.session_state.etudiants.append({"Nom": nom, "Prénom": prenom, "Pays": pays, "Tuteur": tuteur})
            st.success(f"Étudiant {prenom} {nom} ajouté.")
            st.rerun()

    if st.session_state.etudiants:
        st.subheader("Liste des étudiants")
        st.dataframe(pd.DataFrame(st.session_state.etudiants), use_container_width=True)
    if st.button("Suivant > Salles", type="primary", key="etu_suivant"):
        if st.session_state.etudiants:
            st.session_state.etape = "salles"; st.rerun()
        else: st.error("Ajoutez au moins un étudiant.")

elif st.session_state.etape == "salles":
    afficher_navigation()
    st.header(etapes_labels["salles"])
    nb_salles_input = st.number_input("Nombre de salles", 1, 10, st.session_state.nb_salles, 1, key="nb_salles_input")
    if st.button("Valider > Durée", type="primary", key="salles_valider"):
        st.session_state.nb_salles = nb_salles_input
        st.session_state.etape = "duree_soutenance"; st.rerun()

elif st.session_state.etape == "duree_soutenance":
    afficher_navigation()
    st.header(etapes_labels["duree_soutenance"])
    duree_input = st.number_input("Durée soutenance (min)", 30, 120, st.session_state.duree_soutenance, 10, key="duree_input")
    if st.button("Valider > Co-jurys", type="primary", key="duree_valider"):
        st.session_state.duree_soutenance = duree_input
        st.session_state.etape = "co_jury"; st.rerun()

elif st.session_state.etape == "co_jury":
    afficher_navigation()
    st.header(etapes_labels["co_jury"])
    with st.form("ajout_cojury_form"):
        nom_cj = st.text_input("Nom du co-jury")
        if st.form_submit_button("Ajouter co-jury") and nom_cj:
            if nom_cj not in st.session_state.co_jurys:
                st.session_state.co_jurys.append(nom_cj); st.success(f"Co-jury {nom_cj} ajouté."); st.rerun()
            else: st.warning("Co-jury déjà existant.")
    if st.session_state.co_jurys:
        st.subheader("Liste des co-jurys")
        for idx_cj, cj_nom in enumerate(st.session_state.co_jurys):
            col1_cj, col2_cj = st.columns([3,1])
            col1_cj.write(f"👨‍🏫 {cj_nom}")
            if col2_cj.button("Suppr.", key=f"cj_suppr_{idx_cj}"):
                del st.session_state.co_jurys[idx_cj]; st.rerun()
    if st.button("Suivant > Dates", type="primary", key="cojury_suivant"):
        st.session_state.etape = "dates"; st.rerun()

elif st.session_state.etape == "dates":
    afficher_navigation()
    st.header(etapes_labels["dates"])
    nb_jours_input = st.number_input("Nombre de jours de soutenances", 1, 10, len(st.session_state.dates_soutenance) or 2, key="nb_jours_sout_input")
    dates_saisie = []
    for i in range(nb_jours_input):
        default_date = st.session_state.dates_soutenance[i] if i < len(st.session_state.dates_soutenance) else datetime.now().date() + timedelta(days=i)
        dates_saisie.append(st.date_input(f"Date Jour {i+1}", value=default_date, key=f"date_sout_{i}"))
    if st.button("Valider > Créneaux", type="primary", key="dates_valider"):
        st.session_state.dates_soutenance = dates_saisie
        st.session_state.etape = "disponibilites"; st.rerun()

elif st.session_state.etape == "disponibilites": # Génération et affichage des créneaux de l'app
    afficher_navigation()
    st.header(etapes_labels["disponibilites"])
    if st.session_state.dates_soutenance:
        horaires_par_jour_cfg = {}
        for jour_obj_cfg in st.session_state.dates_soutenance:
            jour_str_cfg = jour_obj_cfg.strftime("%A %d/%m/%Y")
            creneaux_cfg = []
            # Étendu à 18h10
            for (debut_cfg, fin_cfg) in [("08:00", "13:00"), ("14:00", "18:10")]:
                current_cfg_dt = datetime.combine(jour_obj_cfg, datetime.strptime(debut_cfg, "%H:%M").time())
                end_cfg_dt = datetime.combine(jour_obj_cfg, datetime.strptime(fin_cfg, "%H:%M").time())
                while current_cfg_dt + timedelta(minutes=st.session_state.duree_soutenance) <= end_cfg_dt:
                    fin_creneau_cfg_dt = current_cfg_dt + timedelta(minutes=st.session_state.duree_soutenance)
                    creneaux_cfg.append(f"{current_cfg_dt.strftime('%H:%M')} - {fin_creneau_cfg_dt.strftime('%H:%M')}")
                    current_cfg_dt = fin_creneau_cfg_dt
            horaires_par_jour_cfg[jour_str_cfg] = creneaux_cfg
        st.session_state.horaires_par_jour = horaires_par_jour_cfg # Stocker pour l'étape suivante

        for jour_disp_cfg, slots_disp_cfg in st.session_state.horaires_par_jour.items():
            st.subheader(f"📅 {jour_disp_cfg}")
            if slots_disp_cfg:
                cols_disp_cfg = st.columns(min(len(slots_disp_cfg), 5)) # Jusqu'à 5 créneaux par ligne
                for i_slot, slot_val in enumerate(slots_disp_cfg):
                    with cols_disp_cfg[i_slot % 5]: st.info(f"🕒 {slot_val}")
            else: st.write("Aucun créneau pour ce jour avec la durée spécifiée.")

        if st.button("Suivant > Saisie Disponibilités", type="primary", key="creneaux_suivant"):
            st.session_state.etape = "disponibilites_selection"; st.rerun()
    else: st.warning("Veuillez d'abord définir les dates des soutenances.")

elif st.session_state.etape == "disponibilites_selection":
    afficher_navigation()
    st.header(etapes_labels["disponibilites_selection"])
    st.subheader("⬇️ Importer les disponibilités depuis Excel")
    # ... (code de l'uploader et appel à importer_disponibilites_excel comme défini précédemment) ...
    uploaded_file_dispo_ui = st.file_uploader("Choisir Excel pour disponibilités", type=["xlsx", "xls"], key="excel_dispo_uploader")
    if uploaded_file_dispo_ui:
        if st.session_state.horaires_par_jour and st.session_state.etudiants and st.session_state.co_jurys:
            tuteurs_app_list = list(set([e["Tuteur"] for e in st.session_state.etudiants]))
            cojurys_app_list = st.session_state.co_jurys
            with st.spinner("Import disponibilités Excel..."):
                s_msg, e_msg, w_msg = importer_disponibilites_excel(uploaded_file_dispo_ui, st.session_state.horaires_par_jour, tuteurs_app_list, cojurys_app_list)
            for m in s_msg: st.success(m)
            for m in e_msg: st.error(m)
            for m in w_msg: st.warning(m)
            # Pour éviter réimport en boucle si rerun, on peut clear l'uploader
            # st.session_state.excel_dispo_uploader = None # Nécessite de gérer la clé de l'uploader
            # Ou, plus simple, on demande à l'utilisateur de continuer.
        else: st.error("Configurez dates, étudiants, co-jurys avant import dispo.")
    
    st.divider()
    st.subheader("✏️ Saisie manuelle ou vérification")
    
    tous_tuteurs_ui = list(set([e["Tuteur"] for e in st.session_state.etudiants]))
    personnes_ui = sorted(list(set(tous_tuteurs_ui + st.session_state.co_jurys)))
    for p_ui in personnes_ui:
        if p_ui not in st.session_state.disponibilites: st.session_state.disponibilites[p_ui] = {}

    for personne_loop in personnes_ui:
        st.markdown(f"#### 👨‍🏫 Disponibilités de {personne_loop}")
        for jour_loop, creneaux_loop_list in st.session_state.horaires_par_jour.items():
            if not creneaux_loop_list: continue # Pas de créneaux, pas de checkboxes
            
            st.markdown(f"**{jour_loop}**")
            creneaux_jour_keys_loop = [f"{jour_loop} | {c_loop}" for c_loop in creneaux_loop_list]
            dispo_journee_personne_bools = [st.session_state.disponibilites[personne_loop].get(k_loop, False) for k_loop in creneaux_jour_keys_loop]
            all_previously_selected_day = all(dispo_journee_personne_bools)

            all_selected_checkbox_key = f"all_selected_{personne_loop}_{jour_loop.replace('/', '_').replace(' ', '_')}"
            all_selected_val = st.checkbox("Disponible toute la journée", value=all_previously_selected_day, key=all_selected_checkbox_key)

            cols_loop = st.columns(min(len(creneaux_loop_list), 4))
            for i_loop, creneau_val_loop in enumerate(creneaux_loop_list):
                with cols_loop[i_loop % 4]:
                    key_dispo_loop = f"{jour_loop} | {creneau_val_loop}"
                    current_cb_val = True if all_selected_val else st.session_state.disponibilites[personne_loop].get(key_dispo_loop, False)
                    individual_cb_key = f"cb_{personne_loop}_{jour_loop.replace('/', '_').replace(' ', '_')}_{i_loop}"
                    
                    # Si "Toute la journée" est cochée, on force la valeur et on met à jour le state
                    if all_selected_val :
                         st.session_state.disponibilites[personne_loop][key_dispo_loop] = True

                    checked_val = st.checkbox(
                        creneau_val_loop,
                        value=st.session_state.disponibilites[personne_loop].get(key_dispo_loop, False), # Lire toujours du state
                        key=individual_cb_key,
                        disabled=all_selected_val # Désactivé si "toute la journée" est coché
                    )
                    # Mettre à jour le state seulement si l'interaction vient de cette checkbox
                    # (pas de la checkbox "toute la journée")
                    if not all_selected_val:
                         st.session_state.disponibilites[personne_loop][key_dispo_loop] = checked_val
            st.markdown("---") # Séparateur plus léger entre les jours pour une personne
        st.divider() # Gros séparateur entre les personnes

    if st.button("Suivant > Générer Planning", type="primary", key="dispo_suivant"):
        st.session_state.etape = "generation"; st.rerun()

elif st.session_state.etape == "generation":
    afficher_navigation()
    st.header(etapes_labels["generation"])
    
    utiliser_ag_ui_val = st.checkbox(
        "Utiliser l'algorithme génétique (recommandé)", value=True, 
        help="L'AG est généralement plus performant. Il sera aussi utilisé si l'algo classique < 80% de réussite.",
        key="utiliser_ag_checkbox"
    )
    params_ag_config_ui = {}
    if utiliser_ag_ui_val:
        with st.expander("⚙️ Paramètres de l'algorithme génétique", expanded=False):
            taille_pop_ui = st.slider("Taille population AG", 20, 200, 80, key="ag_pop_slider")
            nb_gen_ui = st.slider("Nb générations AG", 50, 1000, 300, key="ag_gen_slider")
            taux_mut_ui = st.slider("Taux mutation AG", 0.05, 0.30, 0.12, step=0.01, key="ag_mut_slider")
            params_ag_config_ui = {
                'taille_population': taille_pop_ui, 'nb_generations': nb_gen_ui, 'taux_mutation': taux_mut_ui
            }

    if st.button("🚀 Lancer l'optimisation", type="primary", key="lancer_opti_btn"):
        if not st.session_state.etudiants:
            st.error("Aucun étudiant à planifier. Veuillez en ajouter à l'étape 1.")
        elif not st.session_state.dates_soutenance:
            st.error("Aucune date de soutenance définie. Veuillez les configurer à l'étape 5.")
        elif not st.session_state.horaires_par_jour: # Vérifier si les créneaux ont été générés
            st.error("Les créneaux n'ont pas été générés (Étape 6). Veuillez vérifier les dates et la durée.")

        else:
            with st.spinner("Optimisation en cours... Cela peut prendre quelques minutes..."):
                optimiseur_instance = PlanificationOptimiseeV2(
                    st.session_state.etudiants, st.session_state.co_jurys, st.session_state.dates_soutenance,
                    st.session_state.disponibilites, st.session_state.nb_salles, st.session_state.duree_soutenance
                )
                planning_gen, non_planifies_gen, stats_gen = optimiseur_instance.optimiser_avec_genetique(
                    utiliser_genetique_ui=utiliser_ag_ui_val, **params_ag_config_ui
                )
                st.session_state.planning_final = planning_gen

            if stats_gen:
                st.subheader("🧬 Statistiques de l'Algorithme Génétique")
                col_sg1, col_sg2, col_sg3 = st.columns(3)
                col_sg1.metric("Générations", stats_gen.get('generations', 'N/A'))
                col_sg2.metric("Fitness Finale", f"{stats_gen.get('fitness_finale', 0.0):.1f}")
                col_sg3.metric("Conflits (AG)", stats_gen.get('conflits', 'N/A'))
                if 'amelioration_valeur' in stats_gen and stats_gen['amelioration_valeur'] > 0:
                    st.success(f"AG a ajouté {stats_gen['amelioration_valeur']} soutenances de plus que l'algo classique.")
                
                if stats_gen.get('historique'):
                    import plotly.graph_objects as go
                    df_hist_ag = pd.DataFrame(stats_gen['historique'])
                    if not df_hist_ag.empty:
                        fig_evol_ag = go.Figure()
                        fig_evol_ag.add_trace(go.Scatter(x=df_hist_ag['generation'], y=df_hist_ag['fitness_max'], mode='lines', name='Fitness Max'))
                        fig_evol_ag.add_trace(go.Scatter(x=df_hist_ag['generation'], y=df_hist_ag['soutenances_max'], mode='lines', name='Soutenances Max', yaxis='y2'))
                        fig_evol_ag.update_layout(title="Évolution de l'AG", xaxis_title="Génération", yaxis_title="Fitness", yaxis2=dict(title="Soutenances", overlaying='y', side='right'), height=400)
                        st.plotly_chart(fig_evol_ag, use_container_width=True)

            if st.session_state.planning_final:
                conflits_finaux = optimiseur_instance.verifier_conflits(st.session_state.planning_final)
                if conflits_finaux:
                    st.error("⚠️ Conflits détectés dans le planning final :"); [st.write(f"- {c}") for c in conflits_finaux]
                else: st.success("✅ Aucun conflit détecté dans le planning final.")

                st.success(f"Planning généré! {len(st.session_state.planning_final)} soutenances planifiées.")
                if non_planifies_gen > 0: st.warning(f"⚠️ {non_planifies_gen} étudiant(s) non planifiés.")

                df_planning_final = pd.DataFrame(st.session_state.planning_final)
                st.subheader("📋 Planning détaillé")
                st.dataframe(df_planning_final.drop(['Début', 'Fin'], axis=1, errors='ignore'), use_container_width=True)

                if not df_planning_final.empty:
                    st.subheader("📊 Visualisation Gantt")
                    df_planning_final["Task"] = df_planning_final["Étudiant"] + " (" + df_planning_final["Salle"] + ")"
                    fig_gantt = px.timeline(
                        df_planning_final, x_start="Début", x_end="Fin", y="Tuteur", color="Task",
                        title="Planning par tuteur", hover_data=["Étudiant", "Co-jury", "Salle", "Pays"]
                    )
                    fig_gantt.update_yaxes(autorange="reversed"); fig_gantt.update_layout(height=max(600, len(df_planning_final['Tuteur'].unique())*50 ))
                    st.plotly_chart(fig_gantt, use_container_width=True)
                    
                    st.subheader("📥 Exportation")
                    csv_export = df_planning_final.to_csv(index=False).encode('utf-8')
                    st.download_button("Télécharger CSV", csv_export, "planning_soutenances.csv", "text/csv", key="dl_csv")
                    
                    output_excel = BytesIO()
                    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer_excel:
                        df_planning_final.to_excel(writer_excel, index=False, sheet_name='Planning')
                    st.download_button("Télécharger Excel", output_excel.getvalue(), "planning_soutenances.xlsx", 
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_excel")

            else: st.error("❌ Aucune soutenance planifiée. Vérifiez disponibilités et contraintes.")

# Sidebar Résumé
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Résumé Actuel")
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
