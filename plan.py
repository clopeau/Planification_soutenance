elif st.session_state.etape == 5:
    st.title("5. Génération & Bilan")
    
    # --- CONFIGURATION DU LANCEMENT ---
    with st.expander("Paramètres avancés", expanded=False):
        c1, c2 = st.columns(2)
        n_iter = c1.slider("Itérations", 10, 200, 50)
        w_rand = c2.slider("Exploration (Aléatoire)", 0, 500, 100)
        c3, c4 = st.columns(2)
        w_cont = c3.slider("Poids Contiguïté (Temps)", 0, 5000, 2000)
        w_bal = c4.slider("Poids Équilibre (Charge)", 0, 2000, 500)
        w_room = st.slider("Poids Stabilité Salle", 0, 5000, 3000)
    
    st.info("ℹ️ Règle active : Un tuteur doit être co-jury autant de fois qu'il est tuteur (Bilan = 0).")

    # --- BOUTON LANCER ---
    if st.button("Lancer la planification", type="primary"):
        params = {
            "n_iterations": n_iter, "w_random": w_rand, 
            "w_contiguity": w_cont, "w_balance": w_bal, 
            "w_day": 100, "w_room": w_room
        }
        # Instanciation et lancement du moteur
        eng = SchedulerEngine(
            st.session_state.etudiants, st.session_state.dates, st.session_state.nb_salles, st.session_state.duree, 
            st.session_state.disponibilites, st.session_state.filieres, st.session_state.co_jurys, params
        )
        plan, fail, charges = eng.run_optimization()
        
        # Sauvegarde des résultats
        st.session_state.planning = plan
        st.session_state.failed = fail
        st.session_state.stats_charges = charges
        
    # --- AFFICHAGE DES RÉSULTATS ---
    if st.session_state.planning:
        st.divider()
        c_stat1, c_stat2 = st.columns(2)
        c_stat1.success(f"✅ Soutenances planifiées : {len(st.session_state.planning)}")
        if st.session_state.failed:
            c_stat2.error(f"❌ Non placés : {len(st.session_state.failed)}")
        else:
            c_stat2.success("Tous les étudiants sont placés !")

        # --- TABLEAU DE BILAN DEMANDÉ ---
        if 'stats_charges' in st.session_state:
            st.subheader("📊 Tableau de Contrôle (Bilan Tuteur / Co-jury)")
            
            charges = st.session_state.stats_charges
            data_summary = []
            
            # Récupérer la liste complète des enseignants
            all_profs = set(charges.keys())
            for e in st.session_state.etudiants: 
                if e['Tuteur']: all_profs.add(e['Tuteur'])
            
            for p in sorted(list(all_profs)):
                if not p: continue
                # Récupération des compteurs
                c_t = charges[p]['tuteur']  # Nombre d'étudiants suivis (Jury)
                c_c = charges[p]['cojury']  # Nombre de participations (Co-jury)
                
                # Calcul du bilan : Cojury - Tuteur
                # Si Tuteur = 5 et Cojury = 4 -> 4 - 5 = -1 (Manque 1 soutenance)
                bilan = c_c - c_t 
                
                # On n'affiche que ceux qui ont une activité
                if c_t > 0 or c_c > 0:
                    data_summary.append({
                        "Tuteur": p,
                        "Jury (Tuteur)": c_t,
                        "Co-jury": c_c,
                        "Bilan": bilan
                    })
            
            df_summary = pd.DataFrame(data_summary)
            
            # Fonction de style pour la colonne Bilan
            def color_bilan(val):
                if val == 0:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold;' # Vert
                elif val < 0:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold;' # Rouge
                return ''

            # Affichage du tableau stylisé
            st.dataframe(
                df_summary.style.map(color_bilan, subset=['Bilan'])
                                .format({"Bilan": "{:+d}"}), # Affiche le signe (+0, -1)
                use_container_width=True,
                hide_index=True
            )
            
            # Petit message explicatif sous le tableau
            if not df_summary.empty and (df_summary['Bilan'] < 0).any():
                st.warning("⚠️ Les lignes en rouge indiquent un enseignant qui n'a pas atteint son quota de co-jury (Bilan négatif).")
            elif not df_summary.empty:
                st.success("✅ Parité parfaite respectée pour tous les enseignants.")

        # --- EXPORT EXCEL ---
        st.divider()
        excel_data = generate_excel_planning(st.session_state.planning, st.session_state.nb_salles)
        st.download_button("📥 Télécharger le Planning Complet (.xlsx)", excel_data, "Planning_Soutenances.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

        # --- VISUALISATION DÉTAILLÉE (TABS) ---
        tab1, tab2, tab3 = st.tabs(["📋 Liste Détaillée", "📅 Diagramme de Gantt", "❌ Échecs éventuels"])
        
        with tab1:
            st.dataframe(pd.DataFrame(st.session_state.planning))
            
        with tab2:
            if not pd.DataFrame(st.session_state.planning).empty:
                df_g = []
                for x in st.session_state.planning:
                    # Entrée pour le Tuteur
                    df_g.append({
                        "Enseignant": x['Tuteur'], "Role": "Tuteur", "Etudiant": x['Étudiant'], 
                        "Jour": x['Jour'], "Start": datetime(2000,1,1,x['Début'].hour, x['Début'].minute), 
                        "End": datetime(2000,1,1,x['Fin'].hour, x['Fin'].minute)
                    })
                    # Entrée pour le Co-jury
                    df_g.append({
                        "Enseignant": x['Co-jury'], "Role": "Co-jury", "Etudiant": x['Étudiant'], 
                        "Jour": x['Jour'], "Start": datetime(2000,1,1,x['Début'].hour, x['Début'].minute), 
                        "End": datetime(2000,1,1,x['Fin'].hour, x['Fin'].minute)
                    })
                
                df_viz = pd.DataFrame(df_g).sort_values("Enseignant")
                fig = px.timeline(df_viz, x_start="Start", x_end="End", y="Enseignant", color="Role", 
                                  facet_col="Jour", text="Etudiant", height=max(400, len(all_profs)*30),
                                  color_discrete_map={"Tuteur": "#2E86C1", "Co-jury": "#28B463"})
                fig.update_xaxes(tickformat="%H:%M")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                
        with tab3:
            if st.session_state.failed:
                st.error("Les étudiants suivants n'ont pas pu être placés (manque de créneaux ou de co-jurys disponibles) :")
                st.dataframe(pd.DataFrame(st.session_state.failed))
            else:
                st.info("Aucun échec.")
