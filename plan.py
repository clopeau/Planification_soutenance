if st.button("Lancer", type="primary"):
    params = {"n_iterations": n_iter, "w_random": w_rand, "w_contiguity": w_cont, "w_balance": w_bal, "w_day": 100}
    eng = SchedulerEngine(st.session_state.etudiants, st.session_state.dates, st.session_state.nb_salles, st.session_state.duree, st.session_state.disponibilites, st.session_state.co_jurys, params)
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
            data.append({"Enseignant": p, "Tuteur": c_t, "Co-Jury": c_c, "Delta": c_t-c_c})
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
