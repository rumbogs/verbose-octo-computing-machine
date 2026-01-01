import streamlit as st

pg = st.navigation(
    [
        st.Page("data_cleanup.py", title="1: Data pipeline"),
        st.Page("architecture.py", title="2: Architecture"),
        st.Page("inference.py", title="3: Inference"),
    ]
)
pg.run()
