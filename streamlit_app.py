import streamlit as st

pages = {
    "Main": [
        st.Page("app/home.py", title="Home"),
        st.Page("app/metrics.py", title="Model Perfomance"),
        st.Page("app/visualization.py", title="EDA")
    ],
    "Resources": [
        st.Page("app/documentation.py", title="Documentation")
    ],
}

pg = st.navigation(pages)
pg.run()