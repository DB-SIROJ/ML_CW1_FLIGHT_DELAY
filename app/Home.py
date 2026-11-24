"""Entry point: defines the navigation and the shared look. Run with `streamlit run app/Home.py`."""
import streamlit as st

st.set_page_config(page_title="Flight Delay Lab", page_icon="✈️", layout="wide")

from ui import glossary_sidebar, inject_css  # noqa: E402

pages = {
    "Start": [st.Page("views/home.py", title="Overview", icon="🏠", default=True)],
    "Understand the data": [
        st.Page("views/explore.py", title="Explore the data", icon="🔎"),
        st.Page("views/prep.py", title="How the data was prepared", icon="🧹"),
    ],
    "Build & evaluate": [
        st.Page("views/lab.py", title="Model lab", icon="🧪"),
        st.Page("views/results.py", title="Results & model comparison", icon="🏆"),
    ],
    "Use it": [st.Page("views/checker.py", title="Delay checker", icon="🛫")],
    "More": [st.Page("views/about.py", title="About, limitations & ethics", icon="📘")],
}
nav = st.navigation(pages)
inject_css()
glossary_sidebar()
st.sidebar.caption("Data: U.S. Bureau of Transportation Statistics, Jan–Mar 2025 · "
                   "[Source code](https://github.com/DB-SIROJ/ML_CW1_FLIGHT_DELAY)")
nav.run()
