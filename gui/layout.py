import streamlit as st
from gui.tabs import data_cleaning, annotation, keywords, knn, clustering, bayes, comparison
from core.manager import manager  # Importing manager to pass it down to tabs


def run_app():
    st.title("🧩 PJE - Sentiment Analysis & Algorithm Explorer")

    # Create the 7 Tabs
    tab_names = [
        "1. Data & Cleaning",
        "2. Annotation",
        "3. Keywords",
        "4. KNN",
        "5. Clustering",
        "6. Bayes",
        "7. Comparison"
    ]

    tabs = st.tabs(tab_names)

    # Plug the modules into the tabs
    with tabs[0]:
        data_cleaning.render(manager)

    with tabs[1]:
        annotation.render(manager)

    with tabs[2]:
        keywords.render(manager)

    with tabs[3]:
        knn.render(manager)

    with tabs[4]:
        clustering.render(manager)

    with tabs[5]:
        bayes.render(manager)

    with tabs[6]:
        comparison.render(manager)