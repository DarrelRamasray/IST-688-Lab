#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Labs

import streamlit as st

lab1_page = st.Page("Lab1.py", title="Lab 1", icon=":material/description:")
lab2_page = st.Page("Lab2.py", title="Lab 2", icon=":material/description:", default=True)  #Default page

pg = st.navigation([lab2_page, lab1_page])  #Lab2 listed first so it opens by default
st.set_page_config(page_title="Lab Application", page_icon=":material/edit:")
pg.run()