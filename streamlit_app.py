#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#Labs

import streamlit as st

lab1_page = st.Page("Lab1.py", title="Lab 1", icon=":material/description:")
lab2_page = st.Page("Lab2.py", title="Lab 2", icon=":material/description:")
lab3_page = st.Page("Lab3.py", title="Lab 3", icon=":material/description:", default=True)  #Default page

pg = st.navigation([lab3_page,lab2_page, lab1_page])  #Lab3 listed first so it appears at the top of the sidebar
st.set_page_config(page_title="Lab Application", page_icon=":material/edit:")
pg.run()