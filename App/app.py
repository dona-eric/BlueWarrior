import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
#from src.api.main import ask_question


st.set_page_config(page_title='BlueWarriors',
                   page_icon="🎗️",
                   initial_sidebar_state="expanded",
                   layout="wide")
# title 
st.title("BlueWarriors: Une application intelligente pour la prévention du Cancer de la Prostate")
# affichage image
st.sidebar.image("/home/donerick/ProstaCareAI/cancer-prostate.jpeg", use_container_width=True, width=175, caption="Prévention du Cancer de la Prostate")