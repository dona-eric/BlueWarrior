import streamlit as st, requests,time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

st.write()
st.set_page_config(page_title='BlueWarriors',
                   page_icon="🎗️",
                   initial_sidebar_state="expanded",
                   layout="wide")
# title 
st.title("_BlueWarriors_: :blue[Notre Santé, Notre Vie] :sunflower:")

st.markdown("*Warriors*: **:blue[Ensemble, Soyons les héros de notre santé]**")
# affichage image
st.sidebar.image("/home/donerick/ProstaCareAI/backend/prostate.jpg", width="stretch", caption="Prévention du Cancer de la Prostate")
# custom css
st.markdown("""
<style> 
    html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif, Roboto;
    }   
    .main-header {
        font-size: 3rem;
        color: #FF1493;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FF69B4 0%, #FF1493 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #fff3f8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #FF69B4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)




# ========================
# CONFIG
# ========================
# URL API

API_BASE_URL = "http://127.0.0.1:8000/api/v1" 

def is_small_talk(query: str) -> bool:
    """Détecte les questions de conversation simple (salutations, comment ça va)."""
    query = query.lower().strip()
    greetings = ["bonjour", "salut", "hello", "hi", "coucou"]
    small_talk = ["comment tu vas", "ça va", "quoi de neuf", "tout va bien", "comment vas-tu"]
    
    # Vérifie si c'est une simple salutation (max 10 mots pour éviter les fausses détections)
    if any(g in query for g in greetings) and len(query.split()) < 4:
        return True
    if any(s in query for s in small_talk):
        return True
    
    return False


def render_chatbot_page():
    """Rend l'interface du Chatbot RAG avec gestion du Small Talk."""
    
    st.header(":blue[_Warrior_]", divider=False)
    
    # Définition du message de CTA
    cta_response = (
        "Je vais très bien, merci de demander ! En tant qu'assistant spécialisé BlueWarriors, "
        "je suis prêt à vous aider avec toute question concernant la **prévention du cancer de la prostate**."
        "\n\n👉 **Quel est le sujet qui vous intéresse le plus aujourd'hui (dépistage, risques, traitements) ?**"
    )

    # Initialisation de l'Historique de Chat (inchangé)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Salutation initiale du Bot (peut rester pour commencer la conversation)
        initial_greeting = "Bonjour ! Je suis l'assistant BlueWarriors, spécialisé dans la prévention du cancer de la prostate. N'hésitez pas à me poser vos questions. Prenez soin de vous !"
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting, "sources": []})


    # Affichage de l'Historique (inchangé)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Affichage discret des sources cliquables UNIQUEMENT si la liste n'est pas vide
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- [{src}]({src})")


    # Champ de saisie utilisateur
    if prompt := st.chat_input("Votre question:"):
        
        # 1. Enregistrement du prompt utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        
        # 2. LOGIQUE DE SMALL TALK vs. RAG
        if is_small_talk(prompt):
            answer = cta_response
            sources = []
            role = "assistant"
            
        else:
            # Procéder à l'appel RAG
            with st.spinner("Réflexion..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/ask/", json={"query": prompt})

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("Reponse", "Je suis désolé, je n'ai pas pu trouver de réponse précise. Pouvez-vous reformuler votre question ?")
                        sources = data.get("Sources", [])
                        role = "assistant"
                    else:
                        error_detail = response.json().get('detail', response.text)
                        answer = f"**Erreur API:** {response.status_code}. Détails: {error_detail}. Le service RAG est peut-être indisponible."
                        sources = []
                        role = "assistant"
                
                except Exception as e:
                    answer = f"**Erreur de connexion:** Impossible de contacter l'API. Détails: {e}"
                    sources = []
                    role = "assistant"


        # 3. Affichage et Enregistrement Final
        with st.chat_message(role):
            st.markdown(answer)
            
            # Affichage discret des sources UNIQUEMENT si la liste `sources` n'est PAS vide
            if sources:
                with st.expander("sources"):
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")
        
        # Enregistrement dans l'historique de session
        st.session_state.messages.append({"role": role, "content": answer, "sources": sources})





# --- 1. FONCTIONS DE CHARGEMENT DES DONNÉES (CSV) ---

@st.cache_data
def load_prostate_data():
    """
    Charge, nettoie et prépare les 4 fichiers CSV pour le tableau de bord.
    """
    try:
        # 1. Incidence 2022 (Pour les KPIs et la carte 2022)
        # ASSOMPTION: Colonnes 'Country', 'Cancer_Label', 'Incidence (Abs)', 'Incidence Rate (per 100,000)'
        df_inc_2022 = pd.read_csv('/home/donerick/ProstaCareAI/backend/src/data/dataset-inc-both-sexes-in-2022-prostate.csv')
        if 'Cancer code' in df_inc_2022.columns:
            df_inc_2022 = df_inc_2022[df_inc_2022['Cancer code'] == 27]
        
        # 2. Prédiction Incidence 2050
        # ASSOMPTION: Colonnes 'Country', 'Cancer', 'Value'
        df_pred_inc_2050 = pd.read_csv('/home/donerick/ProstaCareAI/backend/src/data/dataset-absolute-numbers-inc-both-sexes-in-2050-prostate-testis.csv')
        if 'Cancer label' in df_pred_inc_2050.columns:
            df_pred_inc_2050 = df_pred_inc_2050[df_pred_inc_2050['Cancer label'] == "Prostate"]
        df_pred_inc_2050 = df_pred_inc_2050.rename(columns={'Value': 'Predicted_Incidence_2050'})
        
        # 3. Prédiction Tendance Mortalité (2022-2050)
        # ASSOMPTION: Colonnes 'Country', 'Cancer', 'Year', 'Value'
        df_pred_mort_trend = pd.read_csv('/home/donerick/ProstaCareAI/backend/src/data/dataset_number_deaths_2022-2050.csv')
        if 'Cancer label' in df_pred_mort_trend.columns:
             df_pred_mort_trend = df_pred_mort_trend[df_pred_mort_trend['Cancer label'] == 'Penis; Prostate; Testis']
        df_pred_mort_trend = df_pred_mort_trend.rename(columns={'Value': 'Predicted_Deaths'})

        # 4. Données Historiques Mortalité (WHO)
        # ASSOMPTION: 6 lignes d'en-tête à ignorer
        df_hist_mort = pd.read_csv('/home/donerick/ProstaCareAI/backend/src/data/WHOMortalityDatabase_Trends_years_many_countries_by_age_sex-Prostate cancer_14th novembre 2025 01_57.csv', skiprows=6)
        # ASSOMPTION: Colonnes 'Country Name', 'Year', 'Number of deaths'
        df_hist_mort = df_hist_mort.groupby(['Country Name', 'Year'])['Death rate per 100 000 population'].sum().reset_index()
        df_hist_mort = df_hist_mort.rename(columns={'Country Name': 'Country', 'Number of deaths': 'Historical_Deaths'})
        
        return df_inc_2022, df_pred_inc_2050, df_pred_mort_trend, df_hist_mort
    
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement : Fichier introuvable. Assurez-vous que les 4 fichiers CSV sont dans le même répertoire que dashboard.py. Détails : {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement ou du nettoyage des données : {e}. Vérifiez les noms de colonnes dans les CSV.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    
def render_prostate_cancer_dashboard(df_inc_2022, df_pred_inc_2050, df_pred_mort_trend, df_hist_mort):
    """Rend le tableau de bord dynamique basé sur les CSV."""
    
    st.header("🌍 Tableau de Bord Mondial : Cancer de la Prostate")
    
    if df_inc_2022.empty or df_pred_inc_2050.empty or df_pred_mort_trend.empty:
        st.error("Les données du tableau de bord n'ont pas pu être chargées. Vérifiez que les fichiers CSV sont présents et que les noms de colonnes sont corrects.")
        return

    # --- Filtre Pays (avec Bénin par défaut) ---
    all_countries = sorted(df_inc_2022['Country'].unique().tolist())
    countries_with_global = ['Global'] + all_countries
    
    default_index = 0
    if 'Benin' in countries_with_global:
        default_index = countries_with_global.index('Benin')
    
    selected_country = st.sidebar.selectbox(
        "Sélectionnez un Pays",
        countries_with_global,
        index=default_index,
        help="Choisissez 'Global' pour voir la carte et les comparaisons mondiales."
    )

    # Création des onglets
    tab1, tab2 = st.tabs(["📊 Statistiques Actuelles (2022)", "📈 Tendances et Prédictions (2050)"])

    # --- Onglet 1: Statistiques 2022 ---
    with tab1:
        st.subheader(f"Incidence et Mortalité en 2022 pour : **{selected_country}**")
        
        data_2022_country = df_inc_2022
        if selected_country != 'Global':
            data_2022_country = df_inc_2022[df_inc_2022['Country'] == selected_country]

        # --- KPIs 2022 ---
        try:
            kpi_inc_count = data_2022_country['Incidence (Abs)'].sum()
            kpi_inc_rate = data_2022_country['Incidence Rate (per 100,000)'].mean()
            
            col1, col2 = st.columns(2)
            col1.metric("Nouveaux Cas (2022)", f"{kpi_inc_count:,.0f}")
            col2.metric("Taux d'Incidence (pour 100k)", f"{kpi_inc_rate:,.2f}")
        except KeyError:
            st.warning("KPIs 2022 non disponibles. Vérifiez les noms de colonnes 'Incidence (Abs)' et 'Incidence Rate (per 100,000)' dans `dataset-inc-both-sexes-in-2022-prostate.csv`.")
        
        st.markdown("---")

        if selected_country == 'Global':
            # --- Carte Mondiale (Choropleth) ---
            st.subheader("Carte Mondiale du Taux d'Incidence (pour 100,000 habitants)")
            try:
                fig_map = px.choropleth(
                    df_inc_2022,
                    locations="Country",
                    locationmode="country names",
                    color="Incidence Rate (per 100,000)",
                    hover_name="Country",
                    hover_data={"Incidence (Abs)": ':.0f', "Incidence Rate (per 100,000)": ':.2f'},
                    color_continuous_scale=px.colors.sequential.Reds,
                    title="Taux d'Incidence du Cancer de la Prostate (2022)"
                )
                fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors de la création de la carte : {e}. Vérifiez les noms de colonnes.")

            # --- Bar Chart Top 15 ---
            st.subheader("Top 15 des Pays par Nombre Absolu de Cas (2022)")
            df_top15 = df_inc_2022.nlargest(15, 'Incidence (Abs)').sort_values('Incidence (Abs)', ascending=True)
            fig_bar = px.bar(df_top15, y='Country', x='Incidence (Abs)', title="Top 15 des Pays (Nouveaux Cas 2022)",
                             labels={'Incidence (Abs)': 'Nombre de Cas', 'Country': 'Pays'}, text='Incidence (Abs)')
            fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)


    # --- Onglet 2: Prédictions 2050 ---
    with tab2:
        st.subheader(f"Prédictions pour 2050 : **{selected_country}**")

        # Fusion des données d'incidence 2022 et 2050
        df_compare_inc = pd.merge(
            df_inc_2022[['Country', 'Incidence (Abs)']],
            df_pred_inc_2050[['Country', 'Predicted_Incidence_2050']],
            on='Country',
            how='left'
        )
        
        # Filtrer pour le pays
        if selected_country != 'Global':
            data_pred_country = df_compare_inc[df_compare_inc['Country'] == selected_country]
            data_mort_trend_country = df_pred_mort_trend[df_pred_mort_trend['Country'] == selected_country]
        else:
            data_pred_country = df_compare_inc
            data_mort_trend_country = df_pred_mort_trend

        # --- KPIs 2050 ---
        try:
            kpi_pred_inc = data_pred_country['Predicted_Incidence_2050'].sum()
            kpi_inc_2022 = data_pred_country['Incidence (Abs)'].sum()
            augmentation = ((kpi_pred_inc - kpi_inc_2022) / kpi_inc_2022) * 100 if kpi_inc_2022 > 0 else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Prédiction Nouveaux Cas (2050)", f"{kpi_pred_inc:,.0f}")
            col2.metric("Augmentation estimée vs 2022", f"{augmentation:,.1f} %", delta_color="inverse")
        except KeyError:
            st.warning("KPIs 2050 non disponibles. Vérifiez les colonnes 'Predicted_Incidence_2050' et 'Incidence (Abs)'.")
        
        st.markdown("---")

        # --- Graphe de Tendance Mortalité (2022-2050) ---
        st.subheader(f"Tendance de la Mortalité (2022-2050) pour : {selected_country}")
        
        if selected_country == 'Global':
            plot_data_mort = df_pred_mort_trend.groupby('Year')['Predicted_Deaths'].sum().reset_index()
        else:
            plot_data_mort = data_mort_trend_country
        
        if not plot_data_mort.empty:
            fig_mort_trend = px.line(plot_data_mort, x='Year', y='Predicted_Deaths', 
                                     title=f"Prédiction de la Mortalité jusqu'en 2050 - {selected_country}",
                                     labels={'Predicted_Deaths': 'Nombre de Décès Estimés', 'Year': 'Année'})
            fig_mort_trend.update_traces(mode='lines+markers')
            st.plotly_chart(fig_mort_trend, use_container_width=True)
        else:
            st.info(f"Aucune donnée de tendance de mortalité 2022-2050 disponible pour {selected_country}.")
            
        # --- Graphe Comparaison Incidence 2022 vs 2050 ---
        st.subheader(f"Comparaison Incidence 2022 vs 2050 pour : {selected_country}")
        
        if not data_pred_country.empty and data_pred_country['Predicted_Incidence_2050'].sum() > 0:
            if selected_country == 'Global':
                compare_data = pd.DataFrame({
                    'Année': ['2022', '2050'],
                    'Nouveaux Cas Estimés': [df_compare_inc['Incidence (Abs)'].sum(), df_compare_inc['Predicted_Incidence_2050'].sum()]
                })
            else:
                compare_data = pd.DataFrame({
                    'Année': ['2022', '2050'],
                    'Nouveaux Cas Estimés': [data_pred_country['Incidence (Abs)'].iloc[0], data_pred_country['Predicted_Incidence_2050'].iloc[0]]
                })
            
            fig_compare_bar = px.bar(compare_data, x='Année', y='Nouveaux Cas Estimés', 
                                     title=f"Augmentation des Nouveaux Cas (2022 vs 2050) - {selected_country}",
                                     text='Nouveaux Cas Estimés', color='Année')
            fig_compare_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_compare_bar, use_container_width=True)
        else:
             st.info(f"Aucune donnée de prédiction d'incidence 2050 disponible pour {selected_country}.")    
    
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Choisissez une section",
    ("Chatbot AI", "Tableau de Bord Local", "Statistiques Mondiales")
)

# --- Affichage de la Page Sélectionnée ---
if page_selection == "Chatbot AI":
    render_chatbot_page()
elif page_selection == "Tableau de Bord Local":
    # Charge les 4 CSV
    df_inc_2022, df_pred_inc_2050, df_pred_mort_trend, df_hist_mort = load_prostate_data()
    # Rend le tableau de bord avec ces données
    render_prostate_cancer_dashboard(df_inc_2022, df_pred_inc_2050, df_pred_mort_trend, df_hist_mort)