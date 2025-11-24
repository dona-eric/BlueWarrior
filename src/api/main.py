from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from src.database.connection import SessionDeep
from src.database.init_db import create_tables_db
from src.database.user import User, Patient
from src.schemas.schemas_model import UserCreate, UserRead, Patient, Question
from src.rag_engine.pipeline_final import get_rag_chain
from src.utils.logs import setup_logging
from src.utils.load_dataset import load_data, preprocess_data_2050
import os, logging
# import matplotlib.pyplot as plt
import plotly.express as px, plotly.io as pio
import plotly.graph_objects as go

#==================INITIALISATION DES LOGGINGS ===========
setup_logging()
logger= logging.getLogger("==============BLUE WARRIORS============")

# data_2050, data_2022_mortality = load_data()
df = preprocess_data_2050(load_data()[1])
# ====================== FONCTION D'INITIALISATION DE LA CHAINE RAG=============
RAG_CHAIN, RAG_ERROR = None, None

def initial_rag_chain():
    """
    j'initialise la chaine rag en utilisant
    """
    global RAG_CHAIN, RAG_ERROR
    try:
        RAG_CHAIN = get_rag_chain()
        logger.info("RAG chain initialized successfully.")
    except Exception as e:
        RAG_ERROR = str(e)
        logger.error(f"Error initializing RAG chain: {e}")
    

#======================== CREATION DE L'APPLICATION PRINCIPALE =================
app = FastAPI(
    title="BlueWarriors",
    description="API pour la prévention et l’accompagnement des hommes contre le cancer de la prostate",
    version="1.0.1",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/openapi.json",
    include_in_schema=True,
    contact={
        "Email":"donaerickoulodji@gmail.com",
        "Phone Number":"+2290151344289",
        "Linkedin":"www.linkedin.com/in/dona-erick"
        },
)


# Ajouter le CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
create_tables_db()
#================ FONCTION POUR L'APPEL DE RAG ENGINE ===========

@app.on_event(event_type="startup")
def startup_event():
    logger.info(msg="initilisation de rag")
    initial_rag_chain()
    
    
# =================== UTILISATION DE ROUTEUR ==============
router = APIRouter()


#======LES ROUTES ==========
@router.get("/kaithhealthcheck", tags=["Health"])
async def health_check():
    return {"status": "ok"}


@router.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome on BlueWarriors"}


@router.post("/user/",  response_model=UserRead, tags=["User"])
async def create_user(user: UserCreate,  db: SessionDeep):
    db_user = User(
        name=user.name,
        surname=user.surname,
        email=user.email,
        hashed_password=user.password  # Note: In a real application, hash the password!
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post('/patient/', response_model=Patient, tags=['Patient'])
async def  create_patient(patient: Patient, db: SessionDeep):
    """
    Stocke les informations et/ou antécédents médicaux des patients(users)
    souffrant ou non du cancer de la prostate pour faire une prédiction ML

    Args:
        patient (Patient): une classe de schema qui stocke les infos users
        db (SessionDeep): une session de conexion àla DB

    Returns:
        db_patient: une table de base de données pour enregistrer les infos users
    """
    db_patient = Patient(
        name=patient.name,
        surname = patient.surname,
        age = patient.age,
        groupe_sanguin=patient.groupe_sanguin,
        chronic_disease=patient.chronic_disease,
        family_history= patient.family_history
    )
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@router.get('/users/', response_model=list[UserRead], tags=["User"])
async def get_user(db: SessionDeep):
    users = db.exec(select(User)).all()
    return users
    
    
@router.get("/patients/",tags=['Patients'],
            response_model=list[Patient], 
            response_model_exclude={"groupe_sanguin", "chronic_disease"})

async def list_patients(db: SessionDeep):
    patients = db.exec(select(Patient)).all()
    return patients

@router.get('/patient/{id_patient}',
            tags=['Patient_Id'],
            response_model_exclude = {'groupe_sanguin', "chronic_disease"}
            )
async def get_patient_id(id_patient: str, db: SessionDeep):
    patient_id = db.get(Patient, id_patient)
    if not patient_id:
          raise HTTPException(status_code=404, detail="patient not found")
    else:
        return patient_id
  
    
@router.post("/ask/", tags=["RAG Chatbot"])
def ask_question(question: Question):
    """Endpoint principal du chatbot RAG."""
    
    global RAG_CHAIN, RAG_ERROR
    
    #RAG_CHAIN = initial_rag_chain()
    
    """Je verifie d'abord si la chaine RAG est initialisée correctement"""
    if RAG_CHAIN is None:
        logger.error("RAG chain is not initialized.")
        raise HTTPException(status_code=500, detail="RAG chain is not initialized.")
    
    try:
        response = RAG_CHAIN.invoke(
            {
                "query": question.query
            }
        )
        answer = response["result"]
        sources = [doc.metadata.get("source", "inconnu") for doc in response["source_documents"]]
        
        return {
            "question": question.query,
            "Reponse": answer,
            "Sources": sources
        }

    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")
 
 
 
### ===============================================
# ENDPOINT POUR LES STATISTIQUES ET TABLEAU D BOARD 
# ==================================================
def fig_to_json(fig):
    """Retourne la figure Plotly en JSON (string)"""
    return pio.to_json(fig)

@router.get("/stats/", tags=['Statistiques'])

def get_statistics_cancer():
    
    """
        chargement des données (datasets de mortalkkity et de predictiion2050)
    """
    data_2050, data_2022_mortality = load_data()
    df = preprocess_data_2050(data_2050)

    result = {
        "sample": df.head(10).to_dict(orient="records"),
        "charts": {},
        "updated":"2025"
    }
    
    # ========== VISUALISATION AVEC DATA 2050 PREDICTION ===========

    # total prediction et cases_2022 si disponibles
    if "year" in df.columns:
        # total par année:utilise prediction si présente sinon cases_2022 pour 2022
        df_year = df.copy()
        # si prediction contient des années multiples, on considère (year,prediction)
        if "prediction" in df_year.columns and df_year["prediction"].notnull().any():
            # suppose prediction est numérique — sinon ignore
            yearly = df_year.groupby("year")["prediction"].sum().reset_index()
            yearly = yearly.sort_values("year")
            fig_year = px.line(yearly, x="year", y="prediction", title="Évolution prédite des cas")
        elif "cases_2022" in df_year.columns:
            yearly = df_year.groupby("year")["cases_2022"].sum().reset_index()
            fig_year = px.line(yearly, x="year", y="cases_2022", title="Évolution des cas (base)")
        else:
            fig_year = go.Figure()
            fig_year.update_layout(title="Données annuelles indisponibles")
    else:
        fig_year = go.Figure()
        fig_year.update_layout(title="Données annuelles indisponibles")
    result["charts"]["evolution_annuelle"] = fig_to_json(fig_year)

    #Comparaison Hommes vs Femmes (bar)
    if "sex" in df.columns:
        df_sex = df.groupby("sex").agg({"cases_2022": "sum", "prediction": "sum"}).reset_index()
        # Choisir la colonne disponible
        y_col = "prediction" if "prediction" in df_sex.columns and df_sex["prediction"].notnull().any() else "cases_2022"
        fig_sex = px.bar(df_sex, x="sex", y=y_col, title=f"Comparaison par sexe ({y_col})", text=y_col)
    else:
        fig_sex = go.Figure()
        fig_sex.update_layout(title="Colonne 'sex' manquante")
    result["charts"]["sexe_comparaison"] = fig_to_json(fig_sex)

    # Impact: population vs risque (waterfall)
    if {"change_population", "change_risk"}.issubset(set(df.columns)):
        total_population = df["change_population"].sum(min_count=1) or 0
        total_risk = df["change_risk"].sum(min_count=1) or 0
        net_change = df["change_total"].sum(min_count=1) if "change_total" in df.columns else (total_population + total_risk)
        measures = ["relative", "relative", "total"]
        x = ["Due to population", "Due to risk", "Net change"]
        y = [total_population, total_risk, net_change]
        fig_water = go.Figure(go.Waterfall(
            name="impact",
            orientation="v",
            measure=measures,
            x=x,
            text=[f"{v:.0f}" for v in y],
            y=y
        ))
        fig_water.update_layout(title="Impact : population vs risque (somme)")
    else:
        fig_water = go.Figure()
        fig_water.update_layout(title="Colonnes d'impact manquantes")
    result["charts"]["impact_population_risque"] = fig_to_json(fig_water)

    # Type de cancer (sunburst/treemap)
    if "type" in df.columns:
        df_type = df.copy()
        value_col = "prediction" if "prediction" in df_type.columns and df_type["prediction"].notnull().any() else "cases_2022"
        fig_type = px.treemap(df_type, path=["type"], values=value_col, title="Répartition par type de cancer")
    else:
        fig_type = go.Figure()
        fig_type.update_layout(title="Colonne 'type' manquante")
    result["charts"]["type_repartition"] = fig_to_json(fig_type)

    # Répartition géographique (si population contient des noms/pays)
    if "population" in df.columns:
        # on tente une choropleth si population contient des noms de pays
        try:
            df_geo = df.groupby("population").agg({value_col: "sum"}).reset_index().rename(columns={"population": "country", value_col: "value"})
            fig_geo = px.choropleth(df_geo, locations="country", locationmode="country names", color="value", title="Répartition géographique (par population)")
        except Exception:
            fig_geo = go.Figure()
            fig_geo.update_layout(title="Impossible de tracer la choropleth (population non standard)")
    else:
        fig_geo = go.Figure()
        fig_geo.update_layout(title="Colonne 'population' manquante")
    result["charts"]["repartition_geographique"] = fig_to_json(fig_geo)

    #============== VISUALISATION AVEC DATA MORTALITY ===========
    
    # A = Evolution par année de la mortalité due au cancer de la prostate
    data_year = data_2022_mortality.groupby('Year')['Number'].sum().reset_index()
    fig_year = px.line(data_year, x='Year', 
                       y='Number', title='Evolution de la mortalité due au cancer de la prostate par année')
    result["charts"]["evolution_annuelle"] = fig_to_json(fig_year)
    
    fig_region = px.bar(
        data_2022_mortality.groupby("Region_Name")["Number"].sum().reset_index(),
        x="Region_Name",
        y="Number",
        title="Nombre de Décès par région"
    )
    result["charts"]["deces_par_region"] = fig_to_json(fig_region)
    # === Visualisation 3 : death rate par pays ===
    fig_rate = px.choropleth(
        data_2022_mortality,
        locations="Country_Name",
        locationmode="country names",
        color="Death_rate_per_100000",
        title="Taux de mortalité par pays (pour 100,000)"
    )
    result["charts"]["taux_mortalite"] = fig_to_json(fig_rate)
    
    
    return result
    
app.include_router(router, prefix="/v1")
