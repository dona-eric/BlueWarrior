from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from src.database.connection import get_session, engine, SessionDeep
from src.database.init_db import create_tables_db
from src.database.user import User, Patient
from src.schemas.schemas_model import UserCreate, UserRead, Patient, Question
from src.rag_engine.pipeline_final import get_rag_chain
import os, logging, threading

#==================INITIALISATION DES LOGGINGS ===========

logging.basicConfig(level=1, filename="app.log", format="")

# ====================== FONCTION D'INITIALISATION DE LA CHAINE RAG=============
RAG_CHAIN=None
RAG_ERROR = None

def initial_rag_chain():
    """
    j'initialise la chaine rag en utilisant
    """
    global RAG_CHAIN, RAG_ERROR
    try:
        RAG_CHAIN = get_rag_chain()
        logging.info("RAG chain initialized successfully.")
    except Exception as e:
        RAG_ERROR = e
        logging.error(f"Error initializing RAG chain: {e}")
    

#======================== CREATION DE L'APPLICATION PRINCIPALE =================
app = FastAPI(
    title="BlueWarriors",
    description="API pour la prévention et l’accompagnement des hommes contre le cancer de la prostate",
    version="1.0.1",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/openapi.json",
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
initial_rag_chain()
# =================== UTILISATION DE ROUTEUR ==============
router = APIRouter()


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
    patient_id = db.get(Patient, patient_id)
    if not patient_id:
          raise HTTPException(status_code=404, detail="patient not found")
    else:
        return patient_id
  
        
        

@router.post("/ask/", tags=["RAG Chatbot"])
def ask_question(question: Question):
    """Endpoint principal du chatbot RAG."""
    
    global RAG_CHAIN, RAG_ERROR
    
    """Je verifie d'abord si la chaine RAG est initialisée correctement"""
    if RAG_CHAIN is None:
        logging.error("RAG chain is not initialized.")
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
        logging.error(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")
 
app.include_router(router, prefix="/api/v1")
