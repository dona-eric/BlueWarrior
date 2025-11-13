from src.database.user import User
from sqlmodel import SQLModel, create_engine, Session
from src.database.connection import engine
import os


DATABASE_URL = "sqlite:///./test.db"


def create_tables_db():
    database_prosta = SQLModel.metadata.create_all(bind=engine)
    return database_prosta