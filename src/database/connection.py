from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os, urllib, logging, sys, traceback
from sqlmodel import SQLModel, create_engine as sqlmodel_create_engine, Session
from dotenv import load_dotenv
from fastapi import Depends
from typing import Annotated
load_dotenv()

engine = create_engine("sqlite:///./test.db", echo=True)

def get_session():
    with Session(engine) as session:
        yield session
        
        
SessionDeep = Annotated[Session, Depends(get_session)]

