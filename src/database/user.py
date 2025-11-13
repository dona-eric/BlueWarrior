from sqlite3 import Connection as SQLite3Connection
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy import event
from pydantic import EmailStr
from sqlmodel import SQLModel, create_engine as sqlmodel_create_engine, Session, Field, Relationship
import os



class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str
    surname: str
    email: EmailStr
    hashed_password: str
    patients: list["Patient"] = Relationship(back_populates='user')
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False
    
class Patient(SQLModel, table= True):
    id : int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    age: int
    family_history : bool
    group_sanguin : str
    user: User= Relationship(back_populates='patients')    
    
