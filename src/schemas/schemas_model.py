from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, List, Union

class UserCreate(BaseModel):
    name: str
    surname: str
    email: EmailStr
    password: str
    tags: list[str] = []
    
class UserRead(BaseModel):
    id: int
    name: str
    surname: str
    email: EmailStr
    
    model_config = ConfigDict(from_attributes=True)
    
    """
    schemas du patient pour les antécedents medicaux
    """
class Patient (BaseModel):
    name :str
    surname : str
    age :  int 
    groupe_sanguin: str
    chronic_disease : Optional[List] = []
    family_history: bool
    
    
class RiskEvaluation(BaseModel):
    risk_score : float
    risk_level: str
    
class Question(BaseModel):
    query: str
    
#class Audio(BaseModel):
    