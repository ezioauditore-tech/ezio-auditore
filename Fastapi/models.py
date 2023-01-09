from sqlalchemy import  Column, Integer, String
from config import Base

class Vehicle_Base(Base):
    __tablename__ ="kgx-vehicle"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    num_plate=Column(String)