from typing import List, Optional, Generic, TypeVar
from pydantic import BaseModel , Field
from pydantic.generics import GenericModel
import paho.mqtt.client as mqtt



T = TypeVar('T')

class Vehicle(BaseModel):
    id: Optional[int] = None
    title: Optional[str]=None
    num_plate:Optional[str]=None
    class Config:
        orm_mode = True

class Request(GenericModel, Generic[T]):
    parameter: Optional[T] = Field(...)


class RequestVehicle(BaseModel):
    parameter: Vehicle = Field(...)


class Response(GenericModel, Generic[T]):
    code: str
    status: str
    message: str
    result: Optional[T]