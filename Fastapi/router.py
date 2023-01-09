from fastapi import APIRouter, HTTPException, Path
from fastapi import Depends
from config import SessionLocal
from sqlalchemy.orm import Session
from schemas import RequestVehicle, Vehicle, Request, Response, RequestVehicle

import crud

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/create")
async def create_service(request: RequestVehicle, db: Session = Depends(get_db)):
    crud.create_vehicle(db, vehicle=request.parameter)
    return Response(status="Ok",
                    code="200",
                    message="Created successfully").dict(exclude_none=True)


@router.get("/")
async def get_vehicle(db: Session = Depends(get_db)):
    _vehicle = crud.get_vehicle(db)
    return Response(status="Ok", code="200", message="Success fetch all data", result=_vehicle)


