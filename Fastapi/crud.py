from sqlalchemy.orm import Session
from models import Vehicle_Base
from schemas import Vehicle


def get_vehicle(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Vehicle_Base).offset(skip).limit(limit).all()


def get_vehicle_by_id(db: Session, vehicle_id: int):
    return db.query(Vehicle_Base).filter(Vehicle_Base.id == vehicle_id).first()


# def create_vehicle(db: Session, vehicle: Vehicle):
#     with open("a.txt", "r+") as file:

#         for line in file:
#             _vehicle =Vehicle_Base(title=str(line))
#             db.add(_vehicle)
#         db.commit()
#         db.refresh(_vehicle)
#         file.truncate()
#     return _vehicle

