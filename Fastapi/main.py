from email import message
from http import client
from json import load
from fastapi import FastAPI,Depends
import models
from router import router
from config import engine
from fastapi_mqtt import FastMQTT, MQTTConfig
import schemas
# from sqlalchemy.orm import Sessions
# from fastapi import Depends
from config import SessionLocal
models.Base.metadata.create_all(bind=engine)

app = FastAPI()
mqtt_config = MQTTConfig()

mqtt_config = MQTTConfig(host = "broker.hivemq.com",
    port= 1883,
    keepalive = 60
)


mqtt = FastMQTT(
    config=mqtt_config)
mqtt.init_app(app)
@mqtt.on_connect()
def connect(client, flags, rc, properties):
    mqtt.client.subscribe("kgx-vehicle") #subscribing mqtt topic
    print("Connected: ", client, flags, rc, properties)

global mess
@mqtt.on_message()
async def message(client, topic, payload, qos, properties):
    st=str(payload.decode("utf-8"))
    s=list(map(str,st.split(",")))
    vehicle_id=s[0]
    type=s[1]
    plate=s[2]
    print(s)
    vehicle = models.Vehicle_Base(title = str(type))
    db_id=models.Vehicle_Base(id=int(vehicle_id))
    db_plate=models.Vehicle_Base(num_plate=str(plate))
    with SessionLocal() as db: 
        db.add(vehicle)
        # db.add(db_id)
        db.add(db_plate)
        db.commit()
        db.refresh(vehicle)
    # print("Received message: ",topic, payload.decode(), qos, properties)
@mqtt.on_disconnect()
def disconnect(client, packet, exc=None):
    print("Disconnected") 

@mqtt.on_subscribe()
def subscribe(client, mid, qos, properties):
    print("subscribed", client, mid, qos, properties)

app.include_router(router, prefix="/vehicle", tags=["vehicles"])

