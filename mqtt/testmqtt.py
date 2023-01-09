import paho.mqtt.client as mqtt
import time


# def create_client(topic):
#     mqttBroker = "mqtt.eclipseprojects.io"
#     client=mqtt.Client("Phone")
#     client.connect(mqttBroker)
    
#     def on_message(client,userdata,message):
#         print("recieved message: ",str(message.payload.decode("utf-8")))

#     client.loop_start()
#     client.subscribe("Vehicle")
#     client.on_message=on_message
#     time.sleep(30)
#     client.loop_stop()


class mqtt_sub:
    def __init__(self, topic):
        self.topic = topic

    def create_client(self):
        mqttBroker = "mqtt.eclipseprojects.io"
        client=mqtt.Client("Phone")
        client.connect(mqttBroker)
    
        def on_message(client,userdata,message):
            print("recieved message: ",str(message.payload.decode("utf-8")))

        client.loop_start()
        client.subscribe(self.topic)
        client.on_message=on_message
        time.sleep(30)
        client.loop_stop()
        
    
mqtt_obj = mqtt_sub("Temperature")
mqtt_obj.create_client()