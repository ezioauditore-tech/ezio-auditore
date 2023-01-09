import paho.mqtt.client as mqtt

def on_message(client,userdata,message):
    load_msg=message.payload
    print("recieved message: ",str(message.payload.decode("utf-8")))

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("kgx-vehicle")

client=mqtt.Client()
client.on_message=on_message
client.on_connect=on_connect

client.connect("broker.hivemq.com", 1883, 60)
client.loop_forever()
