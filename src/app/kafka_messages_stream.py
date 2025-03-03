import os
import requests
from kafka import KafkaProducer
import json
from time import sleep
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

API_KEY = os.getenv("PRIM_API_KEY")
BASE_API_URL = "https://prim.iledefrance-mobilites.fr/marketplace"
TOPIC_NAME="idfm_message"

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=10,
    acks="all"
)

headers = {
  "apikey": API_KEY
}

response = requests.get(f"{BASE_API_URL}/disruptions_bulk/disruptions/v2", headers=headers)

if(response.status_code == 200): 
  data = response.json()
  
  if(data["disruptions"]): 
    for record in data["disruptions"]:
      record_id = record["id"]
      print(f"Sending message for record {record_id}")
      
      producer.send(TOPIC_NAME, record)
      producer.flush()
      sleep(2)
      

      
  

