from fastapi import FastAPI

# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# UUID management
import uuid

bots: list[uuid] = []

app = FastAPI()

print(tf.__version__)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/bot/add/")
async def bot_add():
    uuid_value = uuid.uuid4()
    bots.append(uuid_value)
    return {"botid": uuid_value}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
