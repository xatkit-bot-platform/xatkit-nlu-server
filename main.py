from fastapi import FastAPI

# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# UUID management
import uuid

from dsl.dsl import Bot

bots: list[Bot] = []

app = FastAPI()

print(tf.__version__)


@app.get("/")
async def root():
    return "Hello World"


@app.post("/bot/new/")
async def bot_add():
    uuid_value = uuid.uuid4()
    bots.append(Bot(uuid_value))
    return {"botid": uuid_value}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
