from fastapi import FastAPI

# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# UUID management
import uuid

from dsl.dsl import Bot
from dto.dto import BotDTO
from dto.dto import from_botdto_to_bot

bots: dict[uuid, Bot] = {}

app = FastAPI()

print(tf.__version__)


@app.get("/")
async def root():
    return "Hello World"


@app.post("/bot/new/")
def bot_add(name: str):
    uuid_value = uuid.uuid4()
    bots[uuid_value] = Bot(uuid_value, name)
    return {"bot_id": uuid_value}


@app.post("/bot/initialize")
def bot_initialize(botdto: BotDTO):
    bot: Bot = bots[botdto.bot_id]
    from_botdto_to_bot(botdto, bot)
    return botdto


@app.post("/bot/train")
def bot_train(bot_id: uuid):
    bot: Bot = bots[uuid]
    bot.train






@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
