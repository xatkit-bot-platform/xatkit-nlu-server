from fastapi import FastAPI

import tensorflow as tf

# UUID management
import uuid

from dsl.dsl import Bot
from dto.dto import BotDTO
from dto.dto import from_botdto_to_bot

bots: dict[uuid, Bot] = {}

app = FastAPI()

# print(tf.__version__)


@app.get("/")
async def root():
    return {"Server running, using tensorflow version:" : tf.__version__}


@app.post("/bot/new/")
def bot_add(name: str):
    uuid_value: uuid = uuid.uuid4()
    bots[uuid_value] = Bot(uuid_value, name)
    return {"uuid": str(uuid_value)}


@app.post("/bot/initialize")
def bot_initialize(botdto: BotDTO):
    bot: Bot = bots[botdto.bot_id]
    from_botdto_to_bot(botdto, bot)
    return "ok"


@app.post("/bot/train")
def bot_train(bot_id: str):
    bot: Bot = bots[bot_id]


@app.post("/bot/predict")
def bot_train(bot_id: str):
    bot: Bot = bots[bot_id]



@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
