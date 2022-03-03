from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy
import uuid
import json
from core.prediction import predict
from core.training import train
from dsl.dsl import Bot, NLUContext
from dto.dto import BotDTO, BotRequestDTO, ConfigurationDTO, configurationdto_to_configuration, PredictDTO
from dto.dto import botdto_to_bot

bots: dict[str, Bot] = {}

app = FastAPI()

# print(tf.__version__)


@app.get("/")
async def root():
    return {"Server running, using tensorflow version:": tf.__version__}


@app.post("/bot/new/")
def bot_add(creation_request: BotRequestDTO):
    if creation_request.name in bots.keys():
        if not creation_request.force_overwrite:
            raise HTTPException(status_code=422, detail="Bot name already in use")
        else:
            # We delete the previous bot with the same name and replace it with this new one
            bot_to_delete = bots[creation_request.name]
            bots.pop(creation_request.name)
            del bot_to_delete
    uuid_value: uuid = uuid.uuid4()
    bots[creation_request.name] = Bot(uuid_value, creation_request.name)
    return {"uuid": str(uuid_value)}


@app.post("/bot/{name}/initialize")
def bot_initialize(name: str, botdto: BotDTO):
    if name not in bots.keys():
        raise HTTPException(status_code=422, detail="Bot does not exist")
    bot: Bot = bots[name]

    botdto_to_bot(botdto, bot)
    return "ok"


@app.post("/bot/{name}/train")
def bot_train(name: str, configurationdto: ConfigurationDTO):
    if name not in bots.keys():
        raise HTTPException(status_code=422, detail="Bot does not exist")
    bot: Bot = bots[name]
    if len(bot.contexts) == 0:
        raise HTTPException(status_code=422, detail="Bot is empty, nothing to train")
    bot.configuration = configurationdto_to_configuration(configurationdto)
    train(bot)


@app.post("/bot/{name}/predict")
def bot_train(name: str, prediction_request: PredictDTO):
    bot: Bot = bots[name]
    context: NLUContext = bot.context[bot.context.index(prediction_request.context)]
    predictions_stemmer: list[numpy.ndarray] = predict(bot, context, bot.configuration)
    return {"prediction": json.dumps(predictions_stemmer.tolist())}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
