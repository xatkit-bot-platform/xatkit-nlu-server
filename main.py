from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import uuid
import json
from typing import Optional
from xatkitnlu.core.prediction import predict
from xatkitnlu.core.training import train
from xatkitnlu.dsl.dsl import Bot, NLUContext, PredictResult
from xatkitnlu.dto.dto import BotDTO, BotRequestDTO, ConfigurationDTO, configurationdto_to_configuration, \
    PredictRequestDTO, PredictResultDTO, ClassificationDTO, MatchedParamDTO
from xatkitnlu.dto.dto import botdto_to_bot

bots: dict[str, Bot] = {}

app = FastAPI()

# print(tf.__version__)


@app.get("/")
async def root():
    return {"Server running, using tensorflow version:": tf.__version__}


@app.get("/count/")
async def get_bots():
    return {"Count:": len(bots)}


@app.post("/bot/new/")
def bot_add(creation_request: BotRequestDTO):
    if creation_request.name in bots:
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


@app.post("/bot/{name}/initialize/")
def bot_initialize(name: str, botdto: BotDTO):
    if name not in bots:
        raise HTTPException(status_code=422, detail="Bot does not exist")
    bot: Bot = bots[name]

    botdto_to_bot(botdto, bot)
    return {"status:": "successful initialization with " + str(len(bot.contexts)) + " contexts"}


@app.post("/bot/{name}/train/")
def bot_train(name: str, configurationdto: ConfigurationDTO):
    if name not in bots:
        raise HTTPException(status_code=422, detail="Bot does not exist")
    bot: Bot = bots[name]
    if len(bot.contexts) == 0:
        raise HTTPException(status_code=422, detail="Bot is empty, nothing to train")
    bot.configuration = configurationdto_to_configuration(configurationdto)
    train(bot)
    return {"status:": "successful training for " + str(len(bot.contexts)) + " contexts"}


@app.post("/bot/{name}/predict/", response_model=PredictResultDTO)
def bot_predict(name: str, prediction_request: PredictRequestDTO):
    if name not in bots:
        raise HTTPException(status_code=422, detail="Bot does not exist")
    if len(prediction_request.utterance) == 0:
        raise HTTPException(status_code=422, detail="Utterance cannot be an empty string")
    bot: Bot = bots[name]
    context: Optional[NLUContext] = None
    i = 0
    while i < len(bot.contexts):
        if bot.contexts[i].name == prediction_request.context:
            context = bot.contexts[i]
            break
        i += 1
    if context is None:
        raise HTTPException(status_code=422, detail="Context not found in bot")
    if context.nlp_model is None:
        raise HTTPException(status_code=422, detail="Cannot predict on a context that has not been trained")

    prediction: PredictResult = predict(context, prediction_request.utterance, bot.configuration)

    # order of prediction values (classifications) matches order of intents.

    prediction_dto: PredictResultDTO = PredictResultDTO()

    for classification in prediction.classifications:
        classification_dto: ClassificationDTO = ClassificationDTO(intent=classification.intent.name,
                                                                  score=classification.score,
                                                                  matched_utterance=classification.matched_utterance,
                                                                  matched_params=[MatchedParamDTO(name=mp.name, value=mp.value, info=mp.info) for mp in classification.matched_params])
        prediction_dto.classifications.append(classification_dto)

    return prediction_dto


@app.get("/hello/{name}/")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
