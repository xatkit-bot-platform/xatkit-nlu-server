import uuid
from pydantic import BaseModel
from typing import Optional

from dsl.dsl import Bot


class EntityDTO(BaseModel):
    """An entity to be recognized as part of the matching process"""
    name: str


class CustomEntityEntryDTO(BaseModel):
    """Each one of the entries (and its synonyms) a CustomEntity consists of"""
    value: str
    synonyms: list[str] = []


class CustomEntityDTO(EntityDTO):
    """ A custom entity, adhoc for the bot """
    entries: list[CustomEntityEntryDTO] = []


class IntentDTO(BaseModel):
    """A chatbot intent"""
    name: str
    training_sentences: list[str] = []


class NLUContextDTO(BaseModel):
    """Context state for which we must choose the right intent to match"""
    name: str
    intents: list[IntentDTO] = []


class BotDTO(BaseModel):
    """Running bot for which we are predicting the intent matching"""
    bot_id: str
    name: Optional[str] = None
    contexts: list[NLUContextDTO] = []


class PredictDTO(BaseModel):
    bot_id: str
    utterance: str
    context: str


class ConfigurationDTO(BaseModel):
    language: str


def from_botdto_to_bot(botdto: BotDTO, bot: Bot):
    """Creates an internal bot representation from a botDTO object """
    pass