from pydantic import BaseModel
from typing import Optional
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, Intent


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




class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True


class BotDTO(OurBaseModel):
    """Running bot for which we are predicting the intent matching"""
    name: str
    contexts: list[NLUContextDTO] = []

class BotRequestDTO(BaseModel):
    name: str
    force_overwrite: bool


class PredictDTO(BaseModel):
    utterance: str
    context: str


class PredictResultDTO(BaseModel):
    prediction_values: list[float]
    intents: list[str]
    matched_utterances: list[str]


class ConfigurationDTO(BaseModel):
    country: Optional[str]
    region: Optional[str]
    num_words: Optional[int]  # max num of words to keep in the index of words
    num_epochs: Optional[int]
    lower: Optional[bool]  # transform sentences to lowercase
    oov_token: Optional[str]  # token for the out of vocabulary words
    embedding_dim: Optional[int]
    input_max_num_tokens: Optional[int]  # max length for the vector representing a sentence
    stemmer: Optional[bool]  # whether to use a stemmer


def botdto_to_bot(botdto: BotDTO, bot: Bot):
    """Creates an internal bot representation from a botDTO object """
    bot.contexts = []
    for context in botdto.contexts:
        bot.contexts.append(contextdto_to_context(context))


def contextdto_to_context(contextdto: NLUContextDTO) -> NLUContext:
    context: NLUContext = NLUContext(contextdto.name)
    for intentdto in contextdto.intents:
        context.intents.append(intentdto_to_intent(intentdto))
    return context


def intentdto_to_intent(intentdto: IntentDTO) -> Intent:
    intent: Intent = Intent(intentdto.name, intentdto.training_sentences)
    return intent


def configurationdto_to_configuration(configurationdto: ConfigurationDTO) -> NlpConfiguration:
    configuration: NlpConfiguration = NlpConfiguration()
    if configurationdto.country is not None:
        configuration.country = configurationdto.country
    if configurationdto.region is not None:
        configuration.region = configurationdto.region
    if configurationdto.num_words is not None:
        configuration.num_words = configurationdto.num_words
    if configurationdto.lower is not None:
        configuration.lower = configurationdto.lower
    if configurationdto.oov_token is not None:
        configuration.oov_token = configurationdto.oov_token
    if configurationdto.num_epochs is not None:
        configuration.num_epochs = configurationdto.num_epochs
    if configurationdto.embedding_dim is not None:
        configuration.embedding_dim = configurationdto.embedding_dim
    if configurationdto.input_max_num_tokens is not None:
        configuration.input_max_num_tokens = configurationdto.input_max_num_tokens
    if configurationdto.stemmer is not None:
        configuration.stemmer = configurationdto.stemmer
    return configuration