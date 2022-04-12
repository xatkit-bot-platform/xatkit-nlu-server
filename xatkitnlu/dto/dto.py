from pydantic import BaseModel
from typing import Optional
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, Intent, Entity, CustomEntity, CustomEntityEntry, EntityReference


class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True


class EntityDTO(BaseModel):
    """An entity to be recognized as part of the matching process"""
    name: str


class CustomEntityEntryDTO(BaseModel):
    """Each one of the entries (and its synonyms) a CustomEntity consists of"""
    value: str
    synonyms: list[str] = []


# We do not inherit from EntityDTO as Pydantic seems to lose the type information at some point when calling initialize in the API
class CustomEntityDTO(BaseModel):
    """ A custom entity, adhoc for the bot """
    name: str
    entries: list[CustomEntityEntryDTO] = []


class EntityReferenceDTO(BaseModel):
    """A reference to an entity from an Intent"""
    entity: EntityDTO
    fragment: str
    name: str


class IntentDTO(BaseModel):
    """A chatbot intent"""
    name: str
    training_sentences: list[str] = []
    entity_parameters: list[EntityReferenceDTO] = []


class NLUContextDTO(BaseModel):
    """Context state for which we must choose the right intent to match"""
    name: str
    intents: list[IntentDTO] = []
    custom_entities: list[CustomEntityDTO] = []


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
    matched_params: dict[str,str]


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
    for custom_entitydto in contextdto.custom_entities:
        context.add_entity(custom_entitydto_to_entity(custom_entitydto))
    for intentdto in contextdto.intents:
        context.add_intent(intentdto_to_intent(intentdto, context))
    return context


def intentdto_to_intent(intentdto: IntentDTO, context: NLUContext) -> Intent:
    intent: Intent = Intent(intentdto.name, intentdto.training_sentences)
    for entityref in intentdto.entity_parameters:
        ref:EntityReference = custom_entityrefdto_to_entityref(entityref, context)
        intent.add_entity_parameter(ref)
    return intent

def custom_entitydto_to_entity(custom_entitydto: CustomEntityDTO) -> Entity:
    if isinstance(custom_entitydto, CustomEntityDTO):
        entity = CustomEntity(name=custom_entitydto.name)
        for entry in custom_entitydto.entries:
            entity.entries.append(CustomEntityEntry(entry.value, entry.synonyms))
    else:
        entity = Entity(custom_entitydto.name)
    return entity


def custom_entityrefdto_to_entityref(entityrefdto: EntityReferenceDTO, context: NLUContext) -> EntityReference:
    entity: CustomEntity = find_custom_entity_in_context_by_name(entityrefdto.entity.name, context)
    entityref: EntityReference = EntityReference(entity=entity, name=entityrefdto.entity.name, fragment=entityrefdto.fragment)
    return entityref

def find_custom_entity_in_context_by_name(name: str, context: NLUContext) -> CustomEntity:
    for entity in context.entities:
        if entity.name == name:
            return entity
    return None

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