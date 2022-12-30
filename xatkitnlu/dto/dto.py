from pydantic import BaseModel
from typing import Optional
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, Intent, Entity, CustomEntity, CustomEntityEntry, EntityReference, \
    BaseEntity, IntentReference


class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True


class CustomEntityEntryDTO(BaseModel):
    """Each one of the entries (and its synonyms) a CustomEntity consists of"""
    value: str
    synonyms: list[str] = []


# We do not have separate classes for Generic entities and custom entities as Pydantic seems to have problems with
# inherited attributes when calling initialize in the API
class EntityDTO(BaseModel):
    """ An entity used in the bot """
    name: str
    # entries will only have values for custom entity types
    entries: list[CustomEntityEntryDTO] = []


class EntityReferenceDTO(BaseModel):
    """A reference to an entity from an Intent"""
    name: str
    fragment: str
    entity: EntityDTO


class IntentDTO(BaseModel):
    """A chatbot intent"""
    name: str
    training_sentences: list[str] = []
    entity_parameters: list[EntityReferenceDTO] = []


class IntentReferenceDTO(BaseModel):
    """A reference to an intent from a context"""
    name: str
    intent: IntentDTO


class NLUContextDTO(BaseModel):
    """Context state for which we must choose the right intent to match"""
    name: str
    intent_refs: list[IntentReferenceDTO] = []


class BotDTO(OurBaseModel):
    """Running bot for which we are predicting the intent matching"""
    name: str
    contexts: list[NLUContextDTO] = []
    entities: list[EntityDTO] = []
    intents: list[IntentDTO] = []


class BotRequestDTO(BaseModel):
    name: str
    force_overwrite: bool


class PredictRequestDTO(BaseModel):
    utterance: str
    context: str


class MatchedParamDTO(BaseModel):
    name: str
    value: Optional[str]
    info: Optional[dict[str, object]]


class ClassificationDTO(BaseModel):
    intent: str
    score: float
    matched_utterance: str
    matched_params: list[MatchedParamDTO]


class PredictResultDTO(BaseModel):
    classifications: list[ClassificationDTO] = []


class ConfigurationDTO(BaseModel):
    country: Optional[str]
    region: Optional[str]
    timezone: Optional[str]
    num_words: Optional[int]  # max num of words to keep in the index of words
    num_epochs: Optional[int]
    lower: Optional[bool]  # transform sentences to lowercase
    oov_token: Optional[str]  # token for the out of vocabulary words
    embedding_dim: Optional[int]
    input_max_num_tokens: Optional[int]  # max length for the vector representing a sentence
    stemmer: Optional[bool]  # whether to use a stemmer
    use_ner_in_prediction: Optional[bool]  # whether to use NER in the prediction


def botdto_to_bot(botdto: BotDTO, bot: Bot):
    """Creates an internal bot representation from a botDTO object """
    bot.contexts = []
    bot.intents = []
    bot.entities = []
    for entity in botdto.entities:
        bot.add_entity(entitydto_to_entity(entity))
    for intent in botdto.intents:
        bot.add_intent(intentdto_to_intent(intent, bot))
    for context in botdto.contexts:
        bot.add_context(contextdto_to_context(context, bot))


def contextdto_to_context(contextdto: NLUContextDTO, bot: Bot) -> NLUContext:
    context: NLUContext = NLUContext(contextdto.name)
    for intentrefdto in contextdto.intent_refs:
        context.add_intent_ref(intentrefdto_to_intentref(intentrefdto, bot))
    return context


def intentrefdto_to_intentref(intentrefdto: IntentReferenceDTO, bot: Bot) -> IntentReference:
    intent: Intent = find_intent_in_bot_by_name(intentrefdto.intent.name, bot)
    intent_ref = IntentReference(intentrefdto.name, intent)
    return intent_ref


def intentdto_to_intent(intentdto: IntentDTO, bot: Bot) -> Intent:
    intent: Intent = Intent(intentdto.name, intentdto.training_sentences)
    for entityref in intentdto.entity_parameters:
        ref: EntityReference = entityrefdto_to_entityref(entityref, bot)
        intent.add_entity_parameter(ref)
    return intent


def entitydto_to_entity(entitydto: EntityDTO) -> Entity:
    if len(entitydto.entries) > 0:  # simple way to check if it is a custom entity
        return custom_entitydto_to_entity(entitydto)
    else:
        return base_entitydto_to_entity(entitydto)


def base_entitydto_to_entity(base_entitydto: EntityDTO) -> BaseEntity:
    entity = BaseEntity(base_entitydto.name)
    return entity


def custom_entitydto_to_entity(custom_entitydto: EntityDTO) -> CustomEntity:
    entity = CustomEntity(name=custom_entitydto.name)
    for entry in custom_entitydto.entries:
        entity.entries.append(CustomEntityEntry(entry.value, entry.synonyms))
    return entity


def entityrefdto_to_entityref(entityrefdto: EntityReferenceDTO, bot: Bot) -> EntityReference:
    entity: Entity = find_entity_in_bot_by_name(entityrefdto.entity.name, bot)
    entityref: EntityReference = EntityReference(entity=entity, name=entityrefdto.name, fragment=entityrefdto.fragment)
    return entityref


def find_entity_in_bot_by_name(name: str, bot: Bot) -> Entity:
    for entity in bot.entities:
        if entity.name == name:
            return entity
    return None


def find_intent_in_bot_by_name(name: str, bot: Bot) -> Intent:
    for intent in bot.intents:
        if intent.name == name:
            return intent
    return None


def configurationdto_to_configuration(configurationdto: ConfigurationDTO) -> NlpConfiguration:
    configuration: NlpConfiguration = NlpConfiguration()
    if configurationdto.country is not None:
        configuration.country = configurationdto.country
    if configurationdto.region is not None:
        configuration.region = configurationdto.region
    if configurationdto.timezone is not None:
        configuration.timezone = configurationdto.timezone
    if configurationdto.num_words is not None:
        configuration.num_words = configurationdto.num_words
    if configurationdto.num_epochs is not None:
        configuration.num_epochs = configurationdto.num_epochs
    if configurationdto.lower is not None:
        configuration.lower = configurationdto.lower
    if configurationdto.oov_token is not None:
        configuration.oov_token = configurationdto.oov_token
    if configurationdto.embedding_dim is not None:
        configuration.embedding_dim = configurationdto.embedding_dim
    if configurationdto.input_max_num_tokens is not None:
        configuration.input_max_num_tokens = configurationdto.input_max_num_tokens
    if configurationdto.stemmer is not None:
        configuration.stemmer = configurationdto.stemmer
    if configurationdto.use_ner_in_prediction is not None:
        configuration.use_ner_in_prediction = configurationdto.use_ner_in_prediction
    return configuration
