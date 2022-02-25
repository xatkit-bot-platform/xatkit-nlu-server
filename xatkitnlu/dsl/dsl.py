import uuid
import tensorflow as tf


class Entity:
    """An entity to be recognized as part of the matching process"""
    name: str

    def __init__(self, name: str):
        self.name = name


class CustomEntityEntry:
    """Each one of the entries (and its synonyms) a CustomEntity consists of"""
    value: str
    synonyms: list[str] = []

    def __init__(self, value: str, synonyms: list[str]):
        self.value = value
        self.synonyms = synonyms


class CustomEntity(Entity):
    """ A custom entity, adhoc for the bot """
    entries: list[CustomEntityEntry] = []

    def __init__(self, name: str, entries: list[CustomEntityEntry]):
        super().__init__(name)
        self.entries = entries


class Intent:
    """A chatbot intent"""
    name: str
    training_sentences: list[str] = []

    def __init__(self, name: str, training_sentences: list[str]):
        self.name = name
        self.training_sentences = training_sentences

    def add_training_sentence(self, sentence: str):
        self.training_sentences.append(sentence)


class NLUContext:
    """Context state for which we must choose the right intent to match"""
    name: str
    intents: list[Intent] = []
    tokenizer: tf.keras.preprocessing.text.Tokenizer = None

    def __init__(self, name: str):
        self.name = name

    def add_intent(self, intent: Intent):
        self.intents.append(intent)


class Configuration:
    country: str = "en"
    region: str = "US"
    numwords: int = 100

    def __init__(self, country: str, region: str):
        self.country = country
        self.region = region


class Bot:
    """Running bot for which we are predicting the intent matching"""
    bot_id: uuid
    name: str
    contexts: list[NLUContext] = []
    configuration: Configuration = None

    def __init__(self, bot_id: uuid, name: str, configuration: Configuration = None):
        self.bot_id = bot_id
        self.name = name
        if configuration is not None:
            self.configuration = configuration

    def initialize(self, contexts: list[NLUContext], configuration: Configuration):
        self.contexts = contexts
        self.configuration = configuration

    def add_context(self, context: NLUContext):
        self.contexts.append(context)
