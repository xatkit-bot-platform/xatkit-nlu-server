import uuid
import tensorflow as tf

from core.nlp_configuration import NlpConfiguration


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
    processed_training_sentences: list[str] = []
    training_sequences: list[int] = []

    def __init__(self, name: str, training_sentences: list[str]):
        self.name = name
        self.training_sentences = training_sentences

    def add_training_sentence(self, sentence: str):
        self.training_sentences.append(sentence)

    def __repr__(self) :
        return f'Intent({self.name},{self.training_sentences})'


class NLUContext:
    """Context state for which we must choose the right intent to match"""
    name: str
    intents: list[Intent] = []
    tokenizer: tf.keras.preprocessing.text.Tokenizer = None
    training_sentences: list[str] = []
    training_sequences: list[int] = []
    training_labels: list[int] = []
    nlp_model : tf.keras.models = None

    def __init__(self, name: str):
        self.name = name

    def add_intent(self, intent: Intent):
        self.intents.append(intent)

    def __repr__(self) :
       return f'Context({self.name},{self.intents})'


class Bot:
    """Running bot for which we are predicting the intent matching"""
    bot_id: uuid
    name: str
    contexts: list[NLUContext] = []
    configuration: NlpConfiguration


    def __init__(self, bot_id: uuid, name: str, configuration: NlpConfiguration = None):
        self.bot_id = bot_id
        self.name = name
        if configuration is not None:
            self.configuration = configuration

    def initialize(self, contexts: list[NLUContext], configuration: NlpConfiguration):
        self.contexts = contexts
        self.configuration = configuration

    def add_context(self, context: NLUContext):
        self.contexts.append(context)

    def __repr__(self):
        return f'Bot({self.bot_id},{self.name},{self.contexts})'

