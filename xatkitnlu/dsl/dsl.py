import uuid
import tensorflow as tf

from xatkitnlu.core.nlp_configuration import NlpConfiguration


class Entity:
    """An entity to be recognized as part of the matching process"""

    def __init__(self, name: str):
        self.name: str = name


class CustomEntityEntry:
    """Each one of the entries (and its synonyms) a CustomEntity consists of"""

    def __init__(self, value: str, synonyms: list[str] = None):
        self.value: str = value
        self.synonyms: list[str] = synonyms


class CustomEntity(Entity):
    """ A custom entity, adhoc for the bot """

    def __init__(self, name: str, entries: list[CustomEntityEntry]):
        super().__init__(name)
        self.entries: list[CustomEntityEntry] = entries


class EntityReference:
    """A parameter of an Intent, representing an entity that is expected to be matched"""

    def __init__(self, name: str, entity: Entity, fragment):
        self.entity: Entity = entity  # Entity type to be matched
        self.name: str = name  # name of the parameter
        self.fragment: str = fragment  # fragment of the text representing the entity ref in a training sentence


class Intent:
    """A chatbot intent"""
    def __init__(self, name: str, training_sentences: list[str], entity_refs: list[EntityReference] = None):
        self.name: str = name
        self.training_sentences: list[str] = training_sentences
        self.processed_training_sentences: list[str] = []
        self.training_sequences: list[int] = []
        self.entity_parameters: list[EntityReference] = entity_refs   # list of references to entities used in the Intent

    def add_training_sentence(self, sentence: str):
        self.training_sentences.append(sentence)

    def __repr__(self):
        return f'Intent({self.name},{self.training_sentences},{self.entity_parameters})'


class NLUContext:
    """Context state for which we must choose the right intent to match"""
    def __init__(self, name: str):
        self.name: str = name
        self.intents: list[Intent] = []
        self.tokenizer: tf.keras.preprocessing.text.Tokenizer = None
        self.training_sentences: list[str] = []
        self.training_sequences: list[int] = []
        self.training_labels: list[int] = []
        self.nlp_model: tf.keras.models = None

    def add_intent(self, intent: Intent):
        self.intents.append(intent)

    def __repr__(self):
        return f'Context({self.name},{self.intents})'


class Bot:
    """Running bot for which we are predicting the intent matching"""

    def __init__(self, bot_id: uuid, name: str, configuration: NlpConfiguration = None):
        self.bot_id: uuid = bot_id
        self.name: str = name
        self.contexts: list[NLUContext] = []
        self.configuration: NlpConfiguration = configuration

    def initialize(self, contexts: list[NLUContext], configuration: NlpConfiguration):
        self.contexts = contexts
        self.configuration = configuration

    def add_context(self, context: NLUContext):
        self.contexts.append(context)

    def __repr__(self):
        return f'Bot({self.bot_id},{self.name},{self.contexts})'
