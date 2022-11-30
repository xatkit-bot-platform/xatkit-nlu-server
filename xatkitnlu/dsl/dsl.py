import uuid
import tensorflow as tf

from xatkitnlu.core.nlp_configuration import NlpConfiguration


class Entity:
    """An entity to be recognized as part of the matching process"""

    def __init__(self, name: str):
        self.name: str = name


class BaseEntity(Entity):
    """ A base entity """

    def __init__(self, name: str):
        super().__init__(name)


class CustomEntityEntry:
    """Each one of the entries (and its synonyms) a CustomEntity consists of"""

    def __init__(self, value: str, synonyms: list[str] = []):
        self.value: str = value
        self.synonyms: list[str] = synonyms
        self.preprocessed_value: str = None
        self.preprocessed_synonyms: list[str] = None


class CustomEntity(Entity):
    """ A custom entity, adhoc for the bot """

    def __init__(self, name: str, entries: list[CustomEntityEntry] = None):
        super().__init__(name)
        if (entries is not None):
            self.entries: list[CustomEntityEntry] = entries
        else:
            self.entries: list[CustomEntityEntry] = []


class EntityReference:
    """A parameter of an Intent, representing an entity that is expected to be matched"""

    def __init__(self, name: str, fragment: str, entity: Entity):
        self.entity: Entity = entity  # Entity type to be matched
        self.name: str = name  # name of the parameter
        self.fragment: str = fragment  # fragment of the text representing the entity ref in a training sentence


class Intent:
    """A chatbot intent"""

    def __init__(self, name: str, training_sentences: list[str]):
        self.name: str = name
        self.training_sentences: list[str] = training_sentences
        self.processed_training_sentences: list[str] = []
        self.training_sequences: list[int] = []
        # list of references to entities used in the Intent
        # we are going to assume that two intents in the same context do not have parameters with the same name unless they refer to the same entity type
        self.entity_parameters: list[EntityReference] = []

    def add_training_sentence(self, sentence: str):
        self.training_sentences.append(sentence)

    def add_entity_parameter(self, entity_parameter: EntityReference):
        self.entity_parameters.append(entity_parameter)

    def get_custom_entity_values_dict(self, preprocessed_values: bool = False) -> dict[str, tuple[EntityReference, str]]:
        all_entity_values: dict[str, tuple[EntityReference, str]] = {}
        # {value/synonym: (entity_ref, value)}
        for entity_ref in self.entity_parameters:
            if isinstance(entity_ref.entity, CustomEntity):
                for entity_entry in entity_ref.entity.entries:
                    if entity_entry.value in all_entity_values.keys():
                        # TODO: ENTITY OVERLAPPING
                        pass
                    if preprocessed_values and entity_entry.preprocessed_value is not None and entity_entry.preprocessed_synonyms is not None:
                        value = entity_entry.preprocessed_value
                        synonyms = entity_entry.preprocessed_synonyms
                    else:
                        value = entity_entry.value
                        synonyms = entity_entry.synonyms

                    all_entity_values[value] = (entity_ref, entity_entry.value)
                    if synonyms is not None:
                        for synonym in synonyms:
                            if synonym in all_entity_values.keys():
                                # TODO: ENTITY OVERLAPPING
                                pass
                            all_entity_values[synonym] = (entity_ref, entity_entry.value)
        return all_entity_values

    def __repr__(self):
        return f'Intent({self.name},{self.training_sentences},{self.entity_parameters})'


class IntentReference:
    """A reference to an intent that can be matched from a given NLUContext"""
    def __init__(self, name: str, intent: Intent):
        self.name = name
        self.intent = intent


class NLUContext:
    """Context state for which we must choose the right intent to match"""

    def __init__(self, name: str):
        self.name: str = name
        self.intent_refs: list[IntentReference] = []
        self.tokenizer: tf.keras.preprocessing.text.Tokenizer = None
        self.training_sentences: list[str] = []
        self.training_sequences: list[int] = []
        self.training_labels: list[int] = []
        self.nlp_model: tf.keras.models = None

    def add_intent_ref(self, intent_ref: IntentReference):
        self.intent_refs.append(intent_ref)

    def get_intents(self):
        return [intent_ref.intent for intent_ref in self.intent_refs]

    def __repr__(self):
        return f'Context({self.name},{self.intent_refs})'


class Bot:
    """Running bot for which we are predicting the intent matching"""

    def __init__(self, bot_id: uuid, name: str, configuration: NlpConfiguration = None):
        self.bot_id: uuid = bot_id
        self.name: str = name
        self.entities: list[Entity] = []
        self.intents: list[Intent] = []
        self.contexts: list[NLUContext] = []
        self.configuration: NlpConfiguration = configuration

    def add_context(self, context: NLUContext):
        self.contexts.append(context)

    def add_intent(self, intent: Intent):
        self.intents.append(intent)

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def __repr__(self):
        return f'Bot({self.bot_id},{self.name},{self.contexts})'


class MatchedParam:

    def __init__(self, name: str, value: str, info: dict[str, object]):
        self.name = name
        self.value = value
        self.info = info


class Classification:

    def __init__(self, intent: Intent, score: float = None, matched_utterance: str = None,
                 matched_params: list[MatchedParam] = None):
        self.intent: Intent = intent
        self.score: float = score
        self.matched_utterance: str = matched_utterance
        self.matched_params: list[MatchedParam] = matched_params
        # if matched_params is None:
        #     self.matched_params: list[MatchedParam] = []


class PredictResult:

    def __init__(self, context: NLUContext):
        self.classifications: list[Classification] = []
        for intent in context.get_intents():
            self.classifications.append(Classification(intent))

    def get_classification(self, intent: Intent) -> Classification:
        for classification in self.classifications:
            if classification.intent == intent:
                return classification
