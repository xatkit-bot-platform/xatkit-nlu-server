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


class IntentParameter:
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
        self.parameters: list[IntentParameter] = []

    def add_training_sentence(self, sentence: str):
        self.training_sentences.append(sentence)

    def add_parameter(self, parameter: IntentParameter):
        self.parameters.append(parameter)

    def get_custom_entity_values_dict(self, preprocessed_values: bool = False) -> dict[str, tuple[list[IntentParameter], str]]:
        # {value/synonym: ([entity_refs], value)}
        all_entity_values: dict[str, tuple[list[IntentParameter], str]] = {}
        entity_refs_dict: dict[Entity, list[IntentParameter]] = {}
        for entity_ref in self.parameters:
            if entity_ref.entity in entity_refs_dict:
                entity_refs_dict[entity_ref.entity].append(entity_ref)
            else:
                entity_refs_dict[entity_ref.entity] = [entity_ref]
        for entity, entity_refs in entity_refs_dict.items():
            if isinstance(entity, CustomEntity):
                # {value/synonym: value}
                entity_values_dict: dict[str, str] = {}
                for entity_entry in entity.entries:
                    if preprocessed_values and entity_entry.preprocessed_value is not None and entity_entry.preprocessed_synonyms is not None:
                        value = entity_entry.preprocessed_value
                        synonyms = entity_entry.preprocessed_synonyms
                    else:
                        value = entity_entry.value
                        synonyms = entity_entry.synonyms
                    values = [value]
                    values.extend(synonyms)
                    for v in values:
                        if v in entity_values_dict:
                            # TODO: duplicated value in entity
                            pass
                        else:
                            entity_values_dict[v] = entity_entry.value

                for v, value in entity_values_dict.items():
                    if v in all_entity_values:
                        # 2 entities have the same v
                        if all_entity_values[v][1] == value:
                            # The same value can be in different entities
                            # We order the merge of all possible references, based on the original order in the intent definition
                            v_refs = [ref for ref in self.parameters
                                      if ref in all_entity_values[v][0] + entity_refs]
                            all_entity_values[v] = (v_refs, value)
                        else:
                            # TODO: duplicated v with different values
                            pass
                    else:
                        all_entity_values[v] = (entity_refs.copy(), value)
        return all_entity_values

    def __repr__(self):
        return f'Intent({self.name},{self.training_sentences},{self.parameters})'


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

    # For testing
    def get_context(self, name: str):
        for context in self.contexts:
            if context.name == name:
                return context
        return None

    def get_intent(self, name: str):
        for intent in self.intents:
            if intent.name == name:
                return intent
        return None

    def get_entity(self, name: str):
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def __repr__(self):
        return f'Bot({self.bot_id},{self.name},{self.contexts})'


class MatchedParameter:

    def __init__(self, name: str, value: str, info: dict[str, object]):
        self.name = name
        self.value = value
        self.info = info


class Classification:

    def __init__(self, intent: Intent, score: float = None, matched_utterance: str = None,
                 matched_parameters: list[MatchedParameter] = None):
        self.intent: Intent = intent
        self.score: float = score
        self.matched_utterance: str = matched_utterance
        self.matched_parameters: list[MatchedParameter] = matched_parameters
        # if matched_parameters is None:
        #     self.matched_parameters: list[MatchedParameter] = []


class PredictResult:

    def __init__(self, context: NLUContext):
        self.classifications: list[Classification] = []
        for intent in context.get_intents():
            self.classifications.append(Classification(intent))

    def get_classification(self, intent: Intent) -> Classification:
        for classification in self.classifications:
            if classification.intent == intent:
                return classification
