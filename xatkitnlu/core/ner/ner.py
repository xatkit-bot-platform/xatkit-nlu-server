from xatkitnlu.core.ner.base.any import ner_any
from xatkitnlu.core.ner.base.base_entities import BaseEntityType, ordered_base_entities
from xatkitnlu.core.ner.base.datetime import ner_datetime
from xatkitnlu.core.ner.base.number import ner_number
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext, EntityReference, MatchedParam, BaseEntity
from xatkitnlu.utils.utils import value_in_sentence, replace_value_in_sentence


def no_ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> dict[Intent, tuple[str, list[MatchedParam]]]:
    result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    for intent in context.get_intents():
        result[intent] = (sentence, [])
    return result


def ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> dict[Intent, tuple[str, list[MatchedParam]]]:
    result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    for intent in context.get_intents():
        intent_matches: list[MatchedParam] = []
        ner_sentence: str = sentence
        # Match custom entities
        preprocessed_values: bool
        if configuration.stemmer:
            preprocessed_values = True
        else:
            preprocessed_values = False
        all_entity_values: dict[str, tuple[EntityReference, str]] = intent.get_custom_entity_values_dict(preprocessed_values)
        for value, (entity_ref, entry_value) in sorted(all_entity_values.items(), key=lambda x: (len(x[0]), x[0].casefold()), reverse=True):
            # entry_value are all entry values of the entity
            # value can be an entry value (i.e. value == entry_value)
            # or a synonym of an entry value (i.e. value is a synonym of entry_value)
            # value can be preprocessed
            if value_in_sentence(value, ner_sentence):
                intent_matches.append(MatchedParam(entity_ref.name, entry_value, {}))
                ner_sentence = replace_value_in_sentence(ner_sentence, value, entity_ref.entity.name.upper())
        # Match base/system entities (after custom entities)
        intent_base_entities: dict[str, str] = get_base_entity_names(intent.entity_parameters)
        for base_entity_name in ordered_base_entities:
            # Base entities must be checked in a specific order
            if base_entity_name in intent_base_entities:
                param_name = intent_base_entities[base_entity_name]
                formatted_ner_sentence, formatted_frag, param_info = base_entity_ner(ner_sentence, base_entity_name, configuration)
                if formatted_ner_sentence is not None and formatted_frag is not None and param_info is not None:
                    intent_matches.append(MatchedParam(param_name, formatted_frag, param_info))
                    ner_sentence = replace_value_in_sentence(formatted_ner_sentence, formatted_frag, base_entity_name.upper())
        matched_params_names = [mp.name for mp in intent_matches]
        for entity_param in intent.entity_parameters:
            if entity_param.name not in matched_params_names:
                intent_matches.append(MatchedParam(entity_param.name, None, {}))
        result[intent] = (ner_sentence, intent_matches)
    return result


def base_entity_ner(sentence: str, entity_name: str, configuration: NlpConfiguration) -> tuple[str, str, dict]:
    if entity_name == BaseEntityType.NUMBER:
        return ner_number(sentence, configuration)
    if entity_name == BaseEntityType.DATETIME:
        return ner_datetime(sentence, configuration)
    if entity_name == BaseEntityType.ANY:
        return ner_any(sentence, configuration)
    return None, None, None


def get_base_entity_names(entity_references: list[EntityReference]) -> dict[str, str]:
    base_entity_names: dict[str, str] = {}
    # {entity_name: param_name}
    for entity_ref in entity_references:
        param_name: str = entity_ref.name
        if isinstance(entity_ref.entity, BaseEntity) and entity_ref.entity.name in ordered_base_entities:
            base_entity_names[entity_ref.entity.name] = param_name
    return base_entity_names
