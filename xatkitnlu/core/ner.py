import re

from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext, EntityReference, CustomEntity, MatchedParam


def no_ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> dict[Intent, tuple[str, list[MatchedParam]]]:
    result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    for intent in context.intents:
        result[intent] = (sentence, [])
    return result


def ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> dict[Intent, tuple[str, list[MatchedParam]]]: # tuple[str, dict[Intent, dict[str, str]]]:
    result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    for intent in context.intents:
        intent_matches: list[MatchedParam] = []
        ner_sentence: str = sentence
        if intent.entity_parameters is not None:
            all_entity_values: dict[str, tuple[str, str, str]] = create_entity_values_dict(intent.entity_parameters)
            for value, (entry_value, param_name, entity_name) in sorted(all_entity_values.items(), reverse=True):
                # entry_value are all entry values of the entity
                # value can be an entry value (i.e. value == entry_value)
                # or a synonym of an entry value (i.e. value is a synonym of entry_value)
                if param_value_in_sentence(value, ner_sentence):
                    intent_matches.append(MatchedParam(param_name, entry_value))
                    ner_sentence = replace_param_value_with_entity_type_name_ner_sentence(ner_sentence, value, entity_name.upper())
            result[intent] = (ner_sentence, intent_matches)
    return result


def create_entity_values_dict(entity_references: list[EntityReference]) -> dict[str, tuple[str, str, str]]:
    all_entity_values: dict[str, tuple[str, str, str]] = {}
    for entity_ref in entity_references:
        if isinstance(entity_ref.entity, CustomEntity):
            param_name: str = entity_ref.name
            entity_name: str = entity_ref.entity.name
            for entity_entry in entity_ref.entity.entries:
                if entity_entry in all_entity_values.keys():
                    # ENTITY OVERLAPPING
                    True
                all_entity_values[entity_entry.value] = (entity_entry.value, param_name, entity_name)
                if entity_entry.synonyms is not None:
                    for synonym in entity_entry.synonyms:
                        if synonym in all_entity_values.keys():
                            # ENTITY OVERLAPPING
                            True
                        all_entity_values[synonym] = (entity_entry.value, param_name, entity_name)
    return all_entity_values


def param_value_in_sentence(param_value: str, sentence: str) -> bool:
    regex = re.compile(r'\b' + re.escape(param_value) + r'\b', re.IGNORECASE)
    return regex.search(sentence) is not None


def replace_param_value_with_entity_type_name_ner_sentence(sentence: str, found_param_value: str, entity_type_name: str) -> str:
    regex = re.compile(r'\b' + re.escape(found_param_value) + r'\b', re.IGNORECASE)
    return regex.sub(repl=entity_type_name, string=sentence, count=1)
