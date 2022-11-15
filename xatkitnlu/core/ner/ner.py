from xatkitnlu.core.ner.base.any import ner_any
from xatkitnlu.core.ner.base.base_entities import BaseEntityType, ordered_base_entities
from xatkitnlu.core.ner.base.datetime import ner_date
from xatkitnlu.core.ner.base.number import ner_number
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext, EntityReference, CustomEntity, MatchedParam, BaseEntity
from xatkitnlu.utils.utils import param_value_in_sentence, replace_fragment_in_sentence


def no_ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> dict[Intent, tuple[str, list[MatchedParam]]]:
    result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    for intent in context.intents:
        result[intent] = (sentence, [])
    return result


def ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> dict[Intent, tuple[str, list[MatchedParam]]]:
    result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    for intent in context.intents:
        intent_matches: list[MatchedParam] = []
        ner_sentence: str = sentence
        if intent.entity_parameters is not None:
            # Match custom entities
            all_entity_values: dict[str, tuple[str, str, str]] = get_custom_entity_values_dict(intent.entity_parameters)
            for value, (entry_value, param_name, entity_name) in sorted(all_entity_values.items(), reverse=True):
                # entry_value are all entry values of the entity
                # value can be an entry value (i.e. value == entry_value)
                # or a synonym of an entry value (i.e. value is a synonym of entry_value)
                if param_value_in_sentence(value, ner_sentence):
                    intent_matches.append(MatchedParam(param_name, entry_value, {}))
                    ner_sentence = replace_fragment_in_sentence(ner_sentence, value, entity_name.upper())
            # Match base/system entities (after custom entities)
            intent_base_entities: dict[str, str] = get_base_entity_names(intent.entity_parameters)
            for base_entity_name in ordered_base_entities:
                # Base entities must be checked in a specific order
                try:
                    param_name = intent_base_entities[base_entity_name]
                    formatted_ner_sentence, formatted_frag, param_info = base_entity_ner(ner_sentence, base_entity_name, configuration)
                    if formatted_ner_sentence is not None and formatted_frag is not None:
                        intent_matches.append(MatchedParam(param_name, formatted_frag, param_info))
                        ner_sentence = replace_fragment_in_sentence(formatted_ner_sentence, formatted_frag, base_entity_name.upper())
                except:
                    pass
        result[intent] = (ner_sentence, intent_matches)
    return result


def base_entity_ner(sentence: str, entity_name: str, configuration: NlpConfiguration) -> tuple[str, str, dict]:
    if entity_name == BaseEntityType.NUMBER:
        return ner_number(sentence, configuration)
    if entity_name == BaseEntityType.DATETIME:
        return ner_date(sentence, configuration)
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


def get_custom_entity_values_dict(entity_references: list[EntityReference]) -> dict[str, tuple[str, str, str]]:
    all_entity_values: dict[str, tuple[str, str, str]] = {}
    # {value/synonym: (value, param_name, entity_name)}
    for entity_ref in entity_references:
        if isinstance(entity_ref.entity, CustomEntity):
            param_name: str = entity_ref.name
            entity_name: str = entity_ref.entity.name
            for entity_entry in entity_ref.entity.entries:
                if entity_entry in all_entity_values.keys():
                    # TODO: ENTITY OVERLAPPING
                    pass
                all_entity_values[entity_entry.value] = (entity_entry.value, param_name, entity_name)
                if entity_entry.synonyms is not None:
                    for synonym in entity_entry.synonyms:
                        if synonym in all_entity_values.keys():
                            # TODO: ENTITY OVERLAPPING
                            pass
                        all_entity_values[synonym] = (entity_entry.value, param_name, entity_name)
    return all_entity_values
