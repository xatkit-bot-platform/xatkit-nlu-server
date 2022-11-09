import re
from datetime import datetime
from dateparser.search import search_dates
from text_to_num import alpha2digit
from xatkitnlu.core.base_entities import BaseEntityType, ordered_base_entities
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext, EntityReference, CustomEntity, MatchedParam, BaseEntity
from zoneinfo import ZoneInfo

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
            # Match custom entities
            all_entity_values: dict[str, tuple[str, str, str]] = create_entity_values_dict(intent.entity_parameters)
            for value, (entry_value, param_name, entity_name) in sorted(all_entity_values.items(), reverse=True):
                # entry_value are all entry values of the entity
                # value can be an entry value (i.e. value == entry_value)
                # or a synonym of an entry value (i.e. value is a synonym of entry_value)
                if param_value_in_sentence(value, ner_sentence):
                    intent_matches.append(MatchedParam(param_name, entry_value))
                    ner_sentence = replace_fragment_in_sentence(ner_sentence, value, entity_name.upper())
            # Match base/system entities (after custom entities)
            intent_base_entities: dict[str, str] = get_base_entity_names(intent.entity_parameters)
            for base_entity_name in ordered_base_entities:
                # Base entities must be checked in a specific order
                try:
                    param_name = intent_base_entities[base_entity_name]
                    formatted_ner_sentence, formatted_frag = base_entity_ner(ner_sentence, base_entity_name, configuration)
                    if formatted_ner_sentence is not None and formatted_frag is not None:
                        intent_matches.append(MatchedParam(param_name, formatted_frag))
                        ner_sentence = replace_fragment_in_sentence(formatted_ner_sentence, formatted_frag, base_entity_name.upper())
                except:
                    pass
        result[intent] = (ner_sentence, intent_matches)
    return result


def base_entity_ner(sentence: str, entity_name: str, configuration: NlpConfiguration) -> tuple[str, str]:
    if entity_name == BaseEntityType.NUMBER:
        return ner_number(sentence, configuration)
    if entity_name == BaseEntityType.DATE:
        return ner_date(sentence, configuration)
    return None, None


def ner_number(sentence: str, configuration: NlpConfiguration) -> tuple[str, str]:
    # First, we parse any number in the sentence expressed in natural language (e.g. "five") to actual numbers
    sentence = alpha2digit(sentence, lang=configuration.country)

    # Negative/positive numbers with optional point/comma followed by more digits
    regex = re.compile(r'(\b|[-+])\d+\.?\d*([.,]\d+)?\b')
    search = regex.search(sentence)
    if search is None:
        return None, None
    matched_frag = search.group(0)
    formatted_frag = matched_frag.replace(',', '.').replace('+', '')
    sentence = sentence[:search.span(0)[0]] + formatted_frag + sentence[search.span(0)[1]:]
    return sentence, formatted_frag


def ner_date(sentence: str, configuration: NlpConfiguration) -> tuple[str, str]:
    timezone = ZoneInfo(configuration.timezone)
    now = datetime.now(tz=timezone)
    date_order = 'MDY'
    if configuration.country == 'es' or configuration.country == 'ca':
        date_order = 'DMY'

    relative_time_parser = ['relative-time']
    date_parsers = ['custom-formats', 'absolute-time']
    sett = {'RETURN_AS_TIMEZONE_AWARE': True,
            'RELATIVE_BASE': now,
            'DATE_ORDER': date_order,
            'PARSERS': date_parsers}
    date_matches: list[tuple[str, datetime]] = search_dates(sentence,
                                                            languages=[configuration.country],
                                                            settings=sett)
    if date_matches is not None:
        matched_frag, date = date_matches[0]
        formatted_frag = date.isoformat()
        sentence = replace_fragment_in_sentence(sentence, matched_frag, formatted_frag)
        return sentence, formatted_frag

    sett['PARSERS'] = relative_time_parser
    relative_time_matches: list[tuple[str, datetime]] = search_dates(sentence,
                                                                     languages=[configuration.country],
                                                                     settings=sett)
    if relative_time_matches is not None:
        matched_frag, date = relative_time_matches[0]
        formatted_frag = date.isoformat()
        sentence = replace_fragment_in_sentence(sentence, matched_frag, formatted_frag)
        return sentence, formatted_frag

    return None, None


def get_base_entity_names(entity_references: list[EntityReference]) -> dict[str, str]:
    base_entity_names: dict[str, str] = {}
    for entity_ref in entity_references:
        if isinstance(entity_ref.entity, BaseEntity) and entity_ref.entity.name in ordered_base_entities:
            base_entity_names[entity_ref.entity.name] = entity_ref.name
    return base_entity_names


def create_entity_values_dict(entity_references: list[EntityReference]) -> dict[str, tuple[str, str, str]]:
    all_entity_values: dict[str, tuple[str, str, str]] = {}
    for entity_ref in entity_references:
        if isinstance(entity_ref.entity, CustomEntity):
            param_name: str = entity_ref.name
            entity_name: str = entity_ref.entity.name
            for entity_entry in entity_ref.entity.entries:
                if entity_entry in all_entity_values.keys():
                    # ENTITY OVERLAPPING
                    pass
                all_entity_values[entity_entry.value] = (entity_entry.value, param_name, entity_name)
                if entity_entry.synonyms is not None:
                    for synonym in entity_entry.synonyms:
                        if synonym in all_entity_values.keys():
                            # ENTITY OVERLAPPING
                            pass
                        all_entity_values[synonym] = (entity_entry.value, param_name, entity_name)
    return all_entity_values


def param_value_in_sentence(param_value: str, sentence: str) -> bool:
    regex = re.compile(r'\b' + re.escape(param_value) + r'\b', re.IGNORECASE)
    return regex.search(sentence) is not None


def replace_fragment_in_sentence(sentence: str, frag: str, repl: str) -> str:
    if frag[0] == '-':
        # Necessary to replace negative numbers properly
        regex = re.compile(re.escape(frag) + r'\b', re.IGNORECASE)
    else:
        regex = re.compile(r'\b' + re.escape(frag) + r'\b', re.IGNORECASE)
    return regex.sub(repl=repl, string=sentence, count=1)
