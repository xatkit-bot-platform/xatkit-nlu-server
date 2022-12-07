from xatkitnlu.core.ner.base.any import ner_any
from xatkitnlu.core.ner.base.base_entities import BaseEntityType, ordered_base_entities
from xatkitnlu.core.ner.base.datetime import ner_datetime
from xatkitnlu.core.ner.base.number import ner_number
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext, EntityReference, MatchedParam, BaseEntity
from xatkitnlu.utils.utils import value_in_sentence, replace_value_in_sentence, replace_temp_value_in_sentence, \
    find_first_temp


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
        all_entity_values: dict[str, tuple[list[EntityReference], str]] = intent.get_custom_entity_values_dict(preprocessed_values)
        temps: dict[str, tuple[list[EntityReference], str]] = {}
        temp_template = r'/temp{}/'
        temp_count = 1
        for value, (entity_refs, entry_value) in sorted(all_entity_values.items(), key=lambda x: (len(x[0]), x[0].casefold()), reverse=True):
            # TODO: This approach doesn't allow 2 repetitions of the same value in a sentence
            # entry_value are all entry values of the entity
            # value can be an entry value (i.e. value == entry_value)
            # or a synonym of an entry value (i.e. value is a synonym of entry_value)
            # value can be preprocessed
            if temp_count > len(intent.entity_parameters):
                # When we detect len(intent.entity_parameters) values, we stop
                break
            if value_in_sentence(value, ner_sentence):
                temp_n = temp_template.format(temp_count)
                temp_count += 1
                ner_sentence = replace_value_in_sentence(ner_sentence, value, temp_n)
                temps[temp_n] = (entity_refs, entry_value)

        entity_refs_done: list[EntityReference] = []
        while len(temps) > 0:
            # We get the temp that appears first in the sentence,
            # and replace it by the 1st entity reference, in order of declaration in the bot definition
            temp = find_first_temp(ner_sentence)
            (entity_refs, value) = temps[temp]
            entity_ref = next(
                (e for e in entity_refs if e not in entity_refs_done),
                None
            )
            entity_refs_done.append(entity_ref)
            ner_sentence = replace_temp_value_in_sentence(ner_sentence, temp, entity_ref.entity.name.upper())
            intent_matches.append(MatchedParam(entity_ref.name, value, {}))
            temps.pop(temp)

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
