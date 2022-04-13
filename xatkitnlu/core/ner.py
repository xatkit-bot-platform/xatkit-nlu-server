from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext


def ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> tuple[str, dict[Intent, dict[str, str]]]:
    # For now we only support custom NERs
    return ner_custom_matching(context, sentence, configuration)


def ner_custom_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> tuple[str, dict[Intent, dict[str, str]]]:
    matched_ners: dict[Intent, dict[str, str]] = {}
    ner_sentence: str = sentence

    for intent in context.intents:
        intent_matches: dict[str, str] = {}
        # Check if sentence matches a value in the intent custom entities
        if intent.entity_parameters is not None:
            for entity_ref in intent.entity_parameters:
                param_name: str = entity_ref.name
                found = False
                for entity_entry in entity_ref.entity.entries:

                    # We compare with the entry value
                    if param_value_in_sentence(entity_entry.value, sentence):
                        intent_matches[param_name] = entity_entry.value
                        ner_sentence = replace_param_value_with_entity_type_name_ner_sentence(ner_sentence, entity_entry.value, entity_ref.entity.name.upper())
                        found = True
                        break

                    # and also with the possible synonyms
                    elif entity_entry.synonyms is not None:
                        for synonym in entity_entry.synonyms:
                            if param_value_in_sentence(synonym, sentence):
                                intent_matches[param_name] = entity_entry.value  # we return the entry value instead of the synonym
                                ner_sentence = replace_param_value_with_entity_type_name_ner_sentence(ner_sentence, synonym, entity_ref.entity.name.upper())
                                found = True
                                break
                    if found:  #We can have two parameters referring to the same entity but each parameter can only match one value
                        break



        if (intent_matches is not None) and (len(intent_matches) > 0):
            matched_ners[intent] = intent_matches

    return ner_sentence, matched_ners


def param_value_in_sentence(param_value: str, sentence: str) -> bool:
    return (param_value in sentence) or (param_value.lower() in sentence) or (param_value.upper() in sentence)


def replace_param_value_with_entity_type_name_ner_sentence(sentence: str, found_param_value: str, entity_type_name: str) -> str:
    replaced_sentence: str = sentence.replace(found_param_value, entity_type_name, 1)
    if replaced_sentence == sentence:
        replaced_sentence = sentence.replace(found_param_value.lower(), entity_type_name, 1)
    if replaced_sentence == sentence:
        replaced_sentence = sentence.replace(found_param_value.upper(), entity_type_name, 1)
    return replaced_sentence
