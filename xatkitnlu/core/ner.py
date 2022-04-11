from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Intent, NLUContext


def ner_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> tuple[str, dict[Intent, dict[str,str]]]:
    # For now we only support custom NERs
    return ner_custom_matching(context, sentence, configuration)


def ner_custom_matching(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> tuple[str, dict[Intent, dict[str,str]]]:
    matched_ners: dict[Intent, dict[str, str]] = {}
    ner_sentence: str = sentence

    for intent in context.intents:
        intent_matches: dict[str, str] = {}
        # Check if sentence matches a value in the intent custom entities
        if intent.entity_parameters is not None:
            for entity_ref in intent.entity_parameters:
                param_name: str = entity_ref.name
                for entity_entry in entity_ref.entity.entries:

                    # We compare with the entry value
                    if entity_entry.value in sentence:
                        intent_matches[param_name] = entity_entry.value
                        ner_sentence = ner_sentence.replace(entity_entry.value, entity_ref.entity.name.upper())
                    # and also with the possible synonyms
                    if entity_entry.synonyms is not None:
                        for synonym in entity_entry.synonyms:
                            if synonym in sentence:
                                intent_matches[param_name] = entity_entry.value  # we return the entry value instead of the synonym
                                ner_sentence = ner_sentence.replace(synonym, entity_ref.entity.name.upper())

        if (intent_matches is not None) and (len(intent_matches) > 0):
            matched_ners[intent] = intent_matches

    return ner_sentence, matched_ners
