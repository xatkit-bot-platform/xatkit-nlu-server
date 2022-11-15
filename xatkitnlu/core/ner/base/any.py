from xatkitnlu.core.nlp_configuration import NlpConfiguration


def ner_any(sentence: str, configuration: NlpConfiguration) -> tuple[str, str, dict]:
    return sentence, sentence, {}
