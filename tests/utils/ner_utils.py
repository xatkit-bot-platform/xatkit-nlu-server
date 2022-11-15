from xatkitnlu.core.nlp_configuration import NlpConfiguration


def check_ner_result(method, configuration: NlpConfiguration, sentences: list[str], expected_result: tuple[str, str, dict]):
    for sentence in sentences:
        ner_result: tuple[str, str, dict] = method(sentence, configuration)
        print('Expected NER RESULT: ', expected_result)
        print('Actual NER RESULT:   ', ner_result)
        assert ner_result == expected_result