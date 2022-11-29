import uuid

from tests.utils.intents_and_entities import intent_temperature_en, intent_greetings_en, entity_number
from tests.utils.ner_utils import check_ner_result
from xatkitnlu.core.ner.base.number import ner_number
from xatkitnlu.core.prediction import predict
from xatkitnlu.core.training import train
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, PredictResult, IntentReference


def test_ner_number():

    configuration = NlpConfiguration()

    # English

    configuration.country = 'en'
    sentences1_en = [
        'the temperature is 3.5 and it is cold',
        'the temperature is +3.5 and it is cold',
        'the temperature is +3,5 and it is cold',
        'the temperature is three point five and it is cold',
        'the temperature is plus three point five and it is cold'
    ]
    expected_result1_en = ('the temperature is 3.5 and it is cold', '3.5', {})
    check_ner_result(ner_number, configuration, sentences1_en, expected_result1_en)

    sentences2_en = [
        'the temperature is -3.5 and it is cold',
        'the temperature is -3,5 and it is cold',
        'the temperature is minus three point five and it is cold'
    ]
    expected_result2_en = ('the temperature is -3.5 and it is cold', '-3.5', {})
    check_ner_result(ner_number, configuration, sentences2_en, expected_result2_en)

    # Catalan

    configuration.country = 'ca'
    sentences1_ca = [
        'la temperatura és de 3.5 i fa fred',
        'la temperatura és de +3.5 i fa fred',
        'la temperatura és de +3,5 i fa fred',
        'la temperatura és de tres coma cinc i fa fred',
        'la temperatura és de més tres coma cinc i fa fred'
    ]
    expected_result1_ca = ('la temperatura és de 3.5 i fa fred', '3.5', {})
    check_ner_result(ner_number, configuration, sentences1_ca, expected_result1_ca)

    sentences2_ca = [
        'la temperatura és de -3.5 i fa fred',
        'la temperatura és de -3,5 i fa fred',
        'la temperatura és de menys tres coma cinc i fa fred'
    ]
    expected_result2_ca = ('la temperatura és de -3.5 i fa fred', '-3.5', {})
    check_ner_result(ner_number, configuration, sentences2_ca, expected_result2_ca)

    # Spanish

    configuration.country = 'es'
    sentences1_es = [
        'la temperatura es de 3.5 y hace frío',
        'la temperatura es de +3.5 y hace frío',
        'la temperatura es de +3,5 y hace frío',
        'la temperatura es de tres coma cinco y hace frío',
        'la temperatura es de mas tres coma cinco y hace frío',
    ]
    expected_result1_es = ('la temperatura es de 3.5 y hace frío', '3.5', {})
    check_ner_result(ner_number, configuration, sentences1_es, expected_result1_es)

    sentences2_es = [
        'la temperatura es de -3.5 y hace frío',
        'la temperatura es de -3,5 y hace frío',
        'la temperatura es de menos tres coma cinco y hace frío'
    ]
    expected_result2_es = ('la temperatura es de -3.5 y hace frío', '-3.5', {})
    check_ner_result(ner_number, configuration, sentences2_es, expected_result2_es)


def test_prediction_with_ner():
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context: NLUContext = NLUContext('context')

    bot.add_entity(entity_number)
    bot.add_intent(intent_temperature_en)
    bot.add_intent(intent_greetings_en)
    context.add_intent_ref(IntentReference(intent_temperature_en.name, intent_temperature_en))
    context.add_intent_ref(IntentReference(intent_greetings_en.name, intent_greetings_en))
    bot.add_context(context)
    bot.configuration.use_ner_in_prediction = True
    bot.configuration.country = 'en'

    train(bot)

    sentence_to_predict = 'the temperature is minus three point five and it is cold'
    prediction: PredictResult = predict(context, sentence_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {sentence_to_predict} is {scores}')
    assert (prediction.get_classification(intent_temperature_en).score > prediction.get_classification(intent_greetings_en).score)
    # sentence is preprocessed, so it doesn't match with the original sentence
    # assert (prediction.get_classification(intent_temperature_en).matched_utterance == 'the temperature is @SYS.NUMBER and it is cold')
    assert (len(prediction.get_classification(intent_temperature_en).matched_params) == 1)
    assert (prediction.get_classification(intent_temperature_en).matched_params[0].name == 'temperature')
    assert (prediction.get_classification(intent_temperature_en).matched_params[0].value == '-3.5')
    assert (prediction.get_classification(intent_temperature_en).matched_params[0].info == {})
