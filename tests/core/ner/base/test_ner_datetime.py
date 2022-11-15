from datetime import datetime
from zoneinfo import ZoneInfo

from dateutil.relativedelta import relativedelta
import uuid

from tests.utils.intents_and_entities import intent_birthday_en, intent_greetings_en
from tests.utils.ner_utils import check_ner_result
from xatkitnlu.core.ner.base.datetime import ner_datetime
from xatkitnlu.core.prediction import predict
from xatkitnlu.core.training import train
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, PredictResult


def test_ner_datetime():

    configuration = NlpConfiguration(get_incomplete_dates=True, timezone='Europe/Madrid')

    # English

    configuration.country = 'en'
    timezone = ZoneInfo(configuration.timezone)

    sentences_en = ['My birthday is in May 4th']
    now = datetime.now(timezone).replace(microsecond=0)
    expected_datetime = datetime(year=now.year, month=5, day=4, hour=0, minute=0, second=0, tzinfo=timezone).isoformat()

    expected_result_en = ('My birthday is ' + expected_datetime,
                           expected_datetime,
                           {'year': False, 'month': True, 'day': True, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    sentences_en = ['My birthday is tomorrow']
    now = datetime.now(timezone)
    expected_datetime = (now + relativedelta(days=1)).replace(microsecond=0).isoformat()
    expected_result_en = ('My birthday is ' + expected_datetime,
                          expected_datetime,
                          {'year': True, 'month': True, 'day': True, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    sentences_en = ['I am going to move in 2 years and three months']
    now = datetime.now(timezone)
    expected_datetime = (now + relativedelta(years=2, months=3)).replace(microsecond=0).isoformat()
    expected_result_en = ('I am going to move ' + expected_datetime,
                          expected_datetime,
                          {'year': True, 'month': True, 'day': False, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)


    # Catalan

    configuration.country = 'ca'
    timezone = ZoneInfo(configuration.timezone)

    sentences_en = ['El meu aniversari és el 4 de maig']
    now = datetime.now(timezone).replace(microsecond=0)
    expected_datetime = datetime(year=now.year, month=5, day=4, hour=0, minute=0, second=0, tzinfo=timezone).isoformat()

    expected_result_en = ('El meu aniversari és el ' + expected_datetime,
                          expected_datetime,
                          {'year': False, 'month': True, 'day': True, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    sentences_en = ['El meu aniversari és demà']
    now = datetime.now(timezone)
    expected_datetime = (now + relativedelta(days=1)).replace(microsecond=0).isoformat()
    expected_result_en = ('El meu aniversari és ' + expected_datetime,
                          expected_datetime,
                          {'year': True, 'month': True, 'day': True, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    # Catalan seems to fail when using plural words ('anys', 'mesos')
    sentences_en = ['Em mudaré en 1 any i un mes']
    now = datetime.now(timezone)
    expected_datetime = (now + relativedelta(years=1, months=1)).replace(microsecond=0).isoformat()
    expected_result_en = ('Em mudaré ' + expected_datetime,
                          expected_datetime,
                          {'year': True, 'month': True, 'day': False, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    # Spanish

    configuration.country = 'es'
    timezone = ZoneInfo(configuration.timezone)

    # Mi cumpleaños es el 4 de mayo => Gets confused: 'Mi' = 'wednesday'
    sentences_en = ['El cumpleaños es el 4 de mayo']
    now = datetime.now(timezone).replace(microsecond=0)
    expected_datetime = datetime(year=now.year, month=5, day=4, hour=0, minute=0, second=0, tzinfo=timezone).isoformat()

    expected_result_en = ('El cumpleaños es el ' + expected_datetime,
                           expected_datetime,
                           {'year': False, 'month': True, 'day': True, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    sentences_en = ['El cumpleaños es mañana']
    now = datetime.now(timezone)
    expected_datetime = (now + relativedelta(days=1)).replace(microsecond=0).isoformat()
    expected_result_en = ('El cumpleaños es ' + expected_datetime,
                          expected_datetime,
                          {'year': True, 'month': True, 'day': True, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)

    sentences_en = ['Me mudaré en 2 años y 3 meses']
    now = datetime.now(timezone)
    expected_datetime = (now + relativedelta(years=2, months=3)).replace(microsecond=0).isoformat()
    expected_result_en = ('Me mudaré ' + expected_datetime,
                          expected_datetime,
                          {'year': True, 'month': True, 'day': False, 'hour': False, 'minute': False, 'second': False})
    check_ner_result(ner_datetime, configuration, sentences_en, expected_result_en)


def test_prediction_with_ner():
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context: NLUContext = NLUContext('context')
    bot.add_context(context)

    context.add_intent(intent_birthday_en)
    context.add_intent(intent_greetings_en)
    bot.configuration.use_ner_in_prediction = True
    bot.configuration.country = 'en'
    train(bot)

    sentence_to_predict = 'My birthday is in May 4th'
    prediction: PredictResult = predict(context, sentence_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {sentence_to_predict} is {scores}')
    assert (prediction.get_classification(intent_birthday_en).score > prediction.get_classification(intent_greetings_en).score)
    assert (prediction.get_classification(intent_birthday_en).matched_utterance == 'My birthday is @SYS.DATE-TIME')
    assert (len(prediction.get_classification(intent_birthday_en).matched_params) == 1)
    assert (prediction.get_classification(intent_birthday_en).matched_params[0].name == 'birthday')

    timezone = ZoneInfo(bot.configuration.timezone)
    now = datetime.now(timezone)
    expected_datetime = datetime(year=now.year, month=5, day=4, hour=0, minute=0, second=0, tzinfo=timezone)
    predicted_datetime = datetime.fromisoformat(prediction.get_classification(intent_birthday_en).matched_params[0].value)
    assert expected_datetime == predicted_datetime
    
    expected_datetime_info = {
        'year': False,
        'month': True,
        'day': True,
        'hour': False,
        'minute': False,
        'second': False
    }
    assert (prediction.get_classification(intent_birthday_en).matched_params[0].info == expected_datetime_info)
