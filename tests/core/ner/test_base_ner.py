from datetime import datetime
from dateutil.relativedelta import relativedelta
import uuid

from xatkitnlu.core.ner.base.base_entities import BaseEntityType
from xatkitnlu.core.prediction import predict
from xatkitnlu.core.training import train
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, Intent, EntityReference, PredictResult, BaseEntity


entity_number: BaseEntity = BaseEntity(BaseEntityType.NUMBER)
entity_date: BaseEntity = BaseEntity(BaseEntityType.DATETIME)

# English

intent_temperature_en: Intent = Intent('intent_temperature_en',
        ['the temperature outside is NUMBER and it is so cold', 'it is NUMBER degrees outside'])
intent_temperature_en.add_entity_parameter(EntityReference('temperature', 'NUMBER', entity_number))

intent_greetings_en: Intent = Intent('intent_greetings_en',
        ['hello', 'how are you'])

intent_birthday_en: Intent = Intent('intent_birthday_en',
        ['My birthday is DATE', 'DATE is my birthday'])
intent_birthday_en.add_entity_parameter(EntityReference('birthday', 'DATE', entity_date))

# Catalan

intent_temperature_ca: Intent = Intent('intent_temperature_ca',
        ['la temperatura fora és de NUMBER i fa molt fred', 'fora fa NUMBER graus'])
intent_temperature_ca.add_entity_parameter(EntityReference('temperature', 'NUMBER', entity_number))

intent_greetings_ca: Intent = Intent('intent_greetings_ca',
        ['hola', 'com estàs'])

# Spanish

intent_temperature_es: Intent = Intent('intent_temperature_es',
        ['la temperatura fuera es de NUMBER y hace mucho frío', 'fuera hace NUMBER grados'])
intent_temperature_es.add_entity_parameter(EntityReference('temperature', 'NUMBER', entity_number))

intent_greetings_es: Intent = Intent('intent_greetings_es',
        ['hola', 'como estás'])


def test_ner_number():

    # English Test

    bot_en: Bot = Bot(uuid.uuid4(), 'test bot_en', NlpConfiguration())
    context_en: NLUContext = NLUContext('context_en')
    bot_en.add_context(context_en)

    context_en.add_intent(intent_temperature_en)
    context_en.add_intent(intent_greetings_en)
    bot_en.configuration.use_ner_in_prediction = True
    bot_en.configuration.country = 'en'
    train(bot_en)

    values = [('3.5', '3.5'),
              ('3.5', '+3.5'),
              ('3.5', 'three point five'),
              ('3.5', 'plus three point five'),
              ('3.5', '+3,5'),
              ('-3.5', '-3.5'),
              ('-3.5', '-3,5'),
              ('-3.5', 'minus three point five')]
    for value, raw_value in values:
        sentence_to_predict = 'the temperature is ' + raw_value + ' and it is cold'
        prediction: PredictResult = predict(context_en, sentence_to_predict, bot_en.configuration)
        scores: list[float] = [classification.score for classification in prediction.classifications]
        print(f'Prediction for {sentence_to_predict} is {scores}')
        assert (prediction.get_classification(intent_temperature_en).score > prediction.get_classification(intent_greetings_en).score)
        assert (prediction.get_classification(intent_temperature_en).matched_utterance == 'the temperature is @SYS.NUMBER and it is cold')
        assert (len(prediction.get_classification(intent_temperature_en).matched_params) == 1)
        assert (prediction.get_classification(intent_temperature_en).matched_params[0].name == 'temperature')
        assert (prediction.get_classification(intent_temperature_en).matched_params[0].value == value)

    # Catalan Test

    bot_ca: Bot = Bot(uuid.uuid4(), 'test bot_ca', NlpConfiguration())
    context_ca: NLUContext = NLUContext('context_ca')
    bot_ca.add_context(context_ca)

    context_ca.add_intent(intent_temperature_ca)
    context_ca.add_intent(intent_greetings_ca)
    bot_ca.configuration.use_ner_in_prediction = True
    bot_ca.configuration.country = 'ca'
    train(bot_ca)

    values = [('3.5', '3.5'),
              ('3.5', '+3.5'),
              ('3.5', 'tres coma cinc'),
              ('3.5', 'més tres coma cinc'),
              ('3.5', '+3,5'),
              ('-3.5', '-3.5'),
              ('-3.5', '-3,5'),
              ('-3.5', 'menys tres coma cinc')]
    for value, raw_value in values:
        sentence_to_predict = 'la temperatura és de ' + raw_value + ' i fa fred'
        prediction: PredictResult = predict(context_ca, sentence_to_predict, bot_ca.configuration)
        scores: list[float] = [classification.score for classification in prediction.classifications]
        print(f'Prediction for {sentence_to_predict} is {scores}')
        assert (prediction.get_classification(intent_temperature_ca).score > prediction.get_classification(intent_greetings_ca).score)
        assert (prediction.get_classification(intent_temperature_ca).matched_utterance == 'la temperatura és de @SYS.NUMBER i fa fred')
        assert (len(prediction.get_classification(intent_temperature_ca).matched_params) == 1)
        assert (prediction.get_classification(intent_temperature_ca).matched_params[0].name == 'temperature')
        assert (prediction.get_classification(intent_temperature_ca).matched_params[0].value == value)

    # Spanish Test

    bot_es: Bot = Bot(uuid.uuid4(), 'test bot_es', NlpConfiguration())
    context_es: NLUContext = NLUContext('context_es')
    bot_es.add_context(context_es)

    context_es.add_intent(intent_temperature_es)
    context_es.add_intent(intent_greetings_es)
    bot_es.configuration.use_ner_in_prediction = True
    bot_es.configuration.country = 'es'
    train(bot_es)

    values = [('3.5', '3.5'),
              ('3.5', '+3.5'),
              ('3.5', 'tres coma cinco'),
              ('3.5', 'mas tres coma cinco'),
              ('3.5', '+3,5'),
              ('-3.5', '-3.5'),
              ('-3.5', '-3,5'),
              ('-3.5', 'menos tres coma cinco')]
    for value, raw_value in values:
        sentence_to_predict = 'la temperatura es de ' + raw_value + ' y hace frío'
        prediction: PredictResult = predict(context_es, sentence_to_predict, bot_es.configuration)
        scores: list[float] = [classification.score for classification in prediction.classifications]
        print(f'Prediction for {sentence_to_predict} is {scores}')
        assert (prediction.get_classification(intent_temperature_es).score > prediction.get_classification(intent_greetings_es).score)
        assert (prediction.get_classification(intent_temperature_es).matched_utterance == 'la temperatura es de @SYS.NUMBER y hace frío')
        assert (len(prediction.get_classification(intent_temperature_es).matched_params) == 1)
        assert (prediction.get_classification(intent_temperature_es).matched_params[0].name == 'temperature')
        assert (prediction.get_classification(intent_temperature_es).matched_params[0].value == value)


def test_ner_date():
    bot_en: Bot = Bot(uuid.uuid4(), 'test bot_en', NlpConfiguration())
    context_en: NLUContext = NLUContext('context_en')
    bot_en.add_context(context_en)

    context_en.add_intent(intent_birthday_en)
    context_en.add_intent(intent_greetings_en)
    bot_en.configuration.use_ner_in_prediction = True
    bot_en.configuration.country = 'en'
    train(bot_en)

    now = datetime.now()
    values = [(now, 'today'),
              (now + relativedelta(days=1), 'tomorrow'),
              (now + relativedelta(months=1), 'next month'),
              (now + relativedelta(months=4), 'in 4 months'),
              (datetime(year=now.year, month=5, day=4), 'in May 4th'),
              (datetime(year=now.year, month=5, day=4, hour=15), 'in May 4th at 3pm')]

    for value, raw_value in values:
        sentence_to_predict = 'My birthday is ' + raw_value
        prediction: PredictResult = predict(context_en, sentence_to_predict, bot_en.configuration)
        scores: list[float] = [classification.score for classification in prediction.classifications]
        print(f'Prediction for {sentence_to_predict} is {scores}')

        assert (prediction.get_classification(intent_birthday_en).score > prediction.get_classification(intent_greetings_en).score)
        assert (prediction.get_classification(intent_birthday_en).matched_utterance == 'My birthday is @SYS.DATE-TIME')
        assert (len(prediction.get_classification(intent_birthday_en).matched_params) == 1)
        assert (prediction.get_classification(intent_birthday_en).matched_params[0].name == 'birthday')

        date = datetime.fromisoformat(prediction.get_classification(intent_birthday_en).matched_params[0].value)
        assert date.year == value.year
        assert date.month == value.month
        assert date.day == value.day
        assert date.hour == value.hour
        assert date.min == value.min
        assert date.second == value.second
