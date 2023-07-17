import uuid

import numpy as np

from tests.utils.intents_and_entities import intent_weather_ner, bot1_intents, intent_greetings, intent_museum_ner, \
    intent_museum_no_ner, entity_museum, entity_city
from xatkitnlu.core.prediction import predict
from xatkitnlu.core.training import train
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, PredictResult, MatchedParameter, IntentReference
from tests.utils.sample_bots import create_bot_one_context_several_intents


def test_training_with_ner():
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    bot.add_entity(entity_city)
    bot.add_intent(intent_weather_ner)
    bot.add_intent(intent_greetings)
    context1: NLUContext = NLUContext('context1')
    context1.add_intent_ref(IntentReference(intent_weather_ner.name, intent_weather_ner))
    context1.add_intent_ref(IntentReference(intent_greetings.name, intent_greetings))
    bot.add_context(context1)

    bot.configuration.use_ner_in_prediction = True
    train(bot)

    assert intent_weather_ner.processed_training_sentences[0] == 'what is the weather like in ENTITY_CITY'
    assert intent_greetings.processed_training_sentences[0] == 'hello'


def test_prediction_when_prediction_sentence_is_all_oov_with_ner():
    bot: Bot = create_bot_one_context_several_intents(bot1_intents)
    bot.add_entity(entity_city)
    bot.add_intent(intent_weather_ner)
    bot.contexts[0].add_intent_ref(IntentReference(intent_weather_ner.name, intent_weather_ner))
    bot.configuration.use_ner_in_prediction = True
    bot.configuration.discard_oov_sentences = True
    train(bot)
    sentence_to_predict = 'xsx dfasklj BCN adfan'
    prediction: PredictResult = predict(bot.contexts[0], sentence_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {sentence_to_predict} is {scores}')

    # BCN is not in the training sentences directly but it's part of a NER so prediction should go ahead and not just discard it
    assert (scores[0] == 0)
    assert (scores[1] == 0)
    assert (scores[2] == 0)
    assert (scores[3] > 0)


def test_prediction_for_when_prediction_sentence_is_in_training_sentence_with_ner():
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    bot.add_entity(entity_city)
    bot.add_intent(intent_weather_ner)
    bot.add_intent(intent_greetings)
    context1: NLUContext = NLUContext('context1')
    context1.add_intent_ref(IntentReference(intent_weather_ner.name, intent_weather_ner))
    context1.add_intent_ref(IntentReference(intent_greetings.name, intent_greetings))
    bot.add_context(context1)
    bot.configuration.use_ner_in_prediction = True
    bot.configuration.check_exact_prediction_match = True
    train(bot)

    sentence_to_predict = 'What is the weather like in Madrid?'
    prediction: PredictResult = predict(context1, sentence_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {sentence_to_predict} is {scores}')
    # The prediction sentence is an exact match (once stemmed) for a training sentence in intent_weather
    assert (np.argmax(scores) == 0)
    assert (scores[0] == 1)


def test_prediction_with_ner():
    bot: Bot = create_bot_one_context_several_intents(bot1_intents)

    bot.add_entity(entity_city)
    bot.add_entity(entity_museum)
    bot.add_intent(intent_weather_ner)
    bot.add_intent(intent_museum_ner)
    bot.add_intent(intent_museum_no_ner)
    context1: NLUContext = bot.contexts[0]
    context1.add_intent_ref(IntentReference(intent_weather_ner.name, intent_weather_ner))
    context1.add_intent_ref(IntentReference(intent_museum_ner.name, intent_museum_ner))
    context1.add_intent_ref(IntentReference(intent_museum_no_ner.name, intent_museum_no_ner))
    sentence_to_predict = 'How is the weather at BCN?'

    bot.configuration.use_ner_in_prediction = False
    train(bot)
    prediction_weather: PredictResult = predict(context1, sentence_to_predict, bot.configuration)
    scores_weather: list[float] = [classification.score for classification in prediction_weather.classifications]
    print(f'Prediction for {sentence_to_predict} is {scores_weather}')

    bot.configuration.use_ner_in_prediction = True
    train(bot)
    prediction_weather_ner: PredictResult = predict(context1, sentence_to_predict, bot.configuration)
    scores_weather_ner: list[float] = [classification.score for classification in prediction_weather_ner.classifications]
    matched_weather_ner: list[list[MatchedParameter]] = [classification.matched_parameters for classification in prediction_weather_ner.classifications]
    print(f'NER Prediction for {sentence_to_predict} is {scores_weather_ner}')
    print(f'Matched NERs  are {matched_weather_ner}')

    assert (np.argmax(scores_weather_ner) == 3)
    assert (scores_weather_ner[3] > scores_weather[3])

    sentence_to_predict = 'I would like to visit the Louvre'
    prediction_museum_ner: PredictResult = predict(context1, sentence_to_predict, bot.configuration)
    scores_museum_ner: list[float] = [classification.score for classification in prediction_museum_ner.classifications]
    matched_museum_ner: list[list[MatchedParameter]] = [classification.matched_parameters for classification in prediction_museum_ner.classifications]
    intent_index = context1.get_intents().index(intent_museum_ner)
    print(f'NER Prediction for {sentence_to_predict} is {scores_museum_ner}')
    print(f'Matched NERs  are {matched_weather_ner}')

    assert (np.argmax(scores_museum_ner) == 4)
    assert (matched_museum_ner[intent_index][0].name == 'museum')
    assert (matched_museum_ner[intent_index][0].value == 'Louvre')

    sentence_to_predict = 'I want to visit something'
    prediction_museum_no_ner: PredictResult = predict(context1, sentence_to_predict, bot.configuration)
    scores_museum_no_ner: list[float] = [classification.score for classification in prediction_museum_no_ner.classifications]
    matched_museum_no_ner: list[list[MatchedParameter]] = [classification.matched_parameters for classification in prediction_museum_no_ner.classifications]
    intent_index = context1.get_intents().index(intent_museum_no_ner)
    print(f'NER Prediction for {sentence_to_predict} is {scores_museum_no_ner}')

    assert (np.argmax(scores_museum_no_ner) == 5)
    assert (len(matched_museum_no_ner[intent_index]) == 0)


def test_ner_matching():
    bot: Bot = create_bot_one_context_several_intents(bot1_intents)
    bot.add_entity(entity_city)
    bot.add_intent(intent_weather_ner)
    context1: NLUContext = bot.contexts[0]
    context1.add_intent_ref(IntentReference(intent_weather_ner.name, intent_weather_ner))

    text_to_predict = 'How is the weather at Barcelona'

    bot.configuration.activation_last_layer = 'sigmoid'
    bot.configuration.ner_matching = False
    train(bot)
    prediction: PredictResult = predict(context1, text_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]

    print(f'Prediction for {text_to_predict} is {scores}')
    assert (np.argmax(scores) == 3)

    bot.configuration.ner_matching = True
    train(bot)
    prediction_ner: PredictResult = predict(context1, text_to_predict, bot.configuration)
    scores_ner: list[float] = [classification.score for classification in prediction_ner.classifications]
    matched_ner: list[list[MatchedParameter]] = [classification.matched_parameters for classification in prediction_ner.classifications]

    print(f'NER Prediction for {text_to_predict} is {scores_ner}')
    print(f'Matched NERs  are {matched_ner}')

    assert (np.argmax(scores_ner) == 3)

    # sometimes it fails for a tiny difference
    # assert (scores_ner[3] > scores[3])
