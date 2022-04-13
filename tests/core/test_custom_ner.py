import uuid

import numpy
from core.prediction import predict
from core.training import train
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext, CustomEntity, CustomEntityEntry, Intent, EntityReference
from tests.utils.sample_bots import create_bot_one_context_several_intents, \
    create_bot_one_context_several_intent_with_one_custom_city_intent_with_ner


def test_training_with_ner():
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)

    entity: CustomEntity = CustomEntity('cityentity',
                                        [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid')])
    intent: Intent = Intent('intent_name',
                            ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny?'])
    intent.add_entity_parameter(EntityReference('city', 'mycity', entity))

    context1.add_intent(intent)
    intent_no_ner: Intent = Intent('Greetings',
                                   ['hello', 'how are you'])
    context1.add_intent(intent_no_ner)
    bot.configuration.use_ner_in_prediction = True
    train(bot)

    assert intent.processed_training_sentences[0] == 'what is the weather like in CITYENTITY'
    assert intent_no_ner.processed_training_sentences[0] == 'hello'


def test_prediction_when_prediction_sentence_is_all_oov_with_ner():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2': ['hello', 'how are you', 'greetings'],
         'intent3': ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']})

    entity: CustomEntity = CustomEntity('cityentity',
                                        [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid')])
    intent_ner: Intent = Intent('intent_name',
                                ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny?'])
    intent_ner.add_entity_parameter(EntityReference('city', 'mycity', entity))


    bot.contexts[0].add_intent(intent_ner)

    bot.configuration.discard_oov_sentences = True
    train(bot)
    sentence_to_predict = 'xsx dfasklj BCN adfan'
    prediction: numpy.ndarray = predict(bot.contexts[0], sentence_to_predict, bot.configuration)[0]
    print(f'Prediction for {sentence_to_predict} is {prediction}')

    # BCN is not in the training sentences directly but it's part of a NER so prediction should go ahead and not just discard it
    assert (max(prediction.tolist()) > 0)


def test_prediction_for_when_prediction_sentence_is_in_training_sentence_with_ner():
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)

    entity: CustomEntity = CustomEntity('cityentity',
                                        [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid')])
    intent: Intent = Intent('intent_name',
                            ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny?'])
    intent.add_entity_parameter(EntityReference('city', 'mycity', entity))

    context1.add_intent(intent)
    intent_no_ner: Intent = Intent('Greetings',
                                   ['hello', 'how are you'])
    context1.add_intent(intent_no_ner)
    bot.configuration.use_ner_in_prediction = True
    train(bot)

    sentence_to_predict = 'What is the weather like in Madrid?'
    prediction: numpy.ndarray = predict(context1, sentence_to_predict, bot.configuration)[0]
    print(f'Prediction for {sentence_to_predict} is {prediction}')
    # The prediction sentence is an exact match (once stemmed) for a training sentence in intent1
    assert (prediction.argmax() == 0)
    assert (prediction.tolist()[0] == 1)


def test_prediction_with_ner():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2': ['hello', 'how are you', 'greetings'],
         'intent3': ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']})

    entity_city: CustomEntity = CustomEntity('cityentity',
                                             [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid')])
    intent_city_ner: Intent = Intent('intent_city_ner',
                                     ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny in mycity?'])
    intent_city_ner.add_entity_parameter(EntityReference('city', 'mycity', entity_city))


    entity_museum: CustomEntity = CustomEntity('museumentity',
                                               [CustomEntityEntry('Louvre', ['Louv, Louvre in Paris']),
                                                CustomEntityEntry('Gaudí', ['Gaudí'])])
    intent_museum_ner: Intent = Intent('intent_museu',
                                       ['I want to visit the mymuseum', 'is the mymuseum open for a visit',
                                        'what about visiting the mymuseum?'])
    intent_museum_ner.add_entity_parameter(EntityReference('museum', 'mymuseum', entity_museum))

    intent_museum_no_ner: Intent = Intent('intent_museu_no_ner',
                                          ['I want to visit something interesting',
                                           'is something cool open for a visit?',
                                           'what ideas do we have for tomorrow?'])

    context1: NLUContext = bot.contexts[0]
    context1.add_intent(intent_city_ner)
    context1.add_intent(intent_museum_ner)
    context1.add_intent(intent_museum_no_ner)
    sentence_to_predict = 'How is the weather at BCN?'

    bot.configuration.use_ner_in_prediction = False
    train(bot)
    prediction: numpy.ndarray = predict(context1, sentence_to_predict, bot.configuration)[0]
    print(f'Prediction for {sentence_to_predict} is {prediction}')

    bot.configuration.use_ner_in_prediction = True
    train(bot)
    prediction_city_ner: numpy.ndarray
    matched_city_ner: dict[str, dict[str, str]] = {}
    prediction_city_ner, matched_city_ner = predict(context1, sentence_to_predict, bot.configuration)

    print(f'NER Prediction for {sentence_to_predict} is {prediction_city_ner}')
    print(f'Matched NERs  are {matched_city_ner}')

    assert (prediction_city_ner.argmax() == 3)
    assert (prediction_city_ner.tolist()[3] > prediction.tolist()[3])

    sentence_to_predict = 'I want to visit something'
    prediction_museum: numpy.ndarray
    matched_museum_ner: dict[str, dict[str, str]]
    prediction_museum, matched_museum_ner = predict(context1, sentence_to_predict, bot.configuration)

    assert (prediction_museum.argmax() == 5)
    assert (len(matched_museum_ner) == 0)

    sentence_to_predict = 'I would like to visit the Louvre'
    prediction_museum, matched_museum_ner = predict(context1, sentence_to_predict, bot.configuration)
    assert (prediction_museum.argmax() == 4)
    assert (matched_museum_ner[intent_museum_ner]['museum'] == 'Louvre')


def test_ner_matching():
    bot: Bot = create_bot_one_context_several_intent_with_one_custom_city_intent_with_ner(
        {'intent1': ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2': ['hello', 'how is he', 'greetings'],
         'intent3': ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']})
    context1: NLUContext = bot.contexts[0]
    text_to_predict = 'How is the weather at Barcelona'

    bot.configuration.activation_last_layer = 'sigmoid'
    bot.configuration.ner_matching = False
    train(bot)
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)[0]
    print(f'Prediction for {text_to_predict} is {prediction}')
    assert (prediction.argmax() == 3)

    bot.configuration.ner_matching = True

    train(bot)
    prediction_ner: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)[0]
    print(f'NER Prediction for {text_to_predict} is {prediction_ner}')
    assert (prediction_ner.argmax() == 3)
    assert (prediction_ner.tolist()[3] > prediction.tolist()[3])
