import numpy
from core.prediction import predict
from core.training import train, stem_training_sentence
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext
from tests.utils.sample_bots import create_bot_one_context_several_intents, \
    create_bot_one_context_several_intent_with_one_custom_city_intent_with_ner

def test_prediction_with_ner():
    bot: Bot = create_bot_one_context_several_intent_with_one_custom_city_intent_with_ner(
        {'intent1' : ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2' : ['hello', 'how are you', 'greetings'],
         'intent3' : ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']})
    context1: NLUContext = bot.contexts[0]
    text_to_predict = 'How is the weather at mycity'

    bot.configuration.ner_matching = False
    train(bot)
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'Prediction for {text_to_predict} is {prediction}')
    assert (prediction.argmax() == 3)

    bot.configuration.ner_matching = True
    train(bot)
    prediction_ner: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'NER Prediction for {text_to_predict} is {prediction_ner}')
    assert (prediction_ner.argmax() == 3)
    assert (prediction_ner.tolist()[3] > prediction.tolist()[3])