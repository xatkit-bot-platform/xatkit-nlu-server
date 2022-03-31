import numpy
from core.prediction import predict
from core.training import train, stem_training_sentence
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext, Intent
from tests.utils.sample_bots import create_bot_one_context_several_intents


def test_predict():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2': ['hello', 'how are you', 'greetings'],
         'intent3': ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']})
    bot.configuration.input_max_num_tokens = 7
    bot.stemmer = False
    train(bot)
    context1: NLUContext = bot.contexts[0]

    text_to_predict = 'I love your dogs'
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'Prediction for {text_to_predict} is {prediction}')
    assert (prediction.argmax() == 0)

    text_to_predict = 'hello!'
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'Prediction for {text_to_predict} is {prediction}')
    assert (prediction.argmax() == 1)

    text_to_predict = 'can I have two more pizzas?'
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'Prediction for {text_to_predict} is {prediction}')
    assert(prediction.argmax() == 2)
    # print(prediction.max()) Max value in the numpy ndarray
    # print(prediction.argmax()) # position of the max value

    predictions: list[float] = prediction.tolist()
    print(predictions)


def test_predict_with_stemmer():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'How cute are your dogs', 'You seem to really! love my dog'],
         'intent2': ['hello', 'how are you', 'greetings'],
         'intent3': ['I prefer cats over dogs', 'I would prefer a cat', 'I love cats', 'I think cats are better']})
    bot.configuration.input_max_num_tokens = 10

    sentences_to_predict = ["He loves dogs", "hello!!", "I'm more of a cat lover"]
    context1: NLUContext = bot.contexts[0]

    bot.configuration.stemmer = False
    predictions_no_stemmer: list[numpy.ndarray] = []
    train(bot)
    for sentence in sentences_to_predict:
        predictions_no_stemmer.append(predict(context1, sentence, bot.configuration))

    bot.configuration.stemmer = True
    train(bot)
    predictions_stemmer: list[numpy.ndarray] = []
    for sentence in sentences_to_predict:
        predictions_stemmer.append(predict(context1, sentence, bot.configuration))

    print("Predictions without stemmer")
    print(predictions_no_stemmer)
    print("Predictions with stemmer")
    print(predictions_stemmer)

    # We check with the stemmer we get the good categories
    assert(predictions_stemmer[0].argmax() == 0)
    assert(predictions_stemmer[1].argmax() == 1)
    assert(predictions_stemmer[2].argmax() == 2)

    # We check the confidence is at least as good as before. Keep in mind sometimes this test can fail
    # just because there is no strong difference in some predictions due to the stemmer so it may happen than
    # the network works slightly better than the stemmed version by chance
    assert (predictions_stemmer[0].tolist()[0] >= predictions_no_stemmer[0].tolist()[0])
    assert (predictions_stemmer[1].tolist()[1] >= predictions_no_stemmer[1].tolist()[1])
    assert (predictions_stemmer[2].tolist()[2] >= predictions_no_stemmer[2].tolist()[2])


def test_prediction_for_sentence_in_training_sentence():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2': ['hello', 'how are you', 'greetings'],
         'intent3': ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']})
    bot.configuration.input_max_num_tokens = 7
    bot.configuration.stemmer = True
    bot.configuration.check_exact_prediction_match = True
    train(bot)
    context1: NLUContext = bot.contexts[0]

    text_to_predict = 'I love your cats'
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'Prediction for {text_to_predict} is {prediction}')
    # The prediction sentence is an exact match (once stemmed) for a training sentence in intent1
    assert (prediction.argmax() == 0)
    assert (prediction.tolist()[0] == 1)

    bot.configuration.check_exact_prediction_match = False
    text_to_predict = 'I love your cats'
    prediction: numpy.ndarray = predict(context1, text_to_predict, bot.configuration)
    print(f'Prediction for {text_to_predict} is {prediction}')
    # The prediction sentence is an exact match (once stemmed) for a training sentence in intent1
    assert (prediction.argmax() == 0)
    assert (prediction.tolist()[0] < 1)

