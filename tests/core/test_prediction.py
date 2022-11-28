import numpy as np

from tests.utils.intents_and_entities import bot1_intents
from xatkitnlu.core.prediction import predict
from xatkitnlu.core.training import train
from xatkitnlu.dsl.dsl import Bot, NLUContext, PredictResult
from tests.utils.sample_bots import create_bot_one_context_several_intents


def test_predict_without_stemmer():
    bot: Bot = create_bot_one_context_several_intents(bot1_intents)
    bot.configuration.input_max_num_tokens = 7
    bot.stemmer = False
    train(bot)
    context1: NLUContext = bot.contexts[0]

    prediction: PredictResult
    scores: list[float]

    text_to_predict = 'I love your dogs'
    prediction = predict(context1, text_to_predict, bot.configuration)
    scores = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {text_to_predict} is {scores}')
    assert (np.argmax(scores) == 0)

    text_to_predict = 'hello!'
    prediction = predict(context1, text_to_predict, bot.configuration)
    scores = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {text_to_predict} is {scores}')
    assert (np.argmax(scores) == 1)

    text_to_predict = 'can I have two more pizzas?'
    prediction = predict(context1, text_to_predict, bot.configuration)
    scores = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {text_to_predict} is {scores}')
    assert (np.argmax(scores) == 2)


def test_predict_with_stemmer():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'How cute are your dogs', 'You seem to really! love my dog'],
         'intent2': ['hello', 'how are you', 'greetings'],
         'intent3': ['I prefer cats over dogs', 'I would prefer a cat', 'I love cats', 'I think cats are better']})
    bot.configuration.input_max_num_tokens = 10

    sentences_to_predict = ["He loves dogs", "hello!!", "I'm more of a cat lover"]
    context1: NLUContext = bot.contexts[0]

    bot.configuration.stemmer = False
    predictions_no_stemmer: list[list[float]] = []
    train(bot)
    for sentence in sentences_to_predict:
        prediction: PredictResult = predict(context1, sentence, bot.configuration)
        scores: list[float] = [classification.score for classification in prediction.classifications]
        predictions_no_stemmer.append(scores)

    bot.configuration.stemmer = True
    train(bot)
    predictions_stemmer: list[list[float]] = []
    for sentence in sentences_to_predict:
        prediction: PredictResult = predict(context1, sentence, bot.configuration)
        scores: list[float] = [classification.score for classification in prediction.classifications]
        predictions_stemmer.append(scores)

    print("Predictions without stemmer")
    print(predictions_no_stemmer)
    print("Predictions with stemmer")
    print(predictions_stemmer)

    # We check with the stemmer we get the good categories
    assert(np.argmax(predictions_stemmer[0]) == 0)
    assert(np.argmax(predictions_stemmer[1]) == 1)
    assert(np.argmax(predictions_stemmer[2]) == 2)

    # We check the confidence is at least as good as before. Keep in mind sometimes this test can fail
    # just because there is no strong difference in some predictions due to the stemmer so it may happen than
    # the network works slightly better than the stemmed version by chance
    assert (predictions_stemmer[0][0] >= predictions_no_stemmer[0][0])
    assert (predictions_stemmer[1][1] >= predictions_no_stemmer[1][1])
    assert (predictions_stemmer[2][2] >= predictions_no_stemmer[2][2])


def test_prediction_when_prediction_sentence_is_all_oov():
    bot: Bot = create_bot_one_context_several_intents(bot1_intents)

    bot.configuration.discard_oov_sentences = True
    train(bot)
    text_to_predict = 'xsx dfasklj adfa'
    prediction: PredictResult = predict(bot.contexts[0], text_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {text_to_predict} is {scores}')
    # The prediction sentence is a set of oov tokens so we expect the prediction to be all zeros
    assert (max(scores) == 0)


def test_prediction_for_when_prediction_sentence_is_in_training_sentence():
    bot: Bot = create_bot_one_context_several_intents(bot1_intents)
    bot.configuration.input_max_num_tokens = 7
    bot.configuration.stemmer = True
    bot.configuration.check_exact_prediction_match = True
    train(bot)
    context1: NLUContext = bot.contexts[0]

    text_to_predict = 'I love your cats'
    prediction: PredictResult = predict(context1, text_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {text_to_predict} is {scores}')
    # The prediction sentence is an exact match (once stemmed) for a training sentence in intent1
    assert (np.argmax(scores) == 0)
    assert (scores[0] == 1)

    bot.configuration.check_exact_prediction_match = False
    text_to_predict = 'I love your cats'
    prediction: PredictResult = predict(context1, text_to_predict, bot.configuration)
    scores: list[float] = [classification.score for classification in prediction.classifications]
    print(f'Prediction for {text_to_predict} is {scores}')
    # The prediction sentence is an exact match (once stemmed) for a training sentence in intent1
    assert (np.argmax(scores) == 0)
    assert (scores[0] < 1)

