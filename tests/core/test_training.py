from xatkitnlu.core.prediction import predict
from xatkitnlu.core.text_preprocessing import stem_text
from xatkitnlu.core.training import train
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, PredictResult
from tests.utils.sample_bots import create_bot_one_context_one_intent, create_bot_one_context_several_intents


def test_stemmer():
    configuration: NlpConfiguration = NlpConfiguration()
    configuration.country = 'en'
    stemmed_sentence = stem_text("He loves my dogs", configuration)
    assert(stemmed_sentence == "He love my dog")
    print(stemmed_sentence)


def test_train():
    bot: Bot = create_bot_one_context_one_intent(['I love your dog', 'I love your cat', 'You really love my dog!'])
    bot.configuration.input_max_num_tokens = 7
    train(bot)
    context1: NLUContext = bot.contexts[0]
    word_index = context1.tokenizer.word_index
    print(word_index)
    assert len(list(word_index.values())) == 9

    print("Training sentences")
    print(context1.training_sentences)
    print("Training sequences")
    print(context1.training_sequences)
    print("Training labels")
    print(context1.training_labels)

    assert len(context1.training_sentences) == len(context1.training_sequences)
    assert len(context1.training_sequences) == len(context1.training_labels)
    assert len(context1.training_sequences[0]) == bot.configuration.input_max_num_tokens


def test_parameter_combinations():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'How cute are your dogs', 'You seem to really like dogs', 'dogs are amazing', 'dogs all day!'],
         'intent2': ['hello', 'how are you', 'greetings', 'hi', 'hello all!', 'how do you do?', 'how are you going?'],
         'intent3': ['I prefer cats over dogs', 'I would prefer a cat', 'I love cats', 'I think cats are better', 'I go for cats']})

    sentences_to_predict = ["He loves dogs", "hello!!", "I'm more of a cat person", 'dafj kñj kdañ jklda','this is my red car']
    context1: NLUContext = bot.contexts[0]

    train(bot)
    predictions: list[PredictResult] = []
    for sentence in sentences_to_predict:
        predictions.append(predict(context1, sentence, bot.configuration))

    print("Predictions standard parameters")
    print(predictions)

    bot.configuration.embedding_dim = 128
    train(bot)
    context1: NLUContext = bot.contexts[0]
    predictions: list[PredictResult] = []
    for sentence in sentences_to_predict:
        predictions.append(predict(context1, sentence, bot.configuration))

    print("Repeated predictions with higher embedding dimensions")
    print(predictions)

    bot: Bot = create_bot_one_context_several_intents(
        {'intent1': ['I love your dog', 'How cute are your dogs', 'You seem to really like dogs', 'dogs are amazing', 'dogs all day!'],
         'intent2': ['hello all you', 'how are you', 'greetings to all', 'hi to all', 'hello all!', 'how do you do?', 'how are you going?'],
         'intent3': ['I prefer cats over dogs', 'I would prefer a cat', 'I love cats', 'I think cats are better', 'I go for cats']})
    bot.configuration.embedding_dim = 128
    # bot.configuration.discard_oov_sentences = False
    train(bot)
    context1: NLUContext = bot.contexts[0]
    predictions: list[PredictResult] = []
    for sentence in sentences_to_predict:
        predictions.append(predict(context1, sentence, bot.configuration))

    print("Repeated predictions with slightly longer training sentences")
    print(predictions)


