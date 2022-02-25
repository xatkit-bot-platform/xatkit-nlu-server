import uuid
from core.training import train, predict
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext, Intent


def test_predict():
    bot: Bot = create_bot_one_context_several_intents(
        {'intent1' : ['I love your dog', 'I love your cat', 'You really love my dog!'],
         'intent2' : ['hello', 'how are you', 'greetings'],
         'intent3' : ['I want a pizza', 'I would love a pizza', 'do you sell pizzas', 'can I order a pizzza?']})
    bot.configuration.input_max_num_tokens = 7
    train(bot)
    context1: NLUContext = bot.contexts[0]
    predict(context1, 'can I have two more pizza?', bot.configuration)
    predict(context1, 'I love your dog', bot.configuration)
    predict(context1, 'hello!', bot.configuration)


def test_train():
    bot: Bot = create_bot_one_context_one_intent(['I love your dog', 'I love your cat', 'You really love my dog!'])
    bot.configuration.input_max_num_tokens=7
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


def create_bot_one_context_one_intent(sentences: list[str]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration(country="en", region="US", lower=True))
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)
    context1.add_intent(Intent("intent1", sentences))
    return bot


def create_bot_one_context_several_intents(sentences: dict[str :[str]]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration(country="en", region="US", lower=True))
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)

    for key in sentences.keys() :
        intent: Intent = Intent(key, sentences.get(key))
        context1.add_intent(intent)

    print(bot)
    return bot
