import uuid

from core.training import train
from dsl.dsl import Bot, NLUContext, Intent, Configuration


def test_train():
    bot: Bot = create_bot_one_context_one_intent(['I love your dog', 'I love your cat', 'You love my dog'])
    train(bot)
    context1 : NLUContext = bot.contexts[0]
    word_index = context1.tokenizer.word_index
    print(word_index)


def create_bot_one_context_one_intent(sentences: list[str]):
    bot: Bot = Bot('test bot', uuid.uuid4(), Configuration("en","US"))
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)
    context1.add_intent(Intent("intent1", sentences))
    return bot