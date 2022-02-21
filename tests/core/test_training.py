import uuid

from core.training import train
from dsl.dsl import Bot, NLUContext


def test_train() :
    bot: Bot = Bot('test bot', uuid.uuid4())
    context1: NLUContext = NLUContext('context1')
    bot.add_context((context1))
    train(bot)
