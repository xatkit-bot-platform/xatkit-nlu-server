import uuid
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext, Intent


def create_bot_one_context_one_intent(sentences: list[str]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration(country="en", region="US", lower=True))
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)
    context1.add_intent(Intent("intent1", sentences))
    return bot


def create_bot_one_context_several_intents(sentences: dict[str:[str]]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)

    for key in sentences.keys():
        intent: Intent = Intent(key, sentences.get(key))
        context1.add_intent(intent)
    print(bot)
    return bot
