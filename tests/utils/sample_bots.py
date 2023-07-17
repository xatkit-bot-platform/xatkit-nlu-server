import uuid
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, Intent, CustomEntity, CustomEntityEntry, IntentParameter, IntentReference


def create_bot_one_context_one_intent(sentences: list[str]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration(country="en", region="US", lower=True))
    context1: NLUContext = NLUContext('context1')
    intent1: Intent = Intent("intent1", sentences)
    intent1ref: IntentReference = IntentReference("intent1", intent1)
    bot.add_context(context1)
    bot.add_intent(intent1)
    context1.add_intent_ref(intent1ref)
    return bot


def create_bot_one_context_several_intents(sentences: dict[str:[str]]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)

    for key in sentences.keys():
        intent: Intent = Intent(key, sentences.get(key))
        bot.add_intent(intent)
        context1.add_intent_ref(IntentReference(intent.name, intent))
    print(bot)
    return bot


# We create a bot with several intents, including one manually added one with a city NER parameter
def create_bot_one_context_several_intent_with_one_custom_city_intent_with_ner(sentences: dict[str:[str]]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)
    # Adding the given intents
    for key in sentences.keys():
        intent: Intent = Intent(key, sentences.get(key))
        bot.add_intent(intent)
        context1.add_intent_ref(IntentReference(intent.name, intent))

    # Adding an additional intent with a custom entity parameter

    entity: CustomEntity = CustomEntity('cityentity',
                                        [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid'), CustomEntityEntry('Valencia')])
    bot.add_entity(entity)

    city_intent: Intent = Intent('city_intent',
                            ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny in mycity?'])
    city_intent.add_parameter(IntentParameter(entity=entity, name='city', fragment='mycity'))
    bot.add_intent(city_intent)
    context1.add_intent_ref(IntentReference(city_intent.name, city_intent))
    print(bot)
    return bot
