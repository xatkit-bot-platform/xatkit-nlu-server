import uuid
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext, Intent, CustomEntity, CustomEntityEntry, EntityReference


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


# We create a bot with several intents, including one manually added one with a city NER parameter
def create_bot_one_context_several_intent_with_one_custom_city_intent_with_ner(sentences: dict[str:[str]]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)
    # Adding the given intents
    for key in sentences.keys():
        intent: Intent = Intent(key, sentences.get(key))
        context1.add_intent(intent)

    # Adding an additional intent with a custom entity parameter

    entity: CustomEntity = CustomEntity('cityentity',
                                        [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid'), CustomEntityEntry('Valencia')])
    city_intent: Intent = Intent('city_intent',
                            ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny in mycity?'])

    city_intent.add_entity_parameter(EntityReference(entity=entity, name='city', fragment='mycity'))


    context1.add_intent(city_intent)
    print(bot)
    return bot


def create_bot_one_context_several_intent_with_several_custom_ner(sentences: dict[str: [[str],[str]]],
                                                                       entities: dict[str: [dict[str: [str]]]],
                                                                       intent_entites_map: dict[str, str]) -> Bot:
    bot: Bot = Bot(uuid.uuid4(), 'test bot', NlpConfiguration())
    context1: NLUContext = NLUContext('context1')
    bot.add_context(context1)
    custom_entities: dict[str, CustomEntity] = []

    # Adding the given intents
    for key in sentences.keys():
        intent: Intent = Intent(key, sentences.get(key))
        context1.add_intent(intent)

    for key in entities.keys():
        entity: CustomEntity = CustomEntity(key)
        entries: entities.get(key)
        custom_entity_entries: list[CustomEntityEntry] = []
        for entry_key in entries.keys:
            custom_entry: CustomEntityEntry = CustomEntityEntry(entry_key, entries.get(entry_key))
            custom_entity_entries.append(custom_entry)
        entity.entries.append(custom_entity_entries)
        custom_entities[key, entity]

    for key in intent_entites_map.keys():
        intent: Intent = context1.get_intent(key)
        entity: CustomEntity = custom_entities[intent_entites_map.get(key)]
        intent.add_entity_reference(EntityReference(entity.name, entity.name, entity))


    print(bot)
    return bot
