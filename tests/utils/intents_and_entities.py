from xatkitnlu.core.ner.base.base_entities import BaseEntityType
from xatkitnlu.dsl.dsl import CustomEntity, CustomEntityEntry, IntentParameter, Intent, BaseEntity

# Base entities

entity_number: BaseEntity = BaseEntity(BaseEntityType.NUMBER)
entity_date: BaseEntity = BaseEntity(BaseEntityType.DATETIME)

# Custom entities

entity_city: CustomEntity = CustomEntity('entity_city',
                                         [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid')])

entity_museum: CustomEntity = CustomEntity('entity_museum',
                                           [CustomEntityEntry('Louvre', ['Louv', 'Louvre in Paris']),
                                            CustomEntityEntry('Gaudí', ['Gaudí'])])

# Intents

bot1_intents: dict[str, list[str]] = {
    'intent1': ['I love your dog', 'I love your cat', 'You really love my dog!'],
    'intent2': ['hello', 'how is he', 'greetings'],
    'intent3': ['I want a pizza', 'I love a pizza', 'do you sell pizzas', 'can I order a pizza?']
}

intent_greetings: Intent = Intent('intent_greetings',
                                  ['hello', 'how are you'])

intent_weather_ner: Intent = Intent('intent_weather_ner',
                                    ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny?'])
intent_weather_ner.add_parameter(IntentParameter('city', 'mycity', entity_city))

intent_museum_ner: Intent = Intent('intent_museum_ner',
                                   ['I want to visit the mymuseum', 'is the mymuseum open for a visit',
                                    'what about visiting the mymuseum?'])
intent_museum_ner.add_parameter(IntentParameter('museum', 'mymuseum', entity_museum))

intent_museum_no_ner: Intent = Intent('intent_museum_no_ner',
                                      ['I want to visit something interesting',
                                       'is something cool open for a visit?',
                                       'what ideas do we have for tomorrow?'])


intent_temperature_en: Intent = Intent('intent_temperature_en',
                                       ['the temperature outside is NUMBER and it is so cold',
                                        'it is NUMBER degrees outside'])
intent_temperature_en.add_parameter(IntentParameter('temperature', 'NUMBER', entity_number))

intent_greetings_en: Intent = Intent('intent_greetings_en',
                                     ['hello', 'how are you'])

intent_birthday_en: Intent = Intent('intent_birthday_en',
                                    ['My birthday is DATE', 'DATE is my birthday'])
intent_birthday_en.add_parameter(IntentParameter('birthday', 'DATE', entity_date))


intent_temperature_ca: Intent = Intent('intent_temperature_ca',
                                       ['la temperatura fora és de NUMBER i fa molt fred', 'fora fa NUMBER graus'])
intent_temperature_ca.add_parameter(IntentParameter('temperature', 'NUMBER', entity_number))

intent_greetings_ca: Intent = Intent('intent_greetings_ca',
                                     ['hola', 'com estàs'])


intent_temperature_es: Intent = Intent('intent_temperature_es',
                                       ['la temperatura fuera es de NUMBER y hace mucho frío',
                                        'fuera hace NUMBER grados'])
intent_temperature_es.add_parameter(IntentParameter('temperature', 'NUMBER', entity_number))

intent_greetings_es: Intent = Intent('intent_greetings_es',
                                     ['hola', 'como estás'])
