from xatkitnlu.dsl.dsl import NLUContext, Intent, CustomEntity, CustomEntityEntry, IntentParameter


def test_nlucontext():
    assert True


def test_nlucontext_initialization():
    nlu_context = NLUContext('a context')
    assert nlu_context.name == 'a context'


def test_intent_with_ner_initialization():
    entity: CustomEntity = CustomEntity('city_entity', [CustomEntityEntry('Barcelona', ['BCN']), CustomEntityEntry('Madrid')])
    intent: Intent = Intent('intent_name', ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny?'])
    intent.add_parameter(IntentParameter('city', 'mycity', entity))
    assert intent.name == 'intent_name'
    assert intent.training_sentences == ['what is the weather like in mycity', 'forecast for mycity', 'is it sunny?']
    assert intent.parameters[0].entity.name == 'city_entity'
    assert intent.parameters[0].fragment == 'mycity'
    assert intent.parameters[0].name == 'city'
    assert isinstance(intent.parameters[0].entity, CustomEntity)
    assert intent.parameters[0].entity.entries[0].value == 'Barcelona'
    assert intent.parameters[0].entity.entries[0].synonyms[0] == 'BCN'


# TODO: Test BaseEntity


