from dsl.dsl import NLUContext


def test_nlucontext() :
    assert True

def test_nlucontext_initialization() :
    nlu_context = NLUContext('a context')
    assert nlu_context.name == 'a context'


