import uuid

class Intent:
    """A chatbot intent"""
    name: str
    training_sentences: list[str] = []


class NLUContext:
    """Context state for which we must choose the right intent to match"""
    name: str

    def __init__(self, name: str):
        """Initialize a context object

        :param name (str): name of the newcontext
        """
        self.name = name


class Bot:
    id: uuid
    contexts: list[NLUContext] = []
