class NLUContext:
    """Context state for which we must choose the right intent to match"""
    name: str

    def __init__(self, name: str):
        self.name = name
