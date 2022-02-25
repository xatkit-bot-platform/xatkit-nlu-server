

class NlpConfiguration:
    country: str
    region: str
    num_words: int
    num_epochs: int
    lower: bool
    oov_token: str
    embedding_dim: int
    input_max_num_tokens: int



    def __init__(self, country: str = "en", region: str = "US", numwords: int = 1000, lower: bool = True, oov_token="<OOV>",
                 num_epochs=300, embedding_dim=16, input_max_num_tokens=15):
        self.country = country
        self.region = region
        self.num_words = numwords
        self.lower = lower
        self.oov_token = oov_token
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_dim
        self.input_max_num_tokens = input_max_num_tokens

