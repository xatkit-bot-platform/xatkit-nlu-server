

class NlpConfiguration:
    country: str
    region: str
    num_words: int # max num of words to keep in the index of words
    num_epochs: int
    lower: bool # transform sentences to lowercase
    oov_token: str #token for the out of vocabulary words
    embedding_dim: int
    input_max_num_tokens: int # max length for the vector representing a sentence
    stemmer: bool #whether to use a stemmer


    def __init__(self, country: str = "en", region: str = "US", numwords: int = 1000, lower: bool = True, oov_token="<OOV>",
                 num_epochs: int = 300, embedding_dim: int = 16, input_max_num_tokens: int = 30, stemmer: bool = True):
        self.country = country
        self.region = region
        self.num_words = numwords
        self.lower = lower
        self.oov_token = oov_token
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_dim
        self.input_max_num_tokens = input_max_num_tokens
        self.stemmer = stemmer

