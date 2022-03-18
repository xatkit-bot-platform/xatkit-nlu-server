class NlpConfiguration:

    def __init__(self, country: str = "en", region: str = "US", numwords: int = 1000, lower: bool = True, oov_token="<OOV>",
                 num_epochs: int = 300, embedding_dim: int = 16, input_max_num_tokens: int = 30, stemmer: bool = True):
        self.country = country
        self.region = region
        self.num_words = numwords # max num of words to keep in the index of words
        self.lower = lower # transform sentences to lowercase
        self.oov_token = oov_token #token for the out of vocabulary words
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_dim
        self.input_max_num_tokens = input_max_num_tokens # max length for the vector representing a sentence
        self.stemmer = stemmer #whether to use a stemmer

