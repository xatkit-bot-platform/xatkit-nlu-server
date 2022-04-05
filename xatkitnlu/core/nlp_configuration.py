class NlpConfiguration:

    def __init__(self, country: str = "en", region: str = "US", numwords: int = 1000, lower: bool = True, oov_token="<OOV>",
                 num_epochs: int = 300, embedding_dim: int = 128, input_max_num_tokens: int = 30, stemmer: bool = True,
                 discard_oov_sentences = True, check_exact_prediction_match = True,
                 use_ner_in_prediction = True):
        self.country = country
        self.region = region
        self.num_words = numwords # max num of words to keep in the index of words
        self.lower = lower # transform sentences to lowercase
        self.oov_token = oov_token #token for the out of vocabulary words
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_dim
        self.input_max_num_tokens = input_max_num_tokens # max length for the vector representing a sentence
        self.stemmer = stemmer #whether to use a stemmer
        self.discard_oov_sentences = discard_oov_sentences #Automatically assign zero probabilities to sentences with all tokens being oov ones
        self.check_exact_prediction_match = check_exact_prediction_match #whether to check for exact match between the sentence to predict and one of the training sentences
        self.use_ner_in_prediction = use_ner_in_prediction #whether to use NER in the prediction
