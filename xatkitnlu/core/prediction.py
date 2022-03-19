import numpy

from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.core.training import preprocess_training_sentence
from xatkitnlu.dsl.dsl import NLUContext
import tensorflow as tf


def predict(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> numpy.ndarray:

    prediction: numpy.ndarray
    sentences = [preprocess_prediction_sentence(sentence, configuration)]
    sequences = context.tokenizer.texts_to_sequences(sentences)
    if configuration.discard_oov_sentences and all(i==1 for i in sequences[0]):
        # the sentence to predict consists of only out of focabulary tokens so we can automatically assign a zero probability to all classes
        prediction = numpy.zeros(len(context.intents))
    else:
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=configuration.input_max_num_tokens, truncating='post')
        full_prediction = context.nlp_model.predict(padded)
        prediction = full_prediction[0] # We return just the a single array with the predictions as we predict for just one sentence

    return prediction


def preprocess_prediction_sentence(sentence: str, configuration: NlpConfiguration) -> str:
    return preprocess_training_sentence(sentence, configuration)
