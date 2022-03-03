import numpy

from core.nlp_configuration import NlpConfiguration
from core.training import preprocess_training_sentence
from dsl.dsl import NLUContext
import tensorflow as tf


def predict(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> numpy.ndarray:
    sentences = [preprocess_prediction_sentence(sentence, configuration)]
    sequence = context.tokenizer.texts_to_sequences(sentences)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=configuration.input_max_num_tokens, truncating='post')
    prediction = context.nlp_model.predict(padded)
    return prediction[0] # We return just the a single array with the predictions as we predict for just one sentence
    # print(f'Prediction for {sentence} is {prediction}')


def preprocess_prediction_sentence(sentence: str, configuration: NlpConfiguration) -> str:
    return preprocess_training_sentence(sentence, configuration)
