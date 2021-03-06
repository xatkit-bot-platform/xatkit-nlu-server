import numpy

from core.ner import ner_matching
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.core.training import preprocess_training_sentence_no_ner
from xatkitnlu.dsl.dsl import NLUContext
import tensorflow as tf


def predict(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> tuple[numpy.ndarray, dict[str, dict[str, str]]]:

    prediction: numpy.ndarray
    matched_ners: dict[str, dict[str, str]] = {}

    # We try to replace all potential entity value with the corresponding entity name
    if configuration.use_ner_in_prediction:
        sentence, matched_ners = ner_matching(context, sentence, configuration)

    sentences = [preprocess_prediction_sentence(sentence, configuration)]
    sequences = context.tokenizer.texts_to_sequences(sentences)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post',
                                                           maxlen=configuration.input_max_num_tokens,
                                                           truncating='post')
    run_full_prediction: bool = True
    if configuration.discard_oov_sentences and all(i == 1 for i in sequences[0]):
        # the sentence to predict consists of only out of focabulary tokens so we can automatically assign a zero probability to all classes
        prediction = numpy.zeros(len(context.intents))
        run_full_prediction = False  # no need to go ahead with the full NN-based prediction
    elif configuration.check_exact_prediction_match:
        found: bool = False
        found_intent: int
        i: int = 0
        for training_sequence in context.training_sequences:
            if numpy.array_equal(padded[0], training_sequence):
                found = True
                found_intent = context.training_labels[i]
                run_full_prediction = False
                break
            i+=1
        if found:
            # We set to true the corresponding intent with full confidence and to zero all the
            # We don't check if there is more than one intent that could be the exact match as this would be an inconsistency in the bot definition anyways
            prediction = numpy.zeros(len(context.intents))
            numpy.put(prediction, found_intent, 1.0, mode='raise')

    if run_full_prediction:
        full_prediction = context.nlp_model.predict(padded)
        prediction = full_prediction[0]  # We return just the a single array with the predictions as we predict for just one sentence

    return prediction, matched_ners


def preprocess_prediction_sentence(sentence: str, configuration: NlpConfiguration) -> str:
    return preprocess_training_sentence_no_ner(sentence, configuration)
