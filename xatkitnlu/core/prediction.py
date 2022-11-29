import numpy as np

from xatkitnlu.core.ner.ner import ner_matching, no_ner_matching
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.core.text_preprocessing import preprocess_text

from xatkitnlu.dsl.dsl import NLUContext, Intent, MatchedParam, Classification, PredictResult
import tensorflow as tf


def predict(context: NLUContext, sentence: str, configuration: NlpConfiguration) -> PredictResult:
    predict_result: PredictResult = PredictResult(context)
    ner_matching_result: dict[Intent, tuple[str, list[MatchedParam]]] = {}
    intent_sentences: dict[str, list[Intent]] = {}
    preprocessed_sentence = preprocess_text(sentence, configuration)
    # We try to replace all potential entity value with the corresponding entity name
    if configuration.use_ner_in_prediction:
        ner_matching_result = ner_matching(context, preprocessed_sentence, configuration)
        for intent, (ner_sentence, _) in ner_matching_result.items():
            # is it necessary to initialize the lists?
            if intent_sentences.get(ner_sentence) is None:
                intent_sentences[ner_sentence] = []
            intent_sentences[ner_sentence].append(intent)
    else:
        # ner_matching_result = no_ner_matching(context, sentence, configuration)
        intent_sentences[preprocessed_sentence] = context.get_intents()

    for (ner_sentence, intents) in intent_sentences.items():
        sentences = [ner_sentence]
        sequences = context.tokenizer.texts_to_sequences(sentences)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post',
                                                               maxlen=configuration.input_max_num_tokens,
                                                               truncating='post')
        run_full_prediction: bool = True
        if configuration.discard_oov_sentences and all(i == 1 for i in sequences[0]):
            # the sentence to predict consists of only out of vocabulary tokens so we can automatically assign a zero probability to all classes
            prediction = np.zeros(len(context.intent_refs))
            run_full_prediction = False  # no need to go ahead with the full NN-based prediction
        elif configuration.check_exact_prediction_match:
            found: bool = False
            found_intent: int
            i: int = 0
            for training_sequence in context.training_sequences:
                if np.array_equal(padded[0], training_sequence):
                    found = True
                    found_intent = context.training_labels[i]
                    run_full_prediction = False
                    break
                i+=1
            if found:
                # We set to true the corresponding intent with full confidence and to zero all the
                # We don't check if there is more than one intent that could be the exact match as this would be an inconsistency in the bot definition anyways
                prediction = np.zeros(len(context.intent_refs))
                np.put(prediction, found_intent, 1.0, mode='raise')

        if run_full_prediction:
            full_prediction = context.nlp_model.predict(padded)
            prediction = full_prediction[0]  # We return just the a single array with the predictions as we predict for just one sentence

        for intent in intents:
            # it is impossible to have a duplicated intent in another ner_sentence
            intent_index = context.get_intents().index(intent)
            matched_ners: list[MatchedParam] = []
            if configuration.use_ner_in_prediction:
                matched_ners = ner_matching_result[intent][1]
            classification: Classification = predict_result.get_classification(intent)
            classification.score = prediction[intent_index]
            classification.matched_utterance = sentence
            classification.matched_params = matched_ners

    return predict_result
