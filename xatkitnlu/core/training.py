# import keras_preprocessing.text
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.dsl.dsl import Bot, NLUContext, Intent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Stemmer
import stanza
# from nltk.tokenize import word_tokenize


def train(bot: Bot):
    for context in bot.contexts:
        __train_context(context, bot.configuration)


def __train_context(context: NLUContext, configuration: NlpConfiguration):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=configuration.num_words, lower=configuration.lower, oov_token=configuration.oov_token)
    total_training_sentences: list[str] = []
    total_labels_training_sentences: list[int] = []
    for intent in context.intents:
        preprocess_training_sentences(intent, configuration)
        index_intent = context.intents.index(intent)
        total_training_sentences.extend(intent.processed_training_sentences)
        total_labels_training_sentences.extend([index_intent for i in range(len(intent.processed_training_sentences))])

    tokenizer.fit_on_texts(total_training_sentences)
    context.tokenizer = tokenizer
    context.training_sentences = total_training_sentences
    context.training_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(total_training_sentences),
                                                                               padding='post', maxlen=configuration.input_max_num_tokens)
    context.training_labels = total_labels_training_sentences

    model: tf.keras.models = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=configuration.num_words, output_dim=configuration.embedding_dim, input_length=configuration.input_max_num_tokens),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),  # tanh is also a valid alternative for these intermediate layers
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(len(context.intents), activation='sigmoid')  # we stick to sigmoid to be able to have all the potential intents that match
    ])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    context.nlp_model = model

    print("Model summary: ")
    model.summary()

    # np conversion is needed to get it to work with TensorFlow 2.x
    history = model.fit(np.array(context.training_sequences), np.array(context.training_labels), epochs=configuration.num_epochs, verbose=2)

    plot_training_graphs_without_validation(history, "accuracy")
    plot_training_graphs_without_validation(history, "loss")


def preprocess_training_sentences(intent: Intent, configuration: NlpConfiguration):
    intent.processed_training_sentences = []
    for i in range(len(intent.training_sentences)):
        intent.processed_training_sentences.append(preprocess_training_sentence(intent, intent.training_sentences[i], configuration))


def preprocess_training_sentence(intent: Intent, sentence: str, configuration: NlpConfiguration):
    preprocessed_sentence: str = sentence

    if configuration.use_ner_in_prediction:
        preprocessed_sentence = replace_ner_in_training_sentence(preprocessed_sentence, intent, configuration)
    if configuration.stemmer:
        preprocessed_sentence = stem_training_sentence(preprocessed_sentence, configuration)
    return preprocessed_sentence

def preprocess_training_sentence_no_ner(sentence: str, configuration: NlpConfiguration):
    preprocessed_sentence: str = sentence
    if configuration.stemmer:
        preprocessed_sentence = stem_training_sentence(preprocessed_sentence, configuration)
    return preprocessed_sentence


def replace_ner_in_training_sentence(sentence: str, intent: Intent, configuration: NlpConfiguration):
    preprocessed_sentence: str = sentence
    if intent.entity_parameters is not None:
        for entity_parameter in intent.entity_parameters:
            # preprocessed_sentence = preprocessed_sentence.replace(entity_parameter.fragment, encoding_ner_token + entity_parameter.entity.name + encoding_ner_token)
            preprocessed_sentence = preprocessed_sentence.replace(entity_parameter.fragment, entity_parameter.entity.name.upper())
    return preprocessed_sentence


def internal_tokenizer_training_sentence(sentence: str, configuration: NlpConfiguration) -> list[str]:
    stanza_pipeline: stanza.pipeline.core.Pipeline = stanza.Pipeline(lang=configuration.country, processors='tokenize', tokenize_no_ssplit=True)
    tokenizer_result: stanza.models.common.doc.Document = stanza_pipeline(sentence)
    token_sentence: stanza.models.common.doc.Sentence = tokenizer_result.sentences[0]
    tokens: list[str] = []
    for token in token_sentence.tokens:
        tokens.append(token.text)
    return tokens


def stem_training_sentence(sentence: str, configuration: NlpConfiguration) -> str:
    tokens: list[str] = internal_tokenizer_training_sentence(sentence, configuration)
    # print(Stemmer.algorithms()) # Names of the languages supported by the stemmer
    stemmer_language: str
    if configuration.country == "en":
        stemmer_language = "english"
    elif configuration.country == "es":
        stemmer_language = "spanish"
    elif configuration.country == "fr":
        stemmer_language = "french"
    elif configuration.country == "it":
        stemmer_language = "italian"
    elif configuration.country == "de":
        stemmer_language = "german"
    elif configuration.country == "nl":
        stemmer_language = "dutch"
    elif configuration.country == "pt":
        stemmer_language = "portuguese"
    elif configuration.country == "ca":
        stemmer_language = "catalan"
    else:
        stemmer_language = "english"  # If not in the list we revert back to english as default

    stemmer = Stemmer.Stemmer(stemmer_language)
    stemmed_sentence: list[str] = []

    # We stem words one by one to be able to skip words all in uppercase (e.g. references to entity types)
    for word in tokens:
        stemmed_word: str = word
        if not word.isupper():
            stemmed_word = stemmer.stemWord(word)
        stemmed_sentence.append(stemmed_word)

    # stemmed_sentence: list[str] = stemmer.stemWords(tokens)
    # print("Stemmed sentence")
    # print(stemmed_sentence)
    joined_string = ' '.join([str(item) for item in stemmed_sentence])
    return joined_string


def plot_training_graphs_with_validation(history, metric: str):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


def plot_training_graphs_without_validation(history, metric: str):
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.show()
