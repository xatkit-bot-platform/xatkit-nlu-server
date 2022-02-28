# import keras_preprocessing.text
from core.nlp_configuration import NlpConfiguration
from dsl.dsl import Bot, NLUContext, Intent
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
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(len(context.intents), activation='sigmoid')  # we stick to sigmoid to be able to have all the potential intents that match
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
        intent.processed_training_sentences.append(preprocess_training_sentence(intent.training_sentences[i], configuration))

def preprocess_training_sentence(sentence: str, configuration: NlpConfiguration):
    if configuration.stemmer:
        return stem_training_sentence(sentence)
    else:
        return sentence

def internal_tokenizer_training_sentence(sentence: str) -> list[str]:
    stanza_pipeline: stanza.pipeline.core.Pipeline = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
    tokenizer_result: stanza.models.common.doc.Document = stanza_pipeline(sentence)
    token_sentence: stanza.models.common.doc.Sentence = tokenizer_result.sentences[0]
    tokens: list[str] = []
    for token in token_sentence.tokens:
        tokens.append(token.text)
    return tokens


def stem_training_sentence(sentence: str) -> str:
    tokens: list[str] = internal_tokenizer_training_sentence(sentence)
    # print(Stemmer.algorithms()) # Names of the languages supported by the stemmer
    stemmer = Stemmer.Stemmer('english')
    stemmed_sentence: list[str] = stemmer.stemWords(tokens)
    print("Stemmed sentence")
    print(stemmed_sentence)
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
