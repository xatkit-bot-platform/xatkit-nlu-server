# import keras_preprocessing.text
from xatkitnlu.core.nlp_configuration import NlpConfiguration
from xatkitnlu.core.text_preprocessing import preprocess_training_sentences, preprocess_custom_entity_entries
from xatkitnlu.dsl.dsl import Bot, NLUContext, CustomEntity
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# from nltk.tokenize import word_tokenize


def train(bot: Bot):
    for context in bot.contexts:
        __train_context(context, bot.configuration)
    for entity in bot.entities:
        if isinstance(entity, CustomEntity):
            preprocess_custom_entity_entries(entity, bot.configuration)


def __train_context(context: NLUContext, configuration: NlpConfiguration):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=configuration.num_words, lower=configuration.lower, oov_token=configuration.oov_token)
    total_training_sentences: list[str] = []
    total_labels_training_sentences: list[int] = []
    for intent_ref in context.intent_refs:
        intent = intent_ref.intent
        preprocess_training_sentences(intent, configuration)
        index_intent = context.intent_refs.index(intent_ref)
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
        tf.keras.layers.Dense(24, activation=configuration.activation_hidden_layers),  # tanh is also a valid alternative for these intermediate layers
        tf.keras.layers.Dense(24, activation=configuration.activation_hidden_layers),
        tf.keras.layers.Dense(len(context.intent_refs), activation=configuration.activation_last_layer)  # choose sigmoid if, in your scenario, a sentence could possibly match several intents
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    context.nlp_model = model

    print("Model summary: ")
    model.summary()

    # np conversion is needed to get it to work with TensorFlow 2.x
    history = model.fit(np.array(context.training_sequences), np.array(context.training_labels), epochs=configuration.num_epochs, verbose=2)

    plot_training_graphs_without_validation(history, "accuracy")
    plot_training_graphs_without_validation(history, "loss")


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
