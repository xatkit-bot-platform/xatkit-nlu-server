# import keras_preprocessing.text

from dsl.dsl import Bot, NLUContext, Configuration
import tensorflow as tf


def train(bot: Bot):
    for context in bot.contexts:
        __train_context(context, bot.configuration)


def __train_context(context: NLUContext, configuration: Configuration):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(configuration.numwords)
    total_training_sentences: list[str] = []
    for intent in context.intents:
        total_training_sentences.extend(intent.training_sentences)

    tokenizer.fit_on_texts(total_training_sentences)
    context.tokenizer = tokenizer
