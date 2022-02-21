from dsl.dsl import Bot, NLUContext, Configuration
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

def train(bot: Bot) -> Bot:
    for context in bot.contexts :
        __train_context(context,bot.configuration)


def __train_context(context: NLUContext, configuration: Configuration) -> NLUContext:
    sentences = [
        'i love my dog',
        'I, love my cat',
        'You love my dog!'
    ]

    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(word_index)

