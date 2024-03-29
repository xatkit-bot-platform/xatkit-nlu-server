A flexible and pragmatic NLU intent matching server for building chatbots
======

[![Wiki Badge](https://img.shields.io/badge/doc-wiki-blue)](https://github.com/xatkit-bot-platform/xatkit/wiki)
[![Twitter](https://img.shields.io/twitter/follow/xatkit?label=Follow&style=social)](https://twitter.com/xatkit)
[![GitHub forks](https://img.shields.io/github/forks/xatkit-bot-platform/xatkit-nlu-server?style=social)](https://github.com/xatkit-bot-platform/xatkit/network/members)
[![GitHub Repo stars](https://img.shields.io/github/stars/xatkit-bot-platform/xatkit-nlu-server?style=social)](https://github.com/xatkit-bot-platform/xatkit/stargazers)


At the core every chatbot there is a intent recognition component in charge of matching user utterances to one of the available chatbot intents. This is [Xatkit](https://github.com/xatkit-bot-platform)'s own NLU server. Note that:

- Following Xatkit's [chatbot orchestration platform](https://xatkit.com/chatbot-orchestration-platform-open-source/) philosophy, you can use any available [Intent Recognition engines](https://github.com/xatkit-bot-platform/xatkit/wiki/Intent-Recognition-Providers) to build your Xatkit bots.
- Thanks to its public REST API, this NLU Engine can be integrated in other NLP solutions, not only in Xatkit. 

> Note that this NLU engine is still in an alpha phase. So, great for learning and playing. Not ready for any type of production use. 
> Keep also in mind the project is quickly evolving, this includes its public APIs. Until a first stable version, breaking changes can occur at any time. 


## What makes this Intent Matching project different?

If there is something we have learnt is that there is no *one size fits all* solution when it comes to the Natural Language processing requirements for a chatbot project.

And we wanted to offer a solution that was easy to adapt to any chatbot requirement and that offer a range of intent matching options, 
mostly adopting a rather pragmatic approach, instead of the **typical** solution of building a large neural network for the whole bot.

Some Xatkit's NLU specific characteristics:

- ### Xatkit creates a *separate neural network* for each bot context. 

We see bots as having different conversation contexts (e.g. as part of a [bot state machine](https://xatkit.com/chatbot-dsl-state-machines-xatkit-language/)). When in a given context,
only the intents that make sense in that context should be evaluated when considering possible matches.

A Xatkit bot is composed of contexts where each contexts may include a number of intents (see the `dsl` package). During the training phase, a NLP model is trained on those intents' training sentences and attached to the context for future predictions).


- ### Xatkit understands that *a neural network is not always the ideal solution* for intent matching

What if the user input text is full of words the NN has never seen before? It's safe to assume that we can directly determine there is no matching and trigger a bot move to the a [default fallback](https://github.com/xatkit-bot-platform/xatkit/wiki/Default-and-Local-Fallback) state.

Or what if the input text is a perfect literal match to one of the training sentences? Shouldn't we assume that's the intent to be returned with maximum confidence? 

This type of pragmatic decisions are at the core of Xatkit to make it a really useful chatbot-specific intent matching project. 


For more detail on how Xatkit's NLU has been built and how to use it, see also [this blog post](https://xatkit.com/how-to-build-chatbot-intent-classifier/)

## Features

Right now, the engine focus on intent matching and NER (named-entity recognition).

### Custom NER

We allow you to define your own set of entities with their respective values and synonyms.

An example entity could be 'CityEntity' with values 'Barcelona' (with synonyms: 'BCN', 'Barna') and 'Madrid'.

### Base NER

We develop predefined entity types that can be used within any intent:

- **Number** (e.g. '3', '3.5', '-3.5', 'minus three point five', etc.)
- **Date-time** (e.g. 'December 1st 2022 at 15:00', 'December', '4pm', 'today', 'in 3 months', etc.)

See [base_entities.py](xatkitnlu/core/ner/base/base_entities.py) to check all the Base Entity Types.

Check also the `tests` folder for some examples of how to use the engine and its configuration options (described below).

## Installing Xatkit NLU

Xatkit NLU engine has been tested with Python 3.9.

Other key requirements are: 

- numpy~=1.22.2
- fastapi~=0.74.0
- tensorflow~=2.8.0
- pydantic~=1.9.0
- matplotlib~=3.5.1
- stanza~=1.3.0
- PyStemmer~=2.0.1
- text2num~=2.5.0
- dateparser~=1.1.3
- python-dateutil~=2.8.2

[FastAPI](https://fastapi.tiangolo.com/) is the web framework that we use to expose the NLU engine as a REST API. You will probably recognize most of the other dependencies :-)

> We use Stanza's language-dependent tokenizer. You'll need to [download the language models](https://stanfordnlp.github.io/stanza/download_models.html) you'll be using in your bots before running the server

## Running Xatkit NLU

FastAPI relies on [uvicorn](https://www.uvicorn.org/) as ASGI web server implementation. 

To expose Xatkit as a web API write: 

` python  -m uvicorn main:app --log-level trace`
 
where main is the module where the FastAPI app resides.

## Configuration options

List of configuration options and default values (see `nlp_configuration.py`)

| Key                            | Values  | Description                                                                                        | Constraint                         |
|--------------------------------|---------|----------------------------------------------------------------------------------------------------|------------------------------------|
| `country`                      | String  | The country language used by the bot                                                               | Optional (default `en`)            |
| `region`                       | String  | The region code used by the bot                                                                    | Optional (default `US`)            |
| `timezone`                     | String  | The timezone used by the bot                                                                       | Optional (default `Europe/Madrid`) |
| `num_words`                    | int     | Max number of words to keep track of in the word index                                             | Optional (default `1000`)          |
| `lower`                        | Boolean | Whether all strings should be transformed to lowercase                                             | Optional (default `true`)          |
| `oov_token`                    | String  | Token to represent out of vocabulary words during prediction                                       | Optional (default `<OOV>`)         |
| `num_epochs`                   | int     | Number of epochs to run during training                                                            | Optional (default `300`)           |
| `embedding_dim`                | int     | Number of dimensions to be sued during the embedding of word tokens                                | Optional (default `128`)           |
| `input_max_num_tokens`         | int     | Max length (in terms of number of tokens) to keep for all sentences                                | Optional (default `30`)            |
| `stemmer`                      | Boolean | Whether to use a Stemmer as part of the training sentences (and user utterances) processing        | Optional (default `True`)          |
| `discard_oov_sentences`        | Boolean | Automatically assign a zero probability to all intents when the user utterance is all OOV tokens   | Optional (default `True`)          |
| `check_exact_prediction_match` | Boolean | Whether to check for exact match between the sentence to predict and one of the training sentences | Optional (default `True`)          |
| `use_ner_in_prediction`        | Boolean | Should entity matches be used during prediction?                                                   | Optional (default `True`)          |
| `activation_last_layer`        | String  | The activation function of the last layer                                                          | Optional (default `sigmoid`)       |
| `activation_hidden_layers`     | String  | The activation function of the hidden layers                                                       | Optional (default `tanh`)          |


## Contributing

Do you want to contribute to Xatkit? We would love to hear from you. Remember that there are [many ways to support open source projects](https://livablesoftware.com/5-ways-to-thank-open-source-maintainers/) beyond committing code!. Talking about Xatkit, writing documentation, contributing examples,... all are great ways to help us.

When contributing code, please first discuss the change you wish to make with us. Start by opening a descriptive issue so that we can advise on the best way to proceed with your bug fix or new feature idea. 

**Thanks for reading until the end! If you like what you see, don't forget to star/watch this repository, you'll make us very happy!**
