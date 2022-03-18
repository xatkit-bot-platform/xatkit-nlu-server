from fastapi.testclient import TestClient

from dto.dto import BotDTO, NLUContextDTO, IntentDTO, ConfigurationDTO, PredictDTO
from main import app, bots

client = TestClient(app)


def test_server_up():
    response = client.get("/")
    assert response.status_code == 200
    response_dict: dict[str, str] = response.json()
    assert '2.8.0' in response_dict.values()


def test_hello_endpoint():
    response = client.get("/hello/Jordi")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Jordi"}


def test_new_bot():
    response = client.post("/bot/new/", json={"name": "newbot", "force_overwrite": "false"} )
    print(response.json())
    assert response.status_code == 200
    assert "newbot" in bots
    response = client.post("/bot/new/", json={"name": "newbot", "force_overwrite": "false"})
    assert response.status_code == 422
    response = client.post("/bot/new/", json={"name": "newbot", "force_overwrite": "true"})


def test_initialize_bot():
    initialization_data: BotDTO = BotDTO(name="newbot")

    context1: NLUContextDTO = NLUContextDTO(name="context1", intents=[
        IntentDTO(name="intent1", training_sentences=['I love your dog', 'I love your cat', 'You really love my dog!']),
        IntentDTO(name="intent2", training_sentences=['Hello', 'Hi'])])
    initialization_data.contexts.append(context1)
    print(initialization_data)
    print(initialization_data.dict())
    print(initialization_data.json())


    response = client.post("/bot/newbot/initialize/", initialization_data.json())
    assert response.status_code == 422

    client.post("/bot/new/", json={"name": "newbot", "force_overwrite": "true"})
    response = client.post("/bot/newbot/initialize/", initialization_data.json())
    assert bots["newbot"].contexts[0].intents[0].training_sentences[0] == "I love your dog"

    print(response.text)
    assert response.status_code == 200


def test_train():
    client.post("/bot/new/", json={"name": "newbot", "force_overwrite": "true"})

    initialization_data: BotDTO = BotDTO(name="newbot")
    context1: NLUContextDTO = NLUContextDTO(name="context1", intents=[
        IntentDTO(name="intent1", training_sentences=['I love your dog', 'I love your cat', 'You really love my dog!']),
        IntentDTO(name="intent2", training_sentences=['Hello', 'Hi'])])
    initialization_data.contexts.append(context1)

    client.post("/bot/newbot/initialize", initialization_data.json())

    configuration: ConfigurationDTO = ConfigurationDTO(input_max_num_tokens = 10)
    response = client.post("/bot/newbot/train", configuration.json())
    assert response.status_code == 200
    assert bots["newbot"].configuration.input_max_num_tokens == 10
    assert bots["newbot"].configuration.oov_token == "<OOV>"


    response = client.post("/bot/newbot/predict", configuration.json())


def test_predict():
    client.post("/bot/new/", json={"name": "newbot", "force_overwrite": "true"})

    initialization_data: BotDTO = BotDTO(name="newbot")
    context1: NLUContextDTO = NLUContextDTO(name="context1", intents=[
        IntentDTO(name="intent1", training_sentences=['I love your dog', 'I love your cat', 'You really love my dog!']),
        IntentDTO(name="intent2", training_sentences=['Hello', 'Hi'])])
    initialization_data.contexts.append(context1)

    client.post("/bot/newbot/initialize", initialization_data.json())

    configuration: ConfigurationDTO = ConfigurationDTO(input_max_num_tokens = 10, stemmer=True)
    response = client.post("/bot/newbot/train", configuration.json())

    prediction_request: PredictDTO = PredictDTO(utterance="he loves dogs", context="context2")
    response = client.post("/bot/newbot/predict", prediction_request.json())
    assert response.status_code == 422

    prediction_request: PredictDTO = PredictDTO(utterance="he loves dogs", context="context1")
    response = client.post("/bot/newbot/predict", prediction_request.json())
    assert response.status_code == 200
    print(response.text)


