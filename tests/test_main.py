from fastapi.testclient import TestClient
from main import app, bots
import uuid

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
    response = client.post("/bot/new", )
    uuid_str = response.json().get("uuid")
    bot_id: uuid = uuid.UUID(uuid_str)
    assert response.status_code == 200
    assert bot_id in bots.keys()

