
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)



def test_hello_endpoint():
    response = client.get("/hello/Jordi")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Jordi"}