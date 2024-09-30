import tests.test_inference as test_inference
from fastapi.testclient import TestClient
from app.inference import app

client = TestClient(app)

def test_predict_endpoint():
    # 예시 데이터를 정의합니다.
    data = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3
    }
    
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    json_response = response.json()
    assert "iris_class" in json_response
