# 02.20_머신러닝_AI API

## FastAPI

웹서버 프레임워크로, Django, Flask를 따라잡고 있음. 

비동기 처리 = 특정코드가 오래 걸리면 일단 제껴두고 다른 코드 실행가능 

(ASGI덕분에 비동기처리 가능)

### 서버

FastAPI : 빠르고, 사용하기 쉽고, 자동 문서화까지 해주는 Python 기반 웹 프레임워크

웹 서버 = 우리가 브라우저에서 URL에 접속하면 어떤 데이터를 보내주거나 처리해주는 프로그램

깃헙 코드스페이스  = ***클라우드에서 호스트되는 개발 환경***

@ - 메타데이터 

type hinting

이 코드는 **FastAPI**라는 파이썬 웹 프레임워크를 사용해서 간단한 웹 API 서버를 만드는 코드야.

Server 실행

fastapi dev api

딕셔너리를 반환(기본)하면 JSON으로 응답한다. 

→ 딕셔너리로 반환을 안하면 응답 형태도 바뀐다. (list 등)

API 응답 JSON으로 주실거죠? “네”

API Server

서버의 기능을 외부에서 쓸 수 있도록 노출한다. 

JSON 메시지 형식으로 응답한다. 

***REST : API를 만드는 스타일(형식)***

**조회**: 클라이언트(브라우저) **GET;** 
- 브라우저가 바로 할 수 있음. 쉽게 볼 수 있기 때문에, 데이터 변경 작업에는 적합하지 않음. 

생성: **POST**

- 내 서버에 접속하는 방식

http://localhost:8000

http://127.0.0.1:8000

http://0.0.0.0:80000

port = 하나의 컴퓨터에 항구 여러 개 있는 것처럼, 여러 개의 포트를 만들 수 있음. 한 컴퓨터 안에 여러 개의 서버를 만들 수 있는데, 그걸 구분하는게 port

en0의 inet이 ‘와이파이 IP’ (ex: 지금 내 맥은 집에서 ‘192.168.0.20’)

→ 내 로컬 IP를 확인한 다음

http://192.168.0.20

내 와이파이 망을 벗어나 public 망에서 내 서버를 띄우고 싶다? 

⇒ ngrok https://ngrok.com/

서버에 관한 강의..도 들어야겠다..ㅎ

### AI 서버

모델 임포트

모델 코드를 잘 가져왔다. (클래스 형식으로)

모델 연동

---

GET을 POST로 바꾸면?

GET predict

POST train

포스트 요청을 하는법

1. HTML form 태그

{시스템에 변화를 주는 건 POST, DELETE, PATCH, PUSH}

<aside>
💡

/train 엔드포인트는 POST 요청으로 설정됨. 
브라우저에서 /train 입력해서 접속하면 GET 방식이니까 안 맞음. 

✅ FastAPI 자동 문서 기능 활용
[https://cuddly-train-7w5475jp4j6hw4r7-8000.app.github.dev/docs](https://cuddly-train-7w5475jp4j6hw4r7-8000.app.github.dev/docs)로 들어가면 자동으로 API 테스트할 수 있는 화면 나옴. 

/train 찾아서 클릭

“Try it out” 버튼 클릭

“Execute” 버튼 클릭 → POST 요청이 보내지고, {”result”: “OK”} 가 뜸. 

POST는 주소창에 직접 치는 게 아니고, 데이터 변경할 때 쓰는 메서드임. 

</aside>

<추가 미션>

- [x]  엔드 포인트 추가
- OR, NOT: 선형 경계
- [x]  **모델 저장과 적재 (로딩)**

```python
# ==========
# model.py
# ==========
import numpy as np

class GateModel:
    def __init__(self, gate_type="AND"):
        # 파라메터
        self.gate_type = gate_type.upper()
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

        if self.gate_type == "NOT":
            self.weights = np.random.rand(1) # NOT은 입력이 1개

    def train(self):
        learning_rate = 0.1
        epochs = 20
        if self.gate_type == 'AND':
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([0, 0, 0, 1])
        elif self.gate_type == 'OR':
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([0, 1, 1, 1])
        elif self.gate_type == 'NOT':
            inputs = np.array([[0], [1]])
            outputs = np.array([1, 0])
        else:
            raise ValueError("지원하지 않는 gate_type 입니다. AND, OR, NOT 중 선택해주세요.")
        
        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        input_data = np.array(input_data)
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    

```

```python
# ==========
# main.py
# ==========

from typing import Union
from fastapi import FastAPI

# model.py를 가져온다.
import model
import pickle 

# 그 안에 있는 AndModel 클래스의 인스턴스를 생성한다.
gate_models = {
    'AND': model.GateModel('AND'),
    'OR': model.GateModel('OR'),
    'NOT': model.GateModel('NOT')
}
# API 서버를 생성한다.
app = FastAPI()

# 모델의 학습을 요청한다. 생성 기능은 POST로 한다.
@app.post("/train/{gate_type}")
def train(gate_type: str):
    gate_type = gate_type.upper()
    if gate_type not in gate_models:
        return {"error": "지원하지 않는 gate_type입니다. AND, OR, NOT 중 선택해주세요."}
    
    gate_model = model.GateModel(gate_type)
    gate_model.train()

    with open(f"{gate_type.lower()}_model.pkl", "wb") as file:
        pickle.dump(gate_model, file)

# endpoint 엔드포인트를 선언하며 GET으로 요청을 받고 경로는 /이다.
@app.get("/")
def read_root():
    # 딕셔너리를 반환하면 JSON으로 직렬화된다.
    return {"Hello": "World"}

# 이 엔드포인트의 전체 경로는 /items/{item_id} 이다.
# 중괄호안의 item_id는 경로 매개변수(파라메터)이며 데코레이터 아래 함수의 인수로 쓰인다.
@app.get("/items/{item_id}") 
def read_item(item_id: int):
    return {"item_id": item_id}

# 모델의 예측 기능을 호출한다. 조회 기능은 GET로 한다.
@app.get("/predict/{gate_type}/{left}/{right}") 
def predict(gate_type: str, left: int, right: int):
    gate_type = gate_type.upper()

    # 지원하지 않는 gate_type이 요청되면 에러 반환
    if gate_type not in gate_models:
        return {"error": "지원하지 않는 gate_type입니다. AND, OR, NOT 중 선택해주세요."}
    
    # 모델 로딩
    try:
        with open(f"{gate_type.lower()}_model.pkl", "rb") as file:
            gate_model = pickle.load(file)
    except FileNotFoundError:
        return {"error": f"{gate_type} 모델 파일이 없습니다. 먼저 학습 후 저장해야 합니다."}
    
    # 예측 수행
    result = gate_model.predict([left, right])
    return {"result": result}

@app.get("/predict/NOT/{input_value}")
def predict_not(input_value: int):
    # NOT 모델 로딩
    try:
        with open("not_model.pkl", "rb") as file:
            not_model = pickle.load(file)
    except FileNotFoundError:
        return {"error": "NOT 모델 파일이 없습니다. 먼저 학습 후 저장해야 합니다."}
    
    result = not_model.predict([input_value])
    return {"result": result}

```

저장하는 방법

1. 파이썬 객체를 저장하는 피클 방식
    1. 피클: 파이썬 객체를 직렬화(serialize)하고, 저장하거나 파일에서 로드할 수 있게 해주는 모듈
    2. 장점 - 파이썬 객체 그대로 저장 가능, 데이터 구조와 상태를 그대로 유지할 수 있음. 
    3. 단점 - 보안 문제: 피클 파일을 열 때 악성 코드가 실행될 수 있음. 
2. JSON으로 저장과 로딩
3. 파이토치 방식

모델 파라미터 3개 = 64비트 4바이트: 정수, 부동소수점

파라미터 1개 저장하는데 8바이트 필요. 3개면 24바이트 필요. 

GPT 경우 파라미터 1조개….

1 tril → 트릴리언 일조

1 bil → 빌리언 십억 = 저장

H100 * 8 = 1 server (이래도 모델 하나 못 올림..)

OR NOT

**XOR (파이토치로) 제공 (필수)**

**XOR 모델 저장 - 로딩 (선택)**

개발 모드 = 코드스페이스 닫으면 실행 X

서버를 운영 모드로 크램폴린이나 AWS에 배포

OS, 데이터베이스, 

네트워크