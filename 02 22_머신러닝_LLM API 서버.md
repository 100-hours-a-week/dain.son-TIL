# 02.22_머신러닝_LLM  API 서버

https://brunch.co.kr/@publichr/128

# 준비

OpenAI API Key 발급

Tavily API Key 발급

랭체인

https://python.langchain.com/docs/introduction/#tutorials

선택: 랭스미스 가입

# 기본

https://brunch.co.kr/@publichr/128

주목: 코드로 프롬프팅을 한다. 

```python
pip install langchain
```

환경 변수: 운영체제 수준에서 설정하는 것이다. 

예) 유닉스 쉘(터미널)

export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."

*기본적으로 ‘유닉스 사용법’은 알아야 한다. AI 엔지니어는 기본적으로 백엔드 엔지니어*

하지만 우리는 위의 직접 설정을 하지 않는다(대신 닷인브 라이브러리를 쓴다)

환경설정을 .env 파일에 저장해두고 그 설정을 읽어들인다. 

```python
OPENAI_API_KEY="YOUR_KEY"
TAVILY_API_KEY="Y"
```

닷인브를 설치한다. 

pip install -qU "langchain[openai]”

환경설정을 .env 에 저장해두고 그 설정을 읽어들인다 (이때 쓰는게 ‘**dotenv**’)

pip install dotenv 

```python
from dotenv import load_dotenv

load_dotenv

```

맨 앞이 . 으로 시작하면 → hidden file

.env는 버전 관리 대상에서 제외해야한다.(중요 정보 노출 위험)

.gitignore에 .env를 추가한다.

코랩에서 할 때는 

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

이렇게 해도 됨. 

모델 선택: OpenAI

모델 패키지를 설치한다.

pip install -qU "langchain[openai]”

프롬프트에 메시지 역할이 여러가지 있다. 

<aside>
💡

<Role>

system: 맥락과 명령. 어떻게 행동하고, 추가적인 context(맥락)을 제공해줌. 

- 모든 채팅 모델이 제공하는 건 아님. openAI는 제공함.

user: 실제 내용. 번역할 대상을 user가 입력하는 것. 

assistant: user로부터 어떤 식으로 도움이 왔으면 좋겠다. (ex: 너는 OO 도우미야.)

tool: third party 부를 때 - (예전엔 function이 이 기능을 했음)

LANGCHAIN은 여러 LLM 을 다루기 때문에, 이게 표준이고 모델마다 다를 수 있음. 

</aside>

## 서버(API)화

GET 요청에 부가 정보로 경로 파라미터 외에 쿼리 파라미터를 쓸 수 있다.

쿼리 파라미터: 경로 뒤에 ?key1=value1&key2=value2

예) https://…../say?text=hi

서버 실행을 한다.

fastapi dev [server.py](http://server.py/)

풀스택과 소통할 때 다음과 같이 말한다. 

GET으로 경로는 translate 이고, 쿼리 매개변수는 text와 language로 주세요. 

예제 )[https://solid-xylophone-wrprrgw4j4x3g47q-8000.app.github.dev/translate?text=hi&language=Chinese](https://solid-xylophone-wrprrgw4j4x3g47q-8000.app.github.dev/translate?text=hi&language=Chinese)

스트리밍

SSE: Server-Side Event 웹 기술을 사용하여 이벤트 소스를 클라이언트에서 연결하고 서버는 이벤트 스트림을 내려준다. 

현재 구조: langchain LLM + fastapi 서버

---

```python
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = app_model.AppModel()

@app.get("/say")
def say_app(text: str = Query()):
    response = model.get_response(text)
    return {"content" :response.content}

@app.get("/translate")
def translate(text: str = Query(), language: str = Query()):
    response = model.get_prompt_response(language, text)
    return {"content" :response.content}

@app.get("/says")
def say_app_stream(text: str = Query()):
    def event_stream():
        for message in model.get_streaming_response(text):
            yield f"data: {message.content}\n\n"
            
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

(여기까지 오전 내용)

app.py

```python

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage("Translate the follwing from English into Italian"),
    HumanMessage("hi!")
]

# response = model.invoke(messages)
# print(model.invoke(messages))

# for token in model.stream(messages):
#     print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Korean", "text": "hi!"})
response = model.invoke(prompt)
print(response.content)
```

app_model.py

```python
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class AppModel:
  def __init__(self):
    load_dotenv() 
    self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
    system_template = "Translate the following from English into {language}"
    self.prompt_template = ChatPromptTemplate.from_messages(
      [("system", system_template), ("user", "{text}")]
    )

  def get_response(self, message):
    return self.model.invoke([HumanMessage(message)])

  def get_prompt_response(self, language, message):
    prompt = self.prompt_template.invoke({"language": language, "text": message})
    return self.model.invoke(prompt)

  def get_streaming_response(self, messages):
    return self.model.stream(messages) #asteam은 비동기, stream은 동기
```

---

# 챗봇

랭그래프 사용

https://python.langchain.com/docs/tutorials/chatbot/

메모리 (대화 내용을 기억하면서 소통)

<aside>
💡

길이 제한 문제, token 제한(?) 이해하기

</aside>

```python
# 환경변수 로딩
from dotenv import load_dotenv
load_dotenv()

# 모델 초기화 
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage

# response = model.invoke([HumanMessage(content="Hi! I'm dain")])
# print(response.content) # 항상 반환값을 받아서 print 해주기

# response = model.invoke([HumanMessage(content="What's my name?")])
# print(response.content)

from langchain_core.messages import AIMessage

response = model.invoke(
    [
        HumanMessage(content="Hi! I'm dain"),
        AIMessage(content="Hello dain! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
        AIMessage(content="Your name is Dain. How can I help you today?"),
        HumanMessage(content="openai API 호출해서 LLM 모델을 서비스에 적용하는 과정에 대해 설명해줘. 나는 코드스페이스에서 작업하고 있고, 서버나 http 같은 배경지식이 없어. 쉽고 친절하게 설명해줘." )
    ]
)
print(response.content)
```

- 대화 내용 백업해놓기 - 서버 꺼도 날아가지 않도록

LangGraph는 내장된 지속성 계층을 구현하여 여러 번의 대화 전환을 지원하는 채팅 애플리케이션에 이상적입니다. 최소한의 LangGraph 애플리케이션으로 채팅 모델을 감싸면 메시지 기록을 자동으로 유지하여 다중 전환 애플리케이션 개발을 간소화할 수 있습니다. LangGraph에는 간단한 인메모리 체크포인터가 제공되며, 아래에서 이를 사용합니다. 다른 지속성 백엔드(예: SQLite 또는 Postgres)를 사용하는 방법 등 자세한 내용은 해당 문서를 참조하세요.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

- 새로운 대화 하고 싶으면, thread_id를 다르게 하면 됨

```python
config = {"configurable": {"thread_id": "abc123"}}
```

app = LangGraph application

app.invoke() 하면 그려놓은 Graph 따라서 처리가 되는 것. 

```python
# 환경변수 로딩
from dotenv import load_dotenv
load_dotenv()

# 모델 초기화 
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm dain."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

→ 아까와 달리, 단발성 메시지를 두 번 보내도 LangGraph에 저장돼서 연결된다. 

- 멀티 유저

```python
config = {"configurable": {"thread_id": "abc123"}}

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc234"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

- 커스텀(State) 상태 정의한 코드 + trimmer ⇒ 최종 코드

```python
# 환경변수 로딩
from dotenv import load_dotenv
load_dotenv() 

# 모델 초기화
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 프롬프트 생성: 언어 번역
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

from langchain_core.messages import AIMessage, SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# 커스텀 상태 정의: 언어 입력
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define a new graph(스키마는 우리가 정의한 State)
workflow = StateGraph(state_schema=State)

# Define the function that calls the model. 인수는 우리가 정의한 State
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

<추가미션>

클래스로 만든다. 

API 서버로 만든다. (server.py에 넣으면 됨) → github에 올려줄 예정. 참고하긔

[chatbot.py](http://chatbot.py) → chatbot_model.py (app_model.py 참고) → server.py에서 chatbot_model.py를 써서 GET 엔드포인트 제공

# 에이전트

= 대화만 하는 게 아니라, 더 많은 기능. 

- 검색 기능 추가 (Tavily 활용)

ex) 날씨를 물어보면, LLM 이 날씨를 찾아주기 어려움. ‘검색’을 활용해서 더 많은 기능을 제공할 수 있는 것. 

<추가미션>

클래스로 만든다.

API 서버로 만든다.(일반 GET 요청 응답)

RAG - 마지막 주차 즈음..

# 추후

- 멀티모달

https://livekit.io/