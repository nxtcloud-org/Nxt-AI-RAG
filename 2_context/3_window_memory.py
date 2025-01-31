import json
import boto3
import streamlit as st
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# AWS Bedrock 클라이언트 초기화
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# LangChain BedrockChat 초기화
bedrock = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
)

# WindowMemory 초기화 (최근 3개의 대화만 유지)
memory = ConversationBufferWindowMemory(
    k=3,  # 최근 3개의 대화만 유지
    return_messages=True,
    memory_key="history",  # chat_history에서 history로 변경
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = memory

# ConversationChain 초기화
conversation = ConversationChain(
    llm=bedrock, memory=st.session_state.memory, verbose=True
)

# Streamlit 앱 설정
st.title("Chatbot Ver.2.2 : 효율적인 메모리 관리 챗봇")
st.caption("최근 3개의 대화 컨텍스트만 유지하여 토큰 사용량을 최적화합니다.")

# 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("Message Bedrock..."):
    # 사용자 입력 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LangChain 대화 체인 실행
    with st.chat_message("assistant"):
        with st.spinner("AI가 답변을 고민 중입니다..."):
            response = conversation.run(input=prompt)
            st.markdown(response)

    # 모델 응답 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

# 메모리 상태 확인 (디버깅용)
with st.expander("현재 메모리 상태 확인"):
    st.write(conversation.memory.load_memory_variables({}))
