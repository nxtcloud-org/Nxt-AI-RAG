import json
import boto3
import streamlit as st
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# AWS Bedrock 클라이언트 초기화
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# LangChain BedrockChat 초기화
bedrock = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
)

memory = ConversationBufferMemory(return_messages=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리
if "memory" not in st.session_state:
    st.session_state.memory = memory

# ConversationChain 초기화 (메모리는 세션 상태에서 사용)
conversation = ConversationChain(llm=bedrock, memory=st.session_state.memory)

# Streamlit 앱 설정
st.title("Chatbot Ver.2.1 : LangChain 기반 대화 맥락 이해 챗봇")

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
