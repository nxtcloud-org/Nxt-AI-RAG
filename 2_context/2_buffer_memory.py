import json
import boto3
import streamlit as st
from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 사이드바 기본값을 접힌 상태로 설정
st.set_page_config(initial_sidebar_state="collapsed")

# AWS Bedrock 클라이언트 초기화
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# LangChain BedrockChat 초기화
bedrock = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리
if "memory" not in st.session_state:
    # 버퍼 메모리 초기화 (모든 대화 기록 유지)
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True, memory_key="history"
    )

# ConversationChain 초기화 (메모리는 세션 상태에서 사용)
conversation = ConversationChain(
    llm=bedrock, memory=st.session_state.memory, verbose=True
)

# Streamlit 앱 설정
st.title("Chatbot Ver.2.1 : 전체 대화 컨텍스트 유지 챗봇")
st.caption("모든 대화 내역을 메모리에 저장하여 장기 컨텍스트를 유지합니다.")

# 사이드바에 메모리 정보 표시
with st.sidebar:
    st.subheader("📊 메모리 통계")
    st.write(f"저장된 대화 수: {len(st.session_state.messages) // 2}")

    # 대화 기록 초기화 버튼
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.success("대화 기록이 초기화되었습니다.")

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

# 메모리 상태 확인
with st.expander("🔍 시스템 상태 확인"):
    st.subheader("메모리 상태")
    st.json(conversation.memory.load_memory_variables({}))

# 푸터 추가
st.markdown("---")
st.caption("© 2024 버퍼 메모리 기반 AI 챗봇 | 모든 대화 내역이 메모리에 저장됩니다.")
