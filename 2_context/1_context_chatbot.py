import json  # JSON 데이터 처리를 위한 라이브러리
import boto3  # AWS SDK for Python
import streamlit as st  # 웹 애플리케이션 구축을 위한 Streamlit 라이브러리

# AWS Bedrock 클라이언트 초기화
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Streamlit 웹 애플리케이션의 제목 설정
st.title("Chatbot Ver.2 : 대화 맥락 이해 챗봇")

# 세션 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리를 저장할 리스트 초기화

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "input_tokens": 0,  # 누적 입력 토큰 수
        "output_tokens": 0,  # 누적 출력 토큰 수
        "total_tokens": 0,  # 누적 통합 토큰 수
        "last_input_tokens": 0,  # 최근 입력 토큰 수
        "last_output_tokens": 0,  # 최근 출력 토큰 수
        "last_total_tokens": 0,  # 최근 통합 토큰 수
    }


def get_response_from_bedrock(messages):
    """
    Bedrock 모델에 대화 히스토리를 기반으로 요청을 보내고 응답을 받는 함수
    :param messages: 대화 히스토리 (role, content 형태의 리스트)
    :return: 모델의 응답 텍스트
    """
    try:
        # Bedrock 모델에 요청할 body 생성
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",  # 모델 버전
                "max_tokens": 1000,  # 응답의 최대 토큰 수
                "messages": messages,  # 대화 히스토리 전달
            }
        )

        # Bedrock 모델 호출
        response = bedrock_runtime.invoke_model(
            modelId=FILL_ME_IN,
            body=body,  # 요청 본문
        )
        response_body = json.loads(response.get("body").read())  # 응답 본문 파싱

        ### AI 응답 데이터의 형식과 내용을 확인하세요. ###
        ### 응답 데이터를 참고하여 아래 추출 데이터를 적절하게 변수에 할당하세요.

        # 모델 출력 추출
        output_text = FILL_ME_IN

        # 토큰 사용량 추출
        input_tokens = FILL_ME_IN  # 입력 토큰 수 추출
        output_tokens = FILL_ME_IN  # 출력 토큰 수 추출
        total_tokens = input_tokens + output_tokens  # 입출력 토큰 수 계산

        # 세션 상태에 토큰 사용량 업데이트

        ## 최신 토큰 기록
        st.session_state.token_usage["last_input_tokens"] = input_tokens
        st.session_state.token_usage["last_output_tokens"] = output_tokens
        st.session_state.token_usage["last_total_tokens"] = total_tokens

        ## 누적 토큰 기록
        st.session_state.token_usage["input_tokens"] += input_tokens
        st.session_state.token_usage["output_tokens"] += output_tokens
        st.session_state.token_usage["total_tokens"] += total_tokens

        return output_text
    except Exception as e:
        # 오류 발생 시 사용자에게 알림
        st.error(f"Bedrock과 통신 중 에러 발생: {str(e)}")
        return "응답 생성 중 에러 발생"


# 화면 상단에 st.metric으로 토큰 사용량 표시
header, button = st.columns(2)
with header:
    st.subheader("토큰 사용량 확인", divider="rainbow")
with button:
    if st.button("토큰 사용량 새로고침"):
        st.rerun()

current, total = st.tabs(["최신", "누적"])
with current:
    input_token, output_token, all_token = st.columns(3)
    with input_token:
        st.metric(
            label="최근 입력 토큰 수",
            value=st.session_state.token_usage["last_input_tokens"],
        )
    with output_token:
        st.metric(
            label="최근 출력 토큰 수",
            value=st.session_state.token_usage["last_output_tokens"],
        )
    with all_token:
        st.metric(
            label="최근 입출력 토큰 수",
            value=st.session_state.token_usage["last_total_tokens"],
        )
with total:
    total_input_token, total_output_token, total_all_token = st.columns(3)
    with total_input_token:
        st.metric(
            label="누적 입력 토큰 수",
            value=st.session_state.token_usage["input_tokens"],
        )
    with total_output_token:
        st.metric(
            label="누적 출력 토큰 수",
            value=st.session_state.token_usage["output_tokens"],
        )
    with total_all_token:
        st.metric(
            label="누적 입출력 토큰 수",
            value=st.session_state.token_usage["total_tokens"],
        )

st.divider()
st.subheader("챗봇 🤖 💬", divider="rainbow")

# 저장된 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # 메시지 역할에 따른 채팅 버블 생성
        st.markdown(message["content"])  # 메시지 내용을 마크다운으로 렌더링

# 사용자 입력 처리
if prompt := st.chat_input("Message Bedrock..."):  # 사용자 입력을 받음
    # 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 사용자 메시지를 화면에 표시
    with st.chat_message("user"):
        st.markdown(prompt)  # 입력 메시지를 마크다운 형식으로 렌더링

    # Bedrock에 대화 히스토리를 전달하여 응답 생성
    with st.chat_message("assistant"):
        response = get_response_from_bedrock(
            st.session_state.messages
        )  # 전체 히스토리 전달
        st.markdown(response)  # 모델 응답을 화면에 표시

    # 모델의 응답을 대화 히스토리에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
