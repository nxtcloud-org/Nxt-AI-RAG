import json  # JSON 데이터 처리를 위한 라이브러리
import boto3  # AWS SDK for Python
import streamlit as st  # 웹 애플리케이션 구축을 위한 Streamlit 라이브러리

# AWS Bedrock 클라이언트 초기화
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# 스트림릿 앱 제목 설정
st.title("Chatbot Ver.1 : 단순 챗봇")

# 세션 상태 관리 - 대화 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # 사용자와 어시스턴트 간 대화 기록을 저장

# 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # 메시지 역할에 따라 채팅 버블 생성
        st.markdown(message["content"])  # 메시지 내용을 화면에 렌더링


def get_response_from_bedrock(prompt):
    """
    Bedrock 모델에 요청을 보내고 응답을 받는 함수
    :param prompt: 사용자 입력 메시지
    :return: 모델의 응답 텍스트
    """
    try:
        # Bedrock에 전달할 요청 본문 생성
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",  # 모델 버전
                "max_tokens": 1000,  # 최대 토큰 수
                "messages": [
                    {
                        "role": "user",  # 사용자 역할로 설정
                        "content": [{"type": "text", "text": prompt}],  # 입력 텍스트
                    }
                ],
            }
        )

        # Bedrock 모델 호출
        response = bedrock_runtime.invoke_model(
            modelId=FILL_ME_IN,  # 사용할 모델 ID
            body=body,  # 요청 본문 전달
        )
        response_body = json.loads(response.get("body").read())  # 응답 본문 JSON 파싱

        ### AI 응답 데이터의 형식과 내용을 확인하세요. ###
        ### 응답 데이터를 참고하여 아래 추출 데이터를 적절하게 변수에 할당하세요.

        # 모델의 응답 텍스트 추출
        output_text = FILL_ME_IN

        return output_text
    except Exception as e:
        # 에러 발생 시 사용자에게 메시지 표시
        st.error(f"Bedrock과 통신 중 오류가 발생했습니다: {str(e)}")
        return "응답 생성 중 오류가 발생했습니다."


# 사용자 입력 처리
if prompt := st.chat_input("Bedrock에 메시지 입력..."):  # 입력 창에 메시지 입력
    # 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 사용자 메시지를 화면에 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bedrock 모델에 요청하여 응답 생성
    with st.chat_message("assistant"):

        ### 대기 스피너를 추가해보세요. ###

        response = get_response_from_bedrock(
            prompt
        )  # 사용자 메시지를 전달하여 응답 생성
        st.markdown(response)  # 모델 응답을 화면에 표시

    # 어시스턴트의 응답을 세션 상태에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
