import boto3
import streamlit as st
from langchain_aws import BedrockEmbeddings
import numpy as np
import pandas as pd

# 사이드바 자동 숨김 설정
st.set_page_config(initial_sidebar_state="collapsed")

st.title("📊 임베딩 개념 이해하기")

# 임베딩 개념 설명 섹션
st.header("0️⃣ 임베딩 개념 이해")

with st.expander("임베딩이란?", expanded=False):
    st.markdown(
        """
   임베딩은 텍스트, 이미지 등의 데이터를 벡터(숫자 배열)로 변환하는 과정입니다.
   아래 예시를 통해 임베딩의 개념을 이해해보세요.
   """
    )

    st.subheader("간단한 임베딩 예시: 창작자와 소비자 속성 분류")

    # 데이터프레임 생성
    data = {
        "역할": ["작가", "화가", "유튜버", "독자", "뷰어", "시청자"],
        "글쓰기": [1, 0, 0, 1, 0, 0],
        "그리기": [0, 1, 0, 0, 1, 0],
        "영상제작": [0, 0, 1, 0, 0, 1],
        "창작자": [1, 1, 1, 0, 0, 0],
        "소비자": [0, 0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.markdown(
        """
   ### 벡터 연산 예시
   1. **독자에서 작가로 전환:**
       * 독자(1,0,0,0,1) - 소비자(0,0,0,0,1) + 창작자(0,0,0,1,0) = 작가(1,0,0,1,0)
       
   2. **유튜버에서 시청자로 전환:**
       * 유튜버(0,0,1,1,0) - 창작자(0,0,0,1,0) + 소비자(0,0,0,0,1) = 시청자(0,0,1,0,1)
   """
    )


# AWS Bedrock 클라이언트 초기화
@st.cache_resource
def init_bedrock():
    embeddings = BedrockEmbeddings(region_name="us-east-1")
    return embeddings


embeddings = init_bedrock()

# 텍스트 입력 섹션
st.header("1️⃣ 텍스트 임베딩 생성")
text_input = st.text_area(
    "텍스트를 입력하세요:", value="cloud", height=100, key="text_input"
)

# 임베딩 생성 및 시각화
if st.button("임베딩 생성", key="embed_button", type="primary"):
    with st.spinner("임베딩을 생성중입니다..."):
        vector = embeddings.embed_query(text_input)

        # 기본 정보 표시
        st.subheader("임베딩 정보")

        with st.expander("벡터의 차원과 norm 이해하기", expanded=False):
            st.markdown(
                """
            ### 벡터 차원 (Vector Dimension)
            벡터 차원은 하나의 데이터를 표현하기 위해 사용되는 숫자의 개수입니다.
            - 예: [0.1, -0.5, 0.8]은 3차원 벡터
            - 텍스트 임베딩에서는 보통 수백~수천 차원 사용
            - 차원이 높을수록 더 풍부한 정보 표현 가능
            
            ### 벡터 norm (Vector Norm)
            벡터 norm은 벡터의 '길이' 또는 '크기'를 나타내는 단일 숫자입니다.
            - L2 norm(유클리드 norm)을 주로 사용
            - 벡터의 각 요소를 제곱하여 더한 후 제곱근을 구함
            - 수식: ||v|| = √(v₁² + v₂² + ... + vₙ²)
            
            #### 예시
            벡터 v = [3, 4]의 경우:
            - 차원 = 2 (숫자가 2개)
            - norm = √(3² + 4²) = √(9 + 16) = √25 = 5
            """
            )

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"벡터 차원: {len(vector)}")
        with col2:
            st.info(f"벡터 norm: {np.linalg.norm(vector):.4f}")

        # 벡터 시각화
        st.subheader("벡터 시각화")
        df = pd.DataFrame({"차원": range(len(vector)), "값": vector})
        st.line_chart(df.set_index("차원"))

        # 처음 10개 값 표시
        st.subheader("벡터 값 샘플 (처음 10개)")
        st.write(vector[:10])

# 유사도 비교 섹션
st.header("2️⃣ 텍스트 유사도 비교")
# 코사인 유사도 설명 섹션
st.header("코사인 유사도 이해하기")
with st.expander("코사인 유사도란?", expanded=False):
    st.markdown(
        """
    ### 코사인 유사도 (Cosine Similarity)
    
    코사인 유사도는 두 벡터 간의 각도를 이용해 유사도를 측정하는 방법입니다.
    
    #### 핵심 특징
    - 결과값 범위: -1 ~ 1
    - 1에 가까울수록: 매우 유사
    - 0에 가까울수록: 관계 없음
    - -1에 가까울수록: 반대 의미
    
    #### 수식
    ```
    cosine_similarity = (A·B) / (||A|| ||B||)
    ```
    - A·B: 두 벡터의 내적
    - ||A||, ||B||: 각 벡터의 크기
    
    #### 유사도 해석 가이드
    - 0.9 이상: 거의 동일한 의미
    - 0.7 ~ 0.9: 매우 유사한 의미
    - 0.5 ~ 0.7: 어느 정도 관련있는 의미
    - 0.3 ~ 0.5: 약간 관련있는 의미
    - 0.3 미만: 거의 관련 없는 의미
    """
    )

    # 간단한 2D 예시
    st.markdown(
        """
    #### 간단한 예시
    - "강아지는 귀여운 동물입니다." vs "강아지는 귀여운 반려동물입니다."
        - 높은 유사도 (약 0.9 이상)
    """
    )

col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area(
        "첫 번째 텍스트:", value="고양이는 귀여운 동물입니다.", key="text1"
    )
with col2:
    text2 = st.text_area(
        "두 번째 텍스트:", value="강아지는 귀여운 동물입니다.", key="text2"
    )

if st.button("유사도 계산", key="similarity_button", type="primary"):
    with st.spinner("유사도를 계산중입니다..."):
        embedding1 = embeddings.embed_query(text1)
        embedding2 = embeddings.embed_query(text2)

        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        # 결과 표시 개선
        st.info(f"코사인 유사도: {similarity:.4f}")
        st.progress(float(similarity))

        # 유사도 해석 추가
        if similarity > 0.9:
            st.success("🌟 두 텍스트가 거의 동일한 의미를 가지고 있습니다.")
        elif similarity > 0.7:
            st.success("✨ 두 텍스트가 매우 유사한 의미를 가지고 있습니다.")
        elif similarity > 0.5:
            st.warning("📝 두 텍스트가 어느 정도 관련이 있습니다.")
        elif similarity > 0.3:
            st.warning("🤔 두 텍스트가 약간의 관련성이 있습니다.")
        else:
            st.error("❌ 두 텍스트는 거의 관련이 없습니다.")
