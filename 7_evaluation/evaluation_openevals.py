import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import plotly.express as px
from langchain_anthropic import ChatAnthropic
import openevals

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="OpenEvals 평가 대시보드", layout="wide")

# 사용 가능한 모델 목록
AVAILABLE_MODELS = {
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Claude 3 Opus": "claude-3-opus-20240229",
}

# 평가 프롬프트
EVALUATION_PROMPT = """다음은 질문과 응답, 그리고 참조 답변입니다. 응답이 참조 답변과 일치하는지 평가해주세요.

질문: {question}
응답: {response}
참조: {reference}

응답이 참조 답변과 일치하면 'true', 일치하지 않으면 'false'를 반환해주세요."""

# 테스트 데이터
TEST_DATA = {
    "questions": [
        "대한민국의 수도는 어디인가요?",
        "인공지능의 주요 유형은 무엇인가요?",
        "최근 5년간 한국의 경제 성장률은 어떻게 되나요?",
    ],
    "responses": [
        "대한민국의 수도는 서울입니다.",
        "인공지능은 크게 강인공지능(Strong AI)과 약인공지능(Weak AI)으로 나눌 수 있습니다.",
        "최근 5년간 한국의 경제 성장률은 2020년 -0.9%, 2021년 4.1%, 2022년 2.6%, 2023년 1.4%, 2024년 2.2%입니다.",
    ],
    "references": [
        "대한민국의 수도는 서울입니다.",
        "인공지능은 강인공지능(Strong AI)과 약인공지능(Weak AI)으로 구분됩니다.",
        "최근 5년간 한국의 경제 성장률은 2020년 -0.9%, 2021년 4.1%, 2022년 2.6%, 2023년 1.4%, 2024년 2.2%입니다.",
    ],
}


def run_openevals_evaluation(questions, responses, references, model_name):
    """OpenEvals를 사용하여 평가를 실행합니다."""
    try:
        # 모델 초기화
        model = ChatAnthropic(
            model_name=model_name, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # 평가 데이터 준비
        eval_data = []
        for q, r, ref in zip(questions, responses, references):
            eval_data.append({"question": q, "response": r, "reference": ref})

        # 평가 실행
        judge = openevals.create_llm_as_judge(prompt=EVALUATION_PROMPT, judge=model)

        results = []
        for data in eval_data:
            result = judge(
                question=data["question"],
                response=data["response"],
                reference=data["reference"],
            )
            results.append(result)

        # 결과 처리
        processed_results = []
        for result in results:
            try:
                # 문자열에서 숫자만 추출
                score_text = result.get("score", "")
                if isinstance(score_text, str):
                    # 'true' 또는 'false' 문자열을 1 또는 0으로 변환
                    score = 1 if "true" in score_text.lower() else 0
                else:
                    score = float(score_text)

                processed_results.append(
                    {
                        "question": result.get("question", ""),
                        "response": result.get("response", ""),
                        "reference": result.get("reference", ""),
                        "score": score,
                    }
                )
            except (ValueError, TypeError) as e:
                print(f"결과 처리 중 오류 발생: {e}")
                print(f"문제가 된 결과: {result}")
                continue

        return processed_results
    except Exception as e:
        print(f"평가 실행 중 오류 발생: {e}")
        return []


def main():
    st.title("OpenEvals 평가 대시보드")

    # 사이드바에 모델 정보 표시
    with st.sidebar:
        st.header("모델 선택")
        selected_model = st.selectbox(
            "평가에 사용할 모델을 선택하세요", list(AVAILABLE_MODELS.keys())
        )

        st.header("평가 항목")
        st.write(
            """
        - 정확성: 응답이 참조 답변과 일치하는지 평가
        - 평가 기준: true/false
        - 평가 모델: 선택한 Claude 모델
        """
        )

    # 메인 영역
    st.header("평가 데이터")

    # 테스트 데이터 표시
    st.subheader("테스트 데이터 미리보기")
    preview_data = pd.DataFrame(
        {
            "질문": TEST_DATA["questions"],
            "응답": TEST_DATA["responses"],
            "참조": TEST_DATA["references"],
        }
    )
    st.dataframe(preview_data)

    # 평가 실행
    if st.button("평가 실행", type="primary"):
        with st.spinner("평가를 실행 중입니다..."):
            results = run_openevals_evaluation(
                questions=TEST_DATA["questions"],
                responses=TEST_DATA["responses"],
                references=TEST_DATA["references"],
                model_name=AVAILABLE_MODELS[selected_model],
            )

            if results:
                # 결과를 DataFrame으로 변환
                df = pd.DataFrame(results)

                # 평균 점수 계산 (숫자형 데이터만)
                if "score" in df.columns:
                    avg_score = df["score"].mean()

                    # 결과 저장
                    result_file = f'evaluation_results_openevals_{selected_model.lower().replace(" ", "_")}.csv'
                    df.to_csv(result_file, index=False)

                    # 결과 표시
                    st.header("평가 결과")

                    # 평균 점수 시각화
                    st.subheader(f"평균 정확성 점수 ({selected_model})")
                    fig = px.bar(
                        x=["정확성"],
                        y=[avg_score],
                        title=f"평균 정확성 점수 - {selected_model}",
                        labels={"x": "평가 항목", "y": "점수"},
                        range_y=[0, 1],
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # 상세 결과 표시
                    st.subheader("상세 평가 결과")
                    st.dataframe(
                        df.style.format({"score": "{:.2f}"}), use_container_width=True
                    )

                    # 결과 다운로드 버튼
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="결과 다운로드",
                        data=csv,
                        file_name=result_file,
                        mime="text/csv",
                    )
                else:
                    st.error("평가 결과에서 점수 데이터를 찾을 수 없습니다.")
            else:
                st.error("평가 결과가 없습니다.")


if __name__ == "__main__":
    main()
