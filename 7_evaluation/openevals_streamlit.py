import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import openevals
import plotly.express as px
import tempfile
import importlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="OpenEvals 평가 대시보드", layout="wide")


# 폰트 설정 함수
def setup_korean_font():
    # matplotlib 차트에서는 한글 폰트를 사용하지 않고 기본 폰트 사용
    try:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        return True
    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
        return False


# 임베딩 기반 유사도 계산 함수
@st.cache_resource
def load_embedding_model():
    """임베딩 모델 로드 (캐시됨)"""
    return SentenceTransformer("all-MiniLM-L6-v2")


def calculate_semantic_similarity(text1, text2, embedding_model):
    """두 텍스트 간의 의미적 유사도 계산"""
    embeddings = embedding_model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return max(0.0, min(1.0, similarity))  # 0-1 범위로 제한


# OpenEvals 평가 함수 (임베딩 + LLM 하이브리드)
def run_openevals_evaluation(eval_data, model_name):
    try:
        # 모델 초기화
        model = ChatAnthropic(
            model_name=model_name, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # 임베딩 모델 로드
        embedding_model = load_embedding_model()
        st.info(
            "임베딩 모델이 로드되었습니다. 의미적 유사도 계산을 포함한 하이브리드 평가를 수행합니다."
        )

        # 다중 메트릭 평가 프롬프트
        evaluation_prompts = {
            "accuracy": """다음은 질문과 응답, 그리고 참조 답변입니다. 응답의 정확성을 0.0에서 1.0 사이의 점수로 평가해주세요.

질문: {question}
응답: {response}
참조: {reference}

정확성 평가 기준:
- 응답이 참조 답변과 얼마나 정확히 일치하는가
- 사실적 오류가 있는가
- 핵심 정보가 올바르게 전달되었는가

0.0에서 1.0 사이의 숫자만 반환해주세요.""",
            "completeness": """다음은 질문과 응답, 그리고 참조 답변입니다. 응답의 완전성을 0.0에서 1.0 사이의 점수로 평가해주세요.

질문: {question}
응답: {response}
참조: {reference}

완전성 평가 기준:
- 질문에 대해 충분히 답변하고 있는가
- 중요한 정보가 누락되지 않았는가
- 참조 답변과 비교했을 때 필요한 내용이 포함되어 있는가

0.0에서 1.0 사이의 숫자만 반환해주세요.""",
            "relevance": """다음은 질문과 응답, 그리고 참조 답변입니다. 응답의 관련성을 0.0에서 1.0 사이의 점수로 평가해주세요.

질문: {question}
응답: {response}
참조: {reference}

관련성 평가 기준:
- 응답이 질문과 얼마나 관련이 있는가
- 불필요한 정보가 포함되어 있지 않은가
- 질문의 의도에 맞게 답변하고 있는가

0.0에서 1.0 사이의 숫자만 반환해주세요.""",
            "clarity": """다음은 질문과 응답, 그리고 참조 답변입니다. 응답의 명확성을 0.0에서 1.0 사이의 점수로 평가해주세요.

질문: {question}
응답: {response}
참조: {reference}

명확성 평가 기준:
- 응답이 이해하기 쉽게 작성되었는가
- 문장 구조가 명확한가
- 모호하거나 혼란스러운 표현이 없는가

0.0에서 1.0 사이의 숫자만 반환해주세요.""",
        }

        # 다중 메트릭 평가 실행 (LLM + 임베딩 하이브리드)
        metrics = [
            "accuracy",
            "completeness",
            "relevance",
            "clarity",
            "semantic_similarity",
        ]
        all_results = []

        for data in eval_data:
            result_dict = {
                "question": data["question"],
                "response": data["response"],
                "reference": data["reference"],
            }

            # 각 메트릭별로 평가 실행
            for metric in metrics:
                if metric == "semantic_similarity":
                    # 임베딩 기반 의미적 유사도 계산
                    similarity_score = calculate_semantic_similarity(
                        data["response"], data["reference"], embedding_model
                    )
                    result_dict[metric] = similarity_score
                else:
                    # LLM 기반 평가
                    judge = openevals.create_llm_as_judge(
                        prompt=evaluation_prompts[metric], judge=model
                    )

                    result = judge(
                        question=data["question"],
                        response=data["response"],
                        reference=data["reference"],
                    )

                    # 점수 추출
                    score_text = result.get("score", "")
                    if isinstance(score_text, str):
                        import re

                        numbers = re.findall(r"0?\.\d+|1\.0|0|1", score_text)
                        if numbers:
                            llm_score = float(numbers[0])
                            llm_score = max(0.0, min(1.0, llm_score))
                        else:
                            llm_score = 0.5
                    else:
                        llm_score = float(score_text)
                        llm_score = max(0.0, min(1.0, llm_score))

                    # 특정 메트릭에 대해 임베딩 유사도와 결합
                    if metric in ["accuracy", "relevance"]:
                        # 임베딩 유사도 계산
                        if metric == "accuracy":
                            embedding_score = calculate_semantic_similarity(
                                data["response"], data["reference"], embedding_model
                            )
                        else:  # relevance
                            embedding_score = calculate_semantic_similarity(
                                data["question"], data["response"], embedding_model
                            )

                        # 하이브리드 점수: LLM 70% + 임베딩 30%
                        hybrid_score = 0.7 * llm_score + 0.3 * embedding_score
                        result_dict[metric] = hybrid_score
                    else:
                        result_dict[metric] = llm_score

            all_results.append(result_dict)

        # 결과 처리 (이미 all_results에 모든 데이터가 포함됨)
        return all_results
    except Exception as e:
        st.error(f"평가 중 오류가 발생했습니다: {str(e)}")
        raise e


# 메인 앱
def main():
    st.title("OpenEvals 평가 대시보드")

    tab1, tab2, tab3 = st.tabs(["평가 결과", "평가 실행", "설정"])

    # 탭 1: 평가 결과 표시
    with tab1:
        st.header("평가 결과")

        # 평가 결과 파일 목록 가져오기
        result_files = [
            f
            for f in os.listdir(".")
            if f.startswith("evaluation_results_openevals_") and f.endswith(".csv")
        ]

        if result_files:
            # 파일명에서 모델 이름 추출
            model_names = [
                f.replace("evaluation_results_openevals_", "").replace(".csv", "")
                for f in result_files
            ]

            # 결과 파일 선택
            if "current_model" in st.session_state:
                default_index = (
                    model_names.index(st.session_state["current_model"])
                    if st.session_state["current_model"] in model_names
                    else 0
                )
            else:
                default_index = 0

            selected_model = st.selectbox("모델 선택", model_names, index=default_index)

            selected_file = f"evaluation_results_openevals_{selected_model}.csv"

            if os.path.exists(selected_file):
                df = pd.read_csv(selected_file)
                # NaN 값을 빈 문자열로 대체
                df = df.fillna("")
                st.success(f"'{selected_model}' 모델의 평가 결과를 표시합니다.")
            else:
                st.warning(
                    "평가 결과 파일이 없습니다. '평가 실행' 탭에서 평가를 실행해 주세요."
                )
                df = None
        else:
            st.warning(
                "평가 결과 파일이 없습니다. '평가 실행' 탭에서 평가를 실행해 주세요."
            )
            df = None

        results_available = df is not None

        if results_available:
            # 결과 개요
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("평균 메트릭 점수")

                # 메트릭 컬럼 확인
                metric_cols = [
                    "accuracy",
                    "completeness",
                    "relevance",
                    "clarity",
                    "semantic_similarity",
                ]
                available_metrics = [col for col in metric_cols if col in df.columns]

                if available_metrics:
                    # 평균 점수 계산
                    avg_scores = df[available_metrics].mean()

                    # 메트릭 이름 매핑
                    metric_names = {
                        "accuracy": "정확성",
                        "completeness": "완전성",
                        "relevance": "관련성",
                        "clarity": "명확성",
                        "semantic_similarity": "의미적 유사도",
                    }

                    # 바 차트
                    fig = px.bar(
                        x=[metric_names.get(m, m) for m in available_metrics],
                        y=avg_scores.values,
                        title=f"Average Metric Scores - {selected_model}",
                        labels={"x": "Metric", "y": "Score"},
                        range_y=[0, 1],
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # 평균 점수 테이블
                    score_df = pd.DataFrame(
                        {
                            "Metric": [
                                metric_names.get(m, m) for m in available_metrics
                            ],
                            "Average Score": avg_scores.values,
                        }
                    )
                    st.table(score_df.set_index("Metric").T)
                else:
                    st.warning("메트릭 데이터를 찾을 수 없습니다.")

            with col2:
                st.subheader("전체 결과")

                if available_metrics:
                    # 히트맵 생성
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # 히트맵 데이터 준비
                    heatmap_data = df[available_metrics].values

                    # 질문 라벨 준비 (영어로 표시)
                    questions = df["question"].tolist()
                    short_questions = [f"Q{i+1}" for i in range(len(questions))]

                    im = ax.imshow(heatmap_data, cmap="Blues", vmin=0, vmax=1)

                    # y 축에 질문 표시
                    ax.set_yticks(np.arange(len(questions)))
                    ax.set_yticklabels(short_questions)

                    # x 축에 메트릭 표시 (영어로)
                    metric_names_en = {
                        "accuracy": "Accuracy",
                        "completeness": "Completeness",
                        "relevance": "Relevance",
                        "clarity": "Clarity",
                        "semantic_similarity": "Semantic Similarity",
                    }
                    ax.set_xticks(np.arange(len(available_metrics)))
                    ax.set_xticklabels(
                        [metric_names_en.get(m, m) for m in available_metrics],
                        rotation=45,
                    )

                    # 색상 바
                    cbar = ax.figure.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")

                    # 각 셀에 값 표시
                    for i in range(len(questions)):
                        for j in range(len(available_metrics)):
                            text = ax.text(
                                j,
                                i,
                                f"{heatmap_data[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="white" if heatmap_data[i, j] > 0.5 else "black",
                            )

                    plt.title("Metric Scores by Question")
                    plt.tight_layout()

                    st.pyplot(fig)
                else:
                    st.warning("히트맵을 생성할 메트릭 데이터가 없습니다.")

            # 상세 결과
            st.subheader("상세 평가 결과")

            # 필요한 열이 있는지 확인
            display_cols = ["question", "response", "reference"] + available_metrics

            # 존재하는 열만 선택
            available_cols = [col for col in display_cols if col in df.columns]

            if available_cols:
                display_df = df[available_cols]

                # 열 이름 매핑
                col_mapping = {
                    "question": "질문",
                    "response": "AI 응답",
                    "reference": "참조 정답",
                    "accuracy": "정확성",
                    "completeness": "완전성",
                    "relevance": "관련성",
                    "clarity": "명확성",
                    "semantic_similarity": "의미적 유사도",
                }

                # 존재하는 열에 대해서만 이름 변경
                rename_mapping = {
                    col: col_mapping.get(col, col) for col in available_cols
                }
                display_df = display_df.rename(columns=rename_mapping)

                # 데이터프레임 표시
                st.dataframe(display_df, use_container_width=True)

                # 질문별 상세 결과 표시
                st.subheader("질문별 상세 결과")
                for i in range(len(display_df)):
                    question_text = str(display_df.iloc[i]["질문"])
                    with st.expander(
                        f"질문 {i+1}: {question_text[:50]}...",
                        expanded=False,
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**질문:**")
                            st.text(display_df.iloc[i]["질문"])

                            st.markdown("**AI 응답:**")
                            st.text(display_df.iloc[i]["AI 응답"])

                        with col2:
                            st.markdown("**참조 정답:**")
                            st.text(display_df.iloc[i]["참조 정답"])

                            # 메트릭 점수들 표시
                            st.markdown("**평가 점수:**")

                            # 메트릭별 점수 표시
                            metric_korean_names = {
                                "정확성": "accuracy",
                                "완전성": "completeness",
                                "관련성": "relevance",
                                "명확성": "clarity",
                                "의미적 유사도": "semantic_similarity",
                            }

                            available_metric_names = [
                                name
                                for name in metric_korean_names.keys()
                                if name in display_df.columns
                            ]

                            if available_metric_names:
                                # 2개씩 묶어서 표시
                                metric_rows = [
                                    available_metric_names[i : i + 2]
                                    for i in range(0, len(available_metric_names), 2)
                                ]

                                for metric_row in metric_rows:
                                    metric_cols = st.columns(len(metric_row))
                                    for j, metric_name in enumerate(metric_row):
                                        value = float(display_df.iloc[i][metric_name])

                                        # 색상 설정
                                        if value >= 0.8:
                                            delta_color = "normal"  # 초록색 (우수)
                                        elif value < 0.6:
                                            delta_color = "inverse"  # 빨간색 (미흡)
                                        else:
                                            delta_color = "off"  # 주황색 (양호)

                                        metric_cols[j].metric(
                                            label=metric_name,
                                            value=f"{value:.2f}",
                                            delta=f"범위: 0-1",
                                            delta_color=delta_color,
                                        )

                                # 평균 점수 계산 및 표시
                                avg_score = sum(
                                    [
                                        float(display_df.iloc[i][name])
                                        for name in available_metric_names
                                    ]
                                ) / len(available_metric_names)
                                st.markdown("---")

                                # 평균 점수 등급
                                if avg_score >= 0.8:
                                    grade = "우수"
                                    grade_color = "normal"
                                elif avg_score >= 0.6:
                                    grade = "양호"
                                    grade_color = "off"
                                elif avg_score >= 0.4:
                                    grade = "보통"
                                    grade_color = "off"
                                elif avg_score >= 0.2:
                                    grade = "미흡"
                                    grade_color = "inverse"
                                else:
                                    grade = "부적절"
                                    grade_color = "inverse"

                                st.metric(
                                    label="종합 점수",
                                    value=f"{avg_score:.2f}",
                                    delta=f"{grade} ({len(available_metric_names)}개 메트릭 평균)",
                                    delta_color=grade_color,
                                )
                            else:
                                st.warning("평가 점수 데이터가 없습니다.")

    # 탭 2: 평가 실행
    with tab2:
        st.header("OpenEvals 평가 실행")

        # 사용 가능한 모델 목록
        AVAILABLE_MODELS = {
            "Claude 3 Haiku": "claude-3-haiku-20240307",
            "Claude 3 Sonnet": "claude-3-sonnet-20240229",
            "Claude 3 Opus": "claude-3-opus-20240229",
        }

        # 모델 선택
        selected_model = st.selectbox(
            "평가에 사용할 모델을 선택하세요", list(AVAILABLE_MODELS.keys())
        )

        # 테스트 데이터 입력
        with st.expander("테스트 데이터", expanded=not results_available):
            # 기본 샘플 데이터
            default_data = {
                "question": [
                    "한국의 수도는 어디인가요?",
                    "인공지능의 종류는 무엇이 있나요?",
                    "최근 5년간 한국 경제 성장률은 어떻게 변했나요?",
                    "영화 '기생충'의 감독은 누구인가요?",
                    "2023년 미국 주식 시장은 어땠나요?",
                ],
                "response": [
                    "한국의 수도는 서울입니다.",
                    "인공지능은 크게 강인공지능과 약인공지능으로 나눌 수 있습니다. 현재 개발된 AI는 대부분 약인공지능에 속합니다.",
                    "한국의 경제 성장률은 2019년 2.0%, 2020년 -1.0%, 2021년 4.0%, 2022년 2.5%를 기록했습니다. 2023년은 아직 집계되지 않았습니다.",
                    "영화 '기생충'은 봉준호 감독이 연출했습니다.",
                    "2023년 미국 주식 시장은 상승세를 보였습니다. 하지만 금리 인상 등의 영향으로 변동성이 큰 한 해였습니다.",
                ],
                "reference": [
                    "대한민국의 수도는 서울특별시이다.",
                    "인공지능은 목표나 능력에 따라 강인공지능과 약인공지능으로 분류할 수 있다.",
                    "한국은행 경제통계시스템에 따르면, 한국의 경제 성장률은 2019년 2.0%, 2020년 -1.0%, 2021년 4.0%, 2022년 2.5%를 기록했다.",
                    "영화 '기생충'은 봉준호 감독이 각본과 연출을 맡았다.",
                    "2023년 미국 주식 시장은 S&P 500 기준으로 20% 이상 상승했다.",
                ],
            }

            st.subheader("테스트 데이터 입력")
            st.info(
                "기본값이 제공된 샘플 데이터를 수정하거나 그대로 사용할 수 있습니다."
            )

            # 데이터 초기화 및 불러오기 버튼
            button_container = st.container()
            with button_container:
                col1, col2 = st.columns([1, 1], gap="small")
                with col1:
                    if st.button(
                        "전체 데이터 지우기", type="secondary", use_container_width=True
                    ):
                        # 현재 항목 수 저장
                        current_n_items = st.session_state.get("n_items", 5)
                        # session_state 완전 초기화
                        st.session_state.clear()
                        # 필요한 상태만 다시 설정
                        st.session_state["n_items"] = current_n_items
                        st.session_state["clear_data"] = True
                        # 모든 입력 필드를 빈 값으로 초기화
                        for i in range(10):  # 최대 가능한 항목 수
                            st.session_state[f"q_{i}"] = ""
                            st.session_state[f"r_{i}"] = ""
                            st.session_state[f"ref_{i}"] = ""
                        st.rerun()
                with col2:
                    if st.button(
                        "기본 테스트 데이터 불러오기",
                        type="secondary",
                        use_container_width=True,
                    ):
                        # 현재 항목 수 저장
                        current_n_items = st.session_state.get("n_items", 5)
                        # session_state 완전 초기화
                        st.session_state.clear()
                        # 필요한 상태만 다시 설정
                        st.session_state["n_items"] = current_n_items
                        st.session_state["clear_data"] = False
                        st.rerun()

            # 여러 항목 입력을 위한 컨테이너
            if "n_items" not in st.session_state:
                st.session_state["n_items"] = 5

            n_items = st.number_input(
                "테스트 항목 수",
                min_value=1,
                max_value=10,
                value=st.session_state["n_items"],
            )
            if n_items != st.session_state["n_items"]:
                st.session_state["n_items"] = n_items
                st.rerun()

            test_data = {"question": [], "response": [], "reference": []}

            for i in range(n_items):
                st.markdown(f"### 테스트 항목 {i+1}")
                col1, col2 = st.columns(2)

                # 각 필드의 키 생성
                q_key = f"q_{i}"
                r_key = f"r_{i}"
                ref_key = f"ref_{i}"

                if "clear_data" in st.session_state and st.session_state["clear_data"]:
                    # 전체 데이터 지우기가 활성화된 경우
                    st.session_state[q_key] = ""
                    st.session_state[r_key] = ""
                    st.session_state[ref_key] = ""
                    default_q = ""
                    default_r = ""
                    default_ref = ""
                else:
                    # 기본 데이터 사용
                    default_q = (
                        default_data["question"][i]
                        if i < len(default_data["question"])
                        else ""
                    )
                    default_r = (
                        default_data["response"][i]
                        if i < len(default_data["response"])
                        else ""
                    )
                    default_ref = (
                        default_data["reference"][i]
                        if i < len(default_data["reference"])
                        else ""
                    )

                with col1:
                    q = st.text_area(f"질문 {i+1}", value=default_q, key=q_key)
                    r = st.text_area(f"응답 {i+1}", value=default_r, key=r_key)

                with col2:
                    ref = st.text_area(
                        f"참조 정답 {i+1}", value=default_ref, key=ref_key
                    )

                if q and r and ref:
                    test_data["question"].append(q)
                    test_data["response"].append(r)
                    test_data["reference"].append(ref)

            if len(test_data["question"]) > 0:
                st.success(f"{len(test_data['question'])}개 항목이 입력되었습니다.")
            else:
                st.warning("유효한 테스트 항목이 없습니다. 모든 필드를 입력해주세요.")

        # 평가 실행 버튼
        if st.button("평가 실행", type="primary"):
            with st.spinner("평가를 실행 중입니다..."):
                try:
                    # 평가 데이터 준비
                    eval_data = []
                    for q, r, ref in zip(
                        test_data["question"],
                        test_data["response"],
                        test_data["reference"],
                    ):
                        eval_data.append(
                            {"question": q, "response": r, "reference": ref}
                        )

                    # 평가 실행
                    results = run_openevals_evaluation(
                        eval_data=eval_data, model_name=AVAILABLE_MODELS[selected_model]
                    )

                    if results:
                        # 결과를 DataFrame으로 변환
                        df = pd.DataFrame(results)

                        # 모델 정보 추가
                        df["model"] = selected_model

                        # 결과 파일명 생성
                        result_file = f"evaluation_results_openevals_{selected_model.lower().replace(' ', '_')}.csv"

                        # 결과 저장
                        df.to_csv(result_file, index=False)

                        # 모델 정보를 세션 상태에 저장
                        st.session_state["current_model"] = selected_model
                        st.session_state["current_results_file"] = result_file

                        st.success(
                            f"평가 완료! '{selected_model}' 모델 평가 결과가 저장되었습니다. '평가 결과' 탭에서 결과를 확인하세요."
                        )
                        st.rerun()
                    else:
                        st.error("평가 결과가 없습니다.")
                except Exception as e:
                    st.error(f"평가 중 오류가 발생했습니다: {str(e)}")

    # 탭 3: 설정
    with tab3:
        st.header("설정")

        # 평가 기준 설명
        with st.expander("OpenEvals 평가 기준 설명", expanded=True):
            st.markdown(
                """
            ### OpenEvals 하이브리드 다중 메트릭 평가 설명
            
            * **5가지 평가 메트릭**: 모델의 응답을 다음 5개 기준으로 각각 0.0-1.0 점수로 평가합니다.
              - **정확성(Accuracy)**: 응답이 참조 답변과 얼마나 정확히 일치하는가 (LLM + 임베딩 하이브리드)
              - **완전성(Completeness)**: 응답이 질문에 대해 충분히 답변하고 있는가 (LLM 기반)
              - **관련성(Relevance)**: 응답이 질문과 얼마나 관련이 있는가 (LLM + 임베딩 하이브리드)
              - **명확성(Clarity)**: 응답이 이해하기 쉽고 명확하게 작성되었는가 (LLM 기반)
              - **의미적 유사도(Semantic Similarity)**: 응답과 참조 답변 간의 의미적 유사도 (임베딩 기반)
            
            * **하이브리드 평가 방식**:
              - **LLM 평가**: Claude 모델이 텍스트를 직접 분석하여 점수 산출
              - **임베딩 평가**: SentenceTransformer로 의미적 유사도 계산
              - **하이브리드 점수**: 정확성/관련성은 LLM 70% + 임베딩 30% 가중평균
            
            * **점수 해석**:
              - **0.8-1.0**: 우수한 답변 (대부분 정확하고 완전함)
              - **0.6-0.7**: 양호한 답변 (부분적으로 정확하거나 불완전함)
              - **0.4-0.5**: 보통 답변 (일부 정확하지만 상당한 오류나 누락)
              - **0.2-0.3**: 미흡한 답변 (대부분 부정확하거나 관련성 낮음)
              - **0.0-0.1**: 부적절한 답변 (완전히 부정확하거나 관련 없음)
            
            * **평가 모델**:
              - LLM: Claude 3 Haiku/Sonnet/Opus
              - 임베딩: SentenceTransformer (all-MiniLM-L6-v2)
              - 종합 점수: 5개 메트릭의 평균값
            """
            )

        # 언어 설정
        st.subheader("언어 설정")
        lang_option = st.radio("언어", ["한국어", "English"], index=0)

        # 폰트 설정
        st.subheader("폰트 설정")
        if st.button("한글 폰트 설정"):
            has_font = setup_korean_font()
            if has_font:
                st.success("한글 폰트 설정이 완료되었습니다!")
            else:
                st.error("한글 폰트 설정에 실패했습니다.")

        # 앱 정보
        st.subheader("앱 정보")
        st.info(
            """
        **OpenEvals 평가 대시보드**
        
        이 애플리케이션은 OpenEvals를 사용하여 
        질의응답 시스템의 성능을 평가하고 시각화합니다.
        
        GitHub: [OpenEvals](https://github.com/openevals/openevals)
        """
        )


if __name__ == "__main__":
    setup_korean_font()
    main()
