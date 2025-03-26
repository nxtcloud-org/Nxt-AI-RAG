import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import boto3
from datasets import Dataset
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy
import os
import tempfile
import importlib

# 페이지 설정
st.set_page_config(page_title="RAGAS 평가 대시보드", layout="wide")

# 폰트 설정 함수
def setup_korean_font():
    font_path = "./NanumGothic.ttf"
    
    # 폰트 파일이 없으면 다운로드
    if not os.path.exists(font_path):
        import requests
        font_url = "https://github.com/naver/nanumfont/blob/master/NanumGothic.ttf?raw=true"
        st.info("한글 폰트를 다운로드 중입니다...")
        response = requests.get(font_url)
        with open(font_path, 'wb') as f:
            f.write(response.content)
        st.success("폰트 다운로드 완료!")

    if os.path.exists(font_path):
        # 폰트 등록
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.sans-serif"] = ["NanumGothic"]
        return True
    else:
        st.error("한글 폰트를 사용할 수 없습니다. 기본 폰트를 사용합니다.")
        return False

# RAGAS 평가 함수
def run_ragas_evaluation(eval_dataset, bedrock_model=None, bedrock_embeddings=None):
    try:
        # RAGAS 라이브러리에서 evaluate 함수 가져오기
        # 순환 참조 오류를 방지하기 위해 동적 임포트 사용
        ragas = importlib.import_module('ragas')
        evaluate = getattr(ragas, 'evaluate')
        
        # 평가 메트릭 정의
        metrics = [Faithfulness(), ContextRecall(), ContextPrecision(), AnswerRelevancy()]
        
        # 평가 실행
        result = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=bedrock_model,
            embeddings=bedrock_embeddings,
        )
        
        return result
    except Exception as e:
        st.error(f"평가 중 오류가 발생했습니다: {str(e)}")
        raise e

# 메인 앱
def main():
    st.title("RAGAS 평가 대시보드")
    
    tab1, tab2, tab3 = st.tabs(["평가 결과", "평가 실행", "설정"])
    
    # 탭 1: 평가 결과 표시
    with tab1:
        st.header("평가 결과")
        
        # 평가 결과 파일 목록 가져오기
        result_files = [f for f in os.listdir('.') if f.startswith('evaluation_results_') and f.endswith('.csv')]
        
        if result_files:
            # 파일명에서 모델 이름 추출
            model_names = [f.replace('evaluation_results_', '').replace('.csv', '') for f in result_files]
            
            # 결과 파일 선택
            if 'current_model' in st.session_state:
                default_index = model_names.index(st.session_state['current_model']) if st.session_state['current_model'] in model_names else 0
            else:
                default_index = 0
                
            selected_model = st.selectbox(
                "모델 선택",
                model_names,
                index=default_index
            )
            
            selected_file = f"evaluation_results_{selected_model}.csv"
            
            if os.path.exists(selected_file):
                df = pd.read_csv(selected_file)
                st.success(f"'{selected_model}' 모델의 평가 결과를 표시합니다.")
            else:
                # 기본 결과 파일 사용
                if os.path.exists("evaluation_results.csv"):
                    df = pd.read_csv("evaluation_results.csv")
                    st.info("가장 최근 실행한 평가 결과를 표시합니다.")
                else:
                    st.warning("평가 결과 파일이 없습니다. '평가 실행' 탭에서 평가를 실행해 주세요.")
                    df = None
        else:
            # 기본 결과 파일만 있는 경우
            if os.path.exists("evaluation_results.csv"):
                df = pd.read_csv("evaluation_results.csv")
                
                # 모델 정보가 있는지 확인
                if 'model' in df.columns:
                    model_name = df['model'].iloc[0]
                    st.info(f"'{model_name}' 모델의 평가 결과를 표시합니다.")
                else:
                    st.info("평가 결과를 표시합니다.")
            else:
                st.warning("평가 결과 파일이 없습니다. '평가 실행' 탭에서 평가를 실행해 주세요.")
                df = None
        
        results_available = df is not None
        
        if results_available:
            # 결과 개요
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("평균 메트릭 점수")
                
                # 숫자 데이터만 선택
                numeric_df = df.select_dtypes(include=[np.number])
                metrics_cols = ['faithfulness', 'context_recall', 'context_precision', 'answer_relevancy']
                
                # 메트릭 컬럼이 있는지 확인
                available_metrics = [col for col in metrics_cols if col in numeric_df.columns]
                
                if available_metrics:
                    metrics_df = numeric_df[available_metrics]
                    
                    # 메트릭 이름 한글-영문 변환 딕셔너리 정의 (재사용을 위해 상단에 배치)
                    metric_names = {
                        'faithfulness': '충실도(faithfulness)',
                        'context_recall': '컨텍스트 재현율(context_recall)',
                        'context_precision': '컨텍스트 정밀도(context_precision)',
                        'answer_relevancy': '응답 관련성(answer_relevancy)'
                    }
                    
                    avg_scores = metrics_df.mean().reset_index()
                    avg_scores.columns = ['메트릭', '평균 점수']
                    
                    # 메트릭 이름 변환 (영어 -> 한글+영어)
                    avg_scores['원본 메트릭'] = avg_scores['메트릭']  # 원본 값 저장
                    avg_scores['메트릭'] = avg_scores['메트릭'].map(lambda x: metric_names.get(x, x))
                    
                    # 바 차트
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(avg_scores['메트릭'], avg_scores['평균 점수'])
                    
                    # 바 위에 값 표시
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    plt.title("평균 메트릭 점수")
                    plt.xlabel("메트릭")
                    plt.ylabel("점수")
                    plt.ylim(0, 1.0)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # 평균 점수 테이블
                    st.table(avg_scores.set_index('메트릭').drop(columns=['원본 메트릭']).T)
                else:
                    st.warning("평가 결과에 메트릭 데이터가 포함되어 있지 않습니다.")
            
            with col2:
                st.subheader("전체 결과")
                
                # 메트릭 컬럼이 있는지 확인
                if available_metrics:
                    # 히트맵 생성
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 히트맵 데이터 준비
                    heatmap_data = metrics_df.values
                    
                    # 질문 라벨 준비 (짧게 표시)
                    if 'user_input' in df.columns:
                        questions = df['user_input'].tolist()
                    else:
                        questions = [f"질문 {i+1}" for i in range(len(df))]
                        
                    short_questions = [q[:15] + '...' if len(q) > 15 else q for q in questions]
                    
                    im = ax.imshow(heatmap_data, cmap='Blues', vmin=0, vmax=1)
                    
                    # y 축에 질문 표시
                    ax.set_yticks(np.arange(len(questions)))
                    ax.set_yticklabels(short_questions)
                    
                    # 메트릭 이름 한글-영문 변환 
                    metric_names = {
                        'faithfulness': '충실도(faithfulness)',
                        'context_recall': '컨텍스트 재현율(context_recall)',
                        'context_precision': '컨텍스트 정밀도(context_precision)',
                        'answer_relevancy': '응답 관련성(answer_relevancy)'
                    }
                    
                    # x 축에 메트릭 표시 (한글-영문 병기)
                    ax.set_xticks(np.arange(len(available_metrics)))
                    ax.set_xticklabels([metric_names.get(m, m) for m in available_metrics], rotation=45)
                    
                    # 색상 바
                    cbar = ax.figure.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel("점수", rotation=-90, va="bottom")
                    
                    # 각 셀에 값 표시
                    for i in range(len(questions)):
                        for j in range(len(available_metrics)):
                            text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                                          ha="center", va="center", color="white" if heatmap_data[i, j] > 0.5 else "black")
                    
                    plt.title("질문별 메트릭 점수")
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                else:
                    st.warning("평가 결과에 메트릭 데이터가 포함되어 있지 않습니다.")
            
            # 상세 결과
            st.subheader("상세 평가 결과")
            
            # 필요한 열이 있는지 확인
            display_cols = ['user_input', 'response', 'reference', 'faithfulness', 
                           'context_recall', 'context_precision', 'answer_relevancy']
            
            # 존재하는 열만 선택
            available_cols = [col for col in display_cols if col in df.columns]
            
            if available_cols:
                display_df = df[available_cols]
                
                # 열 이름 매핑 (같은 딕셔너리 재사용)
                col_mapping = {
                    'user_input': '질문',
                    'response': 'AI 응답',
                    'reference': '참조 정답',
                    'faithfulness': '충실도(faithfulness)',
                    'context_recall': '컨텍스트 재현율(context_recall)',
                    'context_precision': '컨텍스트 정밀도(context_precision)',
                    'answer_relevancy': '응답 관련성(answer_relevancy)'
                }
                
                # 존재하는 열에 대해서만 이름 변경
                rename_mapping = {col: col_mapping.get(col, col) for col in available_cols}
                display_df = display_df.rename(columns=rename_mapping)
                
                # 결과를 확인하기 위한 디버깅 정보
                st.info(f"표시할 열: {list(display_df.columns)}")
                
                # 데이터프레임 표시
                st.dataframe(display_df, use_container_width=True)

                # 질문별 상세 결과 표시를 위한 추가 섹션
                st.subheader("질문별 상세 결과")
                for i in range(len(display_df)):
                    with st.expander(f"질문 {i+1}: {display_df.iloc[i]['질문'][:50]}...", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**질문:**")
                            st.text(display_df.iloc[i]['질문'])
                            
                            st.markdown("**AI 응답:**")
                            if 'AI 응답' in display_df.columns:
                                st.text(display_df.iloc[i]['AI 응답'])
                            else:
                                st.warning("AI 응답 데이터가 없습니다.")
                        
                        with col2:
                            st.markdown("**참조 정답:**")
                            if '참조 정답' in display_df.columns:
                                st.text(display_df.iloc[i]['참조 정답'])
                            else:
                                st.warning("참조 정답 데이터가 없습니다.")
                            
                            # 메트릭 점수 표시
                            st.markdown("**평가 점수:**")
                            metrics = ['충실도(faithfulness)', '컨텍스트 재현율(context_recall)', 
                                       '컨텍스트 정밀도(context_precision)', '응답 관련성(answer_relevancy)']
                            
                            # 존재하는 메트릭만 표시
                            available_metrics = [m for m in metrics if m in display_df.columns]
                            
                            if available_metrics:
                                # 메트릭을 2개씩 묶어서 표시하기 위한 레이아웃
                                metric_rows = [available_metrics[i:i+2] for i in range(0, len(available_metrics), 2)]
                                
                                for metric_row in metric_rows:
                                    metric_cols = st.columns(len(metric_row))
                                    for j, metric in enumerate(metric_row):
                                        # 메트릭 이름에서 영문 부분 제거하고 표시
                                        metric_name = metric.split('(')[0]
                                        value = float(display_df.iloc[i][metric])
                                        # 색상 설정 (0.7 이상이면 초록색, 0.4 미만이면 빨간색, 나머지는 주황색)
                                        if value >= 0.7:
                                            delta_color = "normal"  # 초록색
                                        elif value < 0.4:
                                            delta_color = "inverse"  # 빨간색
                                        else:
                                            delta_color = "off"  # 주황색
                                            
                                        # 점수 범위 표시 및 값 표시
                                        metric_cols[j].metric(
                                            label=metric_name,
                                            value=f"{value:.2f}",
                                            delta=f"범위: 0-1",
                                            delta_color=delta_color
                                        )
                            else:
                                st.warning("평가 점수 데이터가 없습니다.")
                            
                            # 종합 점수 표시 (모든 메트릭의 평균)
                            if available_metrics:
                                st.markdown("---")
                                avg_score = sum([float(display_df.iloc[i][m]) for m in available_metrics]) / len(available_metrics)
                                st.metric(
                                    label="종합 점수",
                                    value=f"{avg_score:.2f}",
                                    delta=f"메트릭 {len(available_metrics)}개의 평균",
                                    delta_color="normal" if avg_score >= 0.7 else "off"
                                )
    # 탭 2: 평가 실행
    with tab2:
        st.header("RAGAS 평가 실행")
        
        # AWS Bedrock 모델 설정
        with st.expander("AWS Bedrock 모델 설정", expanded=not results_available):
            aws_enabled = st.checkbox("AWS Bedrock 사용", value=True)
            
            if aws_enabled:
                bedrock_model_id = st.selectbox("Bedrock LLM 모델", 
                                              ["anthropic.claude-3-haiku-20240307-v1:0",
                                               "anthropic.claude-3-sonnet-20240229-v1:0",
                                               "meta.llama3-8b-instruct-v1:0"],
                                              index=0)
                
                embedding_model_id = st.selectbox("임베딩 모델", 
                                                ["amazon.titan-embed-text-v1"],
                                                index=0)
                
                temperature = 0.1
            else:
                st.info("AWS Bedrock을 사용하지 않을 경우, 샘플 데이터에 대한 결과만 볼 수 있습니다.")
        
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
                "contexts": [
                    [
                        "대한민국의 수도는 서울특별시이다.",
                        "서울은 대한민국의 정치, 경제, 사회, 문화의 중심지이다.",
                    ],
                    [
                        "인공지능은 다양한 기준에 따라 여러 가지로 분류될 수 있다.",
                        "일반적으로는 강인공지능과 약인공지능으로 나눌 수 있다.",
                    ],
                    [
                        "한국의 경제 성장률은 한국은행 경제통계시스템에서 확인할 수 있다.",
                        "경제 성장률은 국가 경제의 건전성을 나타내는 중요한 지표 중 하나이다.",
                    ],
                    [
                        "봉준호 감독은 한국을 대표하는 영화감독 중 한 명이다.",
                        "영화 '기생충'은 2019년 칸 영화제에서 황금종려상을 수상했다.",
                    ],
                    [
                        "미국 주식 시장은 세계 경제에 큰 영향을 미친다.",
                        "2023년 미국 주식 시장은 금리 인상, 인플레이션 등의 변수에 영향을 받았다.",
                    ],
                ],
            }
            
            st.subheader("테스트 데이터 입력")
            st.info("기본값이 제공된 샘플 데이터를 수정하거나 그대로 사용할 수 있습니다.")
            
            # 데이터 초기화 및 불러오기 버튼
            button_container = st.container()
            with button_container:
                col1, col2 = st.columns([1, 1], gap="small")
                with col1:
                    if st.button("전체 데이터 지우기", type="secondary", use_container_width=True):
                        # 현재 항목 수 저장
                        current_n_items = st.session_state.get('n_items', 5)
                        # session_state 완전 초기화
                        st.session_state.clear()
                        # 필요한 상태만 다시 설정
                        st.session_state['n_items'] = current_n_items
                        st.session_state['clear_data'] = True
                        # 모든 입력 필드를 빈 값으로 초기화
                        for i in range(10):  # 최대 가능한 항목 수
                            st.session_state[f"q_{i}"] = ""
                            st.session_state[f"r_{i}"] = ""
                            st.session_state[f"ref_{i}"] = ""
                            st.session_state[f"ctx_{i}"] = ""
                        st.rerun()
                with col2:
                    if st.button("기본 테스트 데이터 불러오기", type="secondary", use_container_width=True):
                        # 현재 항목 수 저장
                        current_n_items = st.session_state.get('n_items', 5)
                        # session_state 완전 초기화
                        st.session_state.clear()
                        # 필요한 상태만 다시 설정
                        st.session_state['n_items'] = current_n_items
                        st.session_state['clear_data'] = False
                        st.rerun()
            
            # 여러 항목 입력을 위한 컨테이너
            if 'n_items' not in st.session_state:
                st.session_state['n_items'] = 5
            
            n_items = st.number_input("테스트 항목 수", min_value=1, max_value=10, value=st.session_state['n_items'])
            if n_items != st.session_state['n_items']:
                st.session_state['n_items'] = n_items
                st.rerun()
            
            test_data = {
                "question": [],
                "response": [],
                "reference": [],
                "contexts": []
            }
            
            for i in range(n_items):
                st.markdown(f"### 테스트 항목 {i+1}")
                col1, col2 = st.columns(2)
                
                # 각 필드의 키 생성
                q_key = f"q_{i}"
                r_key = f"r_{i}"
                ref_key = f"ref_{i}"
                ctx_key = f"ctx_{i}"
                
                if 'clear_data' in st.session_state and st.session_state['clear_data']:
                    # 전체 데이터 지우기가 활성화된 경우
                    st.session_state[q_key] = ""
                    st.session_state[r_key] = ""
                    st.session_state[ref_key] = ""
                    st.session_state[ctx_key] = ""
                    default_q = ""
                    default_r = ""
                    default_ref = ""
                    default_ctx = ""
                else:
                    # 기본 데이터 사용
                    default_q = default_data["question"][i] if i < len(default_data["question"]) else ""
                    default_r = default_data["response"][i] if i < len(default_data["response"]) else ""
                    default_ref = default_data["reference"][i] if i < len(default_data["reference"]) else ""
                    default_ctx = "\n".join(default_data["contexts"][i]) if i < len(default_data["contexts"]) else ""
                
                with col1:
                    q = st.text_area(f"질문 {i+1}", value=default_q, key=q_key)
                    r = st.text_area(f"응답 {i+1}", value=default_r, key=r_key)
                
                with col2:
                    ref = st.text_area(f"참조 정답 {i+1}", value=default_ref, key=ref_key)
                    ctx = st.text_area(f"컨텍스트 {i+1} (줄바꿈으로 구분)", value=default_ctx, key=ctx_key)
                
                if q and r and ref and ctx:
                    test_data["question"].append(q)
                    test_data["response"].append(r)
                    test_data["reference"].append(ref)
                    test_data["contexts"].append(ctx.split('\n'))
            
            if len(test_data["question"]) > 0:
                st.success(f"{len(test_data['question'])}개 항목이 입력되었습니다.")
            else:
                st.warning("유효한 테스트 항목이 없습니다. 모든 필드를 입력해주세요.")
                
        # 평가 실행 버튼
        if st.button("평가 실행", type="primary"):
            with st.spinner("평가를 실행 중입니다..."):
                try:
                    # 샘플 데이터 사용 또는 실제 AWS Bedrock 사용하여 평가
                    if aws_enabled:
                        # AWS 리전 설정 (Cloud9에서는 자격 증명이 이미 설정되어 있음)
                        aws_region = "us-east-1"
                        
                        # AWS Bedrock 클라이언트 설정
                        bedrock_client = boto3.client("bedrock-runtime", region_name=aws_region)
                        
                        # Bedrock 모델 설정
                        bedrock_model = ChatBedrock(
                            client=bedrock_client,
                            model_id=bedrock_model_id,
                            model_kwargs={"temperature": temperature},
                        )
                        
                        # 모델 이름을 파일명에 포함
                        if "claude" in bedrock_model_id.lower():
                            # Claude 모델의 경우 더 구체적인 식별자 사용
                            # 예: anthropic.claude-3-haiku-20240307-v1:0 -> claude-3-haiku-20240307
                            model_parts = bedrock_model_id.split(".")
                            if len(model_parts) > 1:
                                model_id_parts = model_parts[1].split("-")
                                if len(model_id_parts) >= 3:
                                    # claude-3-haiku 또는 claude-3-sonnet 부분 추출
                                    model_short_name = f"{model_id_parts[0]}-{model_id_parts[1]}-{model_id_parts[2]}"
                                    # 날짜 부분도 추가
                                    if len(model_id_parts) > 3:
                                        model_short_name += f"-{model_id_parts[3]}"
                                else:
                                    model_short_name = model_parts[1].split(":")[0]
                            else:
                                model_short_name = "claude"
                        elif "llama" in bedrock_model_id.lower():
                            # Llama 모델의 경우
                            model_parts = bedrock_model_id.split(".")
                            if len(model_parts) > 1:
                                model_id_parts = model_parts[1].split("-")
                                if len(model_id_parts) >= 2:
                                    # llama3-8b 등의 형태로 추출
                                    model_short_name = f"{model_id_parts[0]}-{model_id_parts[1]}"
                                else:
                                    model_short_name = model_parts[1].split(":")[0]
                            else:
                                model_short_name = "llama"
                        else:
                            # 기타 모델의 경우 기존 방식 사용
                            model_short_name = bedrock_model_id.split('.')[-1].split('-')[0]
                        
                        results_filename = f"evaluation_results_{model_short_name}.csv"
                        
                        # Bedrock 임베딩 설정
                        bedrock_embeddings = BedrockEmbeddings(
                            client=bedrock_client, model_id=embedding_model_id
                        )
                        
                        # 데이터 준비
                        if len(test_data["question"]) > 0:
                            eval_dataset = Dataset.from_dict(test_data)
                        else:
                            # 테스트 데이터가 없는 경우 기본 샘플 데이터 사용
                            st.warning("유효한 테스트 데이터가 없어 기본 샘플 데이터를 사용합니다.")
                            eval_dataset = Dataset.from_dict(default_data)
                        
                        # 모델 정보를 세션 상태에 저장
                        st.session_state['current_model'] = model_short_name
                        st.session_state['current_results_file'] = results_filename
                        
                        # 평가 실행
                        result = run_ragas_evaluation(
                            eval_dataset,
                            bedrock_model=bedrock_model,
                            bedrock_embeddings=bedrock_embeddings,
                        )
                        
                        # 결과를 DataFrame으로 변환
                        df = result.to_pandas()
                        
                        # 모델 정보 추가
                        df['model'] = model_short_name
                        
                        # 결과 저장 (모델별 파일명)
                        df.to_csv(results_filename)
                        
                        # 기본 결과 파일도 업데이트 (최신 실행 결과를 보여줌)
                        df.to_csv("evaluation_results.csv")
                        
                        # 결과 시각화
                        st.success(f"평가 완료! '{model_short_name}' 모델 평가 결과가 저장되었습니다. '평가 결과' 탭에서 결과를 확인하세요.")
                        st.rerun()
                        
                    else:
                        # AWS를 사용하지 않는 경우 샘플 결과 생성
                        st.info("AWS 자격 증명이 제공되지 않아 샘플 결과를 생성합니다.")
                        
                        # 샘플 결과 데이터 생성
                        sample_data = {
                            "Unnamed: 0": [0, 1, 2, 3, 4],
                            "user_input": [
                                "한국의 수도는 어디인가요?",
                                "인공지능의 종류는 무엇이 있나요?",
                                "최근 5년간 한국 경제 성장률은 어떻게 변했나요?",
                                "영화 '기생충'의 감독은 누구인가요?",
                                "2023년 미국 주식 시장은 어땠나요?",
                            ],
                            "retrieved_contexts": [
                                str(["대한민국의 수도는 서울특별시이다.", "서울은 대한민국의 정치, 경제, 사회, 문화의 중심지이다."]),
                                str(["인공지능은 다양한 기준에 따라 여러 가지로 분류될 수 있다.", "일반적으로는 강인공지능과 약인공지능으로 나눌 수 있다."]),
                                str(["한국의 경제 성장률은 한국은행 경제통계시스템에서 확인할 수 있다.", "경제 성장률은 국가 경제의 건전성을 나타내는 중요한 지표 중 하나이다."]),
                                str(["봉준호 감독은 한국을 대표하는 영화감독 중 한 명이다.", "영화 '기생충'은 2019년 칸 영화제에서 황금종려상을 수상했다."]),
                                str(["미국 주식 시장은 세계 경제에 큰 영향을 미친다.", "2023년 미국 주식 시장은 금리 인상, 인플레이션 등의 변수에 영향을 받았다."]),
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
                            "faithfulness": [0.6, 0.7, 0.5, 0.65, 0.55],
                            "context_recall": [0.8, 0.75, 0.85, 0.7, 0.9],
                            "context_precision": [0.95, 0.85, 0.9, 0.8, 0.95],
                            "answer_relevancy": [0.8, 0.7, 0.85, 0.75, 0.7],
                            "model": ["sample-model"] * 5
                        }
                        
                        df = pd.DataFrame(sample_data)
                        sample_filename = "evaluation_results_sample.csv"
                        df.to_csv(sample_filename)
                        df.to_csv("evaluation_results.csv")
                        
                        st.session_state['current_model'] = "sample"
                        st.session_state['current_results_file'] = sample_filename
                        
                        st.success("샘플 평가 결과가 생성되었습니다. '평가 결과' 탭에서 결과를 확인하세요.")
                        st.rerun()
                except Exception as e:
                    st.error(f"평가 중 오류가 발생했습니다: {str(e)}")
    
    # 탭 3: 설정
    with tab3:
        st.header("설정")
        
        # 메트릭 설명
        with st.expander("RAGAS 메트릭 설명", expanded=True):
            st.markdown("""
            ### RAGAS 메트릭 설명
            
            * **충실도(Faithfulness)**: 모델의 응답이 제공된 컨텍스트에 얼마나 충실한지 평가합니다. 
              높은 점수는 응답이 컨텍스트에 있는 정보만을 사용하고 추가 정보를 생성하지 않았음을 의미합니다.
            
            * **컨텍스트 재현율(Context Recall)**: 참조 답변에 필요한 정보가 검색된 컨텍스트에 얼마나 포함되어 있는지 측정합니다. 
              높은 점수는 필요한 정보가 대부분 검색되었음을 의미합니다.
            
            * **컨텍스트 정밀도(Context Precision)**: 검색된 컨텍스트가 질문에 얼마나 관련이 있는지 측정합니다. 
              높은 점수는 검색된 컨텍스트가 질문과 밀접하게 관련되어 있음을 의미합니다.
            
            * **응답 관련성(Answer Relevancy)**: 응답이 주어진 질문에 얼마나 관련이 있는지 측정합니다. 
              높은 점수는 응답이 질문에 직접적으로 답하고 있음을 의미합니다.
            """)
        
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
        st.info("""
        **RAGAS 평가 대시보드**
        
        이 애플리케이션은 RAGAS(Retrieval Augmented Generation Assessment Suite)를 사용하여 
        질의응답 시스템의 성능을 평가하고 시각화합니다.
        
        GitHub: [RAGAS](https://github.com/explodinggradients/ragas)
        """)

if __name__ == "__main__":
    setup_korean_font()
    main()
