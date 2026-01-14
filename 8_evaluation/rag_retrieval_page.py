"""
RAG Retrieval Evaluator - Streamlit Application
다중 벡터 데이터베이스의 검색 품질을 실시간으로 비교 평가
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ==================== 초기 설정 ====================
st.set_page_config(
    page_title="RAG Retrieval Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 모듈 Import
try:
    import rag_retrieval_evaluator
    import importlib
    importlib.reload(rag_retrieval_evaluator)
    
    from rag_retrieval_evaluator import (
        ChromaDBRetriever, PostgreSQLRetriever, AWSKnowledgeBaseRetriever,
        RetrievalEvaluator, TEST_DATASET
    )
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
    import boto3
    from langchain_aws import ChatBedrock
except ImportError as e:
    st.error(f"모듈 import 실패: {e}")
    st.stop()

# ==================== 스타일 ====================
st.markdown("""
<style>
.retriever-card {
    background-color: #F0F2F6;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #DDE1E6;
    transition: all 0.3s ease;
}
[data-theme="dark"] .retriever-card {
    background-color: #1E2129;
    border: 1px solid #30363D;
}
.retriever-card:hover {
    border-color: #58A6FF;
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.15);
}
.result-content {
    font-size: 1rem;
    line-height: 1.6;
    color: #1F2328;
    background: #FFFFFF;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
    border-left: 4px solid #0068C9;
}
[data-theme="dark"] .result-content {
    color: #C9D1D9;
    background: #0D1117;
    border-left-color: #58A6FF;
}
.rank-badge {
    background: #238636;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: bold;
}
.system-header {
    background: #1E2129;
    padding: 10px;
    border-radius: 8px;
    border-bottom: 2px solid #58A6FF;
    margin-bottom: 20px;
}
.system-title {
    color: #58A6FF;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==================== 초기화 함수 ====================
@st.cache_resource
def initialize_retrievers():
    """사용 가능한 모든 Retriever 초기화"""
    available = {}
    
    # ChromaDB
    pdf_path = os.getenv("CHROMA_PDF_PATH", os.path.abspath(os.path.join(current_dir, "../5_RAG/data/univ-data.pdf")))
    db_path = os.getenv("CHROMA_DB_PATH", os.path.abspath(os.path.join(current_dir, "../5_RAG/vector_db")))
    
    if os.path.exists(pdf_path):
        try:
            available["ChromaDB"] = ChromaDBRetriever(pdf_path=pdf_path, vector_db_path=db_path)
        except:
            pass

    # PostgreSQL
    if os.getenv("DB_HOST"):
        try:
            available["PostgreSQL"] = PostgreSQLRetriever()
        except:
            pass

    # AWS Knowledge Base
    kb_ids = [kb.strip() for kb in os.getenv("AWS_KB_IDS", "").split(",") if kb.strip()]
    if kb_ids:
        try:
            available["KnowledgeBase"] = AWSKnowledgeBaseRetriever(knowledge_base_ids=kb_ids)
        except:
            pass
        
    return available

@st.cache_resource
def initialize_llm():
    """답변 생성용 LLM 초기화"""
    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        return ChatBedrock(
            client=client,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={"temperature": 0}
        )
    except:
        return None

def generate_answer(llm, question: str, contexts: list) -> str:
    """검색된 컨텍스트로 답변 생성"""
    if not contexts:
        return "관련 정보를 찾을 수 없습니다."
    
    prompt = f"""참고 자료:
{chr(10).join(contexts)}

질문: {question}

위 자료의 정보만을 사용하여 질문에 직접 답변하세요.

규칙:
1. "자료에 따르면", "컨텍스트에서", "위 내용을 보면" 등의 메타 표현 금지
2. "~입니다", "~합니다" 등 단정적 어조 사용
3. 핵심 정보만 간결하게 전달
4. 질문에서 묻지 않은 추가 설명 금지

답변:"""

    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"답변 생성 오류: {str(e)}"

# ==================== 전역 변수 ====================
all_retrievers = initialize_retrievers()
llm = initialize_llm()

METRICS = {
    "context_precision": {"name": "Context Precision (문맥 정밀도)", "instance": ContextPrecision(), "default": True},
    "context_recall": {"name": "Context Recall (문맥 재현율)", "instance": ContextRecall(), "default": True},
    "faithfulness": {"name": "Faithfulness (충실도)", "instance": Faithfulness(), "default": False},
    "answer_relevancy": {"name": "Answer Relevancy (답변 관련성)", "instance": AnswerRelevancy(), "default": False}
}

# ==================== 사이드바 ====================
st.sidebar.title("🚀 RAG Controller")

st.sidebar.markdown("### 🔍 Retriever 선택")
selected_retrievers = [name for name in all_retrievers if st.sidebar.checkbox(name, value=True)]

st.sidebar.markdown("### 📊 평가 메트릭 선택")
selected_metrics = [info["instance"] for key, info in METRICS.items() if st.sidebar.checkbox(info["name"], value=info["default"])]

if not selected_metrics:
    st.sidebar.warning("⚠️ 최소 하나의 메트릭을 선택해주세요.")

# ==================== 메인 UI ====================
st.title("🔍 Multi-Vector RAG Evaluator")
st.markdown("다양한 데이터베이스 백엔드의 검색 품질을 실시간으로 비교합니다.")

with st.expander("📅 평가 데이터셋 정보"):
    st.write(f"현재 로드된 질문 수: **{len(TEST_DATASET['questions'])}**개")
    st.markdown("### 질문 리스트")
    for i, q in enumerate(TEST_DATASET['questions'], 1):
        st.text(f"{i}. {q}")

# 유효성 검사
if not selected_retrievers:
    st.warning("사이드바에서 최소 하나의 Retriever를 선택해주세요.")
    st.stop()

if not selected_metrics:
    st.warning("사이드바에서 최소 하나의 평가 메트릭을 선택해주세요.")
    st.stop()

# 세션 스테이트
if 'mode' not in st.session_state:
    st.session_state.mode = 'single'

# 모드 선택
st.markdown("---")
col1, col2 = st.columns(2)
if col1.button("🧪 단일 테스트", type="primary", use_container_width=True):
    st.session_state.mode = 'single'
if col2.button("📈 배치 평가", use_container_width=True):
    st.session_state.mode = 'batch'

# ==================== 단일 테스트 함수 ====================
def display_single_results(query, k, retriever_names):
    """검색 결과 표시 (병렬 처리)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    st.markdown("### 🎯 검색 결과 비교")
    
    # 병렬로 검색 실행
    def retrieve_with_timing(name):
        start = time.time()
        try:
            results = all_retrievers[name].retrieve(query, k=k)
            duration = time.time() - start
            return name, results, duration, None
        except Exception as e:
            duration = time.time() - start
            return name, None, duration, str(e)
    
    # 병렬 실행
    search_results = {}
    with ThreadPoolExecutor(max_workers=len(retriever_names)) as executor:
        futures = {executor.submit(retrieve_with_timing, name): name for name in retriever_names}
        
        for future in as_completed(futures):
            name, results, duration, error = future.result()
            search_results[name] = {
                'results': results,
                'duration': duration,
                'error': error
            }
    
    # 결과 표시 (원래 순서대로)
    cols = st.columns(len(retriever_names))
    
    for idx, name in enumerate(retriever_names):
        with cols[idx]:
            st.markdown(f"<div class='system-header'><div class='system-title'>{name}</div></div>", unsafe_allow_html=True)
            
            data = search_results[name]
            st.caption(f"⏱️ DB 조회: {data['duration']:.3f}초")
            
            if data['error']:
                st.error(f"오류: {data['error']}")
            elif not data['results']:
                st.info("컨텍스트를 찾지 못했습니다.")
            else:
                results = data['results']
                
                # AI 답변 생성 (시간 측정 제외)
                if llm:
                    with st.spinner("답변 생성 중..."):
                        answer = generate_answer(llm, query, results)
                        st.markdown("**🤖 AI 생성 답변**")
                        st.info(answer)
                    st.markdown("---")
                
                for rank, result in enumerate(results, 1):
                    st.markdown(f"""
                    <div class='retriever-card'>
                        <span class='rank-badge'>순위 {rank}</span>
                        <div class='result-content'>{result}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ==================== 배치 평가 함수 ====================
def run_batch_evaluation(retrievers, metrics, k_value):
    """배치 평가 실행"""
    st.markdown("---")
    st.subheader("📈 성능 분석")

    retriever_list = [all_retrievers[name] for name in retrievers]
    evaluator = RetrievalEvaluator(
        retrievers=retriever_list,
        questions=TEST_DATASET["questions"],
        gold_contexts=TEST_DATASET["gold_contexts"]
    )

    completed = st.container()
    
    with st.status("평가 중...", expanded=True) as status:
        # 1단계: 문서 검색
        st.write("📋 **1단계: 문서 검색**")
        all_results = {}
        for name in retrievers:
            st.write(f"   → {name} 검색 중...")
            results = [all_retrievers[name].retrieve(q, k=k_value) for q in TEST_DATASET["questions"]]
            all_results[all_retrievers[name].get_system_name()] = results
        
        with completed:
            st.success("✅ **1단계 완료**: 문서 검색 완료")
        
        # 1.5단계: 답변 생성
        has_gen = any(isinstance(m, (Faithfulness, AnswerRelevancy)) for m in metrics)
        all_answers = None
        
        if has_gen:
            if llm is None:
                st.warning("⚠️ LLM 초기화 실패. 검색 메트릭만 평가합니다.")
                metrics = [m for m in metrics if not isinstance(m, (Faithfulness, AnswerRelevancy))]
            else:
                st.write("🤖 **1.5단계: 답변 생성**")
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                total = len(retrievers) * len(TEST_DATASET["questions"])
                current = 0
                
                all_answers = {}
                for system_name, contexts_list in all_results.items():
                    system_answers = []
                    for q, ctx in zip(TEST_DATASET["questions"], contexts_list):
                        system_answers.append(generate_answer(llm, q, ctx))
                        current += 1
                        progress = current / total
                        progress_bar.progress(progress)
                        progress_text.text(f"답변 생성 중... {current}/{total} ({progress*100:.1f}%)")
                    all_answers[system_name] = system_answers
                
                progress_bar.empty()
                progress_text.empty()
                
                with completed:
                    st.success("✅ **1.5단계 완료**: 답변 생성 완료")
        
        # 2단계: 메트릭 평가
        st.write("📊 **2단계: 메트릭 평가**")
        eval_progress = st.progress(0)
        eval_text = st.empty()
        
        all_dfs = {}
        total_sys = len(all_results)
        
        for idx, (system_name, contexts) in enumerate(all_results.items()):
            eval_text.text(f"평가 중... {system_name} ({idx+1}/{total_sys})")
            system_answers = all_answers.get(system_name) if all_answers else None
            df = evaluator.evaluate_system(system_name, contexts, system_answers)
            
            if df is not None:
                all_dfs[system_name] = df
            
            eval_progress.progress((idx + 1) / total_sys)
        
        eval_progress.empty()
        eval_text.empty()
        
        with completed:
            st.success("✅ **2단계 완료**: 메트릭 평가 완료")
        
        comparison_df = evaluator.create_comparison_report(all_dfs)
        status.update(label="✅ 평가 완료!", state="complete", expanded=False)
    
    # 시각화
    display_visualizations(comparison_df, all_dfs)
    display_detailed_logs(all_dfs)
    
    st.balloons()

def display_visualizations(comparison_df, all_dfs):
    """시각화 표시"""
    if comparison_df is None or comparison_df.empty:
        st.error("⚠️ 비교 결과를 생성할 수 없습니다.")
        return
    
    if not all(col in comparison_df.columns for col in ['Metric', 'Mean', 'System']):
        st.warning("⚠️ 비교 리포트 형식이 예상과 다릅니다.")
        st.dataframe(comparison_df)
        return
    
    st.markdown("---")
    st.markdown("## 📊 시각화")
    
    # 전체 통계 요약
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("평가 시스템 수", len(all_dfs))
    with col2:
        st.metric("평가 메트릭 수", len(comparison_df['Metric'].unique()))
    with col3:
        avg_score = comparison_df['Mean'].mean()
        st.metric("전체 평균 점수", f"{avg_score:.3f}")
    with col4:
        best_system = comparison_df.groupby('System')['Mean'].mean().idxmax()
        st.metric("최고 성능", best_system)
    
    st.markdown("---")
    
    # 메트릭별 성능 비교
    st.markdown("### 📊 메트릭별 성능 비교")
    fig_bar = px.bar(
        comparison_df, x="Metric", y="Mean", color="System",
        barmode="group", title="데이터베이스별 메트릭 성능 비교",
        template="plotly_dark", height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # 성능 히트맵
    st.markdown("### 📊 성능 히트맵")
    pivot = comparison_df.pivot(index="System", columns="Metric", values="Mean")
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale='Blues',
        text=pivot.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 12},
        colorbar=dict(title="Score")
    ))
    
    fig_heat.update_layout(xaxis_title="Metric", yaxis_title="System", height=300)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    
    # 시스템별 종합 점수
    st.markdown("### 🏆 시스템별 종합 점수")
    system_avg = comparison_df.groupby('System')['Mean'].mean().sort_values(ascending=False)
    
    cols = st.columns(len(system_avg))
    for idx, ((system, score), col) in enumerate(zip(system_avg.items(), cols), 1):
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "📊"
        with col:
            st.metric(f"{medal} {idx}위", system, f"{score:.3f}")

def display_detailed_logs(all_dfs):
    """상세 평가 로그 표시"""
    if not all_dfs:
        return
    
    st.markdown("---")
    st.markdown("### 📄 상세 평가 로그")
    
    tabs = st.tabs(list(all_dfs.keys()))
    
    for idx, system_name in enumerate(all_dfs.keys()):
        with tabs[idx]:
            df = all_dfs[system_name]
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # 시스템별 요약
            st.markdown(f"#### {system_name} 평가 요약")
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                st.markdown("**평균 ± 표준편차**")
                for col in numeric_cols:
                    avg, std = df[col].mean(), df[col].std()
                    st.write(f"• **{col}**: {avg:.3f} (±{std:.3f})")
            
            with col_sum2:
                st.markdown("**최소 ~ 최대**")
                for col in numeric_cols:
                    min_val, max_val = df[col].min(), df[col].max()
                    st.write(f"• **{col}**: {min_val:.3f} ~ {max_val:.3f}")
            
            st.markdown("---")
            st.markdown("**전체 평가 데이터**")
            
            # 카드 형식으로 표시
            for row_idx in range(len(df)):
                display_evaluation_card(df.iloc[row_idx], row_idx, numeric_cols)

def display_evaluation_card(row, row_idx, numeric_cols):
    """개별 평가 카드 표시"""
    # 점수 계산
    scores = [row[col] for col in numeric_cols if pd.notna(row[col])]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # 점수에 따른 색상
    if avg_score >= 0.9:
        score_color = "🟢"
    elif avg_score >= 0.7:
        score_color = "🟡"
    else:
        score_color = "🔴"
    
    # 질문 미리보기
    question_preview = ""
    if 'user_input' in row:
        question_text = str(row['user_input'])
        question_preview = question_text[:80] + "..." if len(question_text) > 80 else question_text
    
    # Expander 제목
    expander_title = f"{score_color} 질문 {row_idx + 1}"
    if question_preview:
        expander_title += f" {question_preview}"
    expander_title += f" - 평균 점수: {avg_score:.3f}"
    
    with st.expander(expander_title):
        # 질문
        if 'user_input' in row:
            st.markdown("**📝 질문**")
            st.info(row['user_input'])
        
        # 답변
        if 'response' in row:
            st.markdown("**🤖 AI 답변**")
            st.success(row['response'])
        
        # 검색된 컨텍스트
        if 'retrieved_contexts' in row:
            st.markdown("**📚 검색된 컨텍스트**")
            contexts = row['retrieved_contexts']
            if isinstance(contexts, list):
                for ctx_idx, ctx in enumerate(contexts, 1):
                    st.markdown(f"**{ctx_idx}.** {ctx}")
            else:
                st.markdown(contexts)
        
        # 정답 참조
        if 'reference' in row:
            st.markdown("**✅ 정답 참조**")
            st.warning(row['reference'])
        
        st.markdown("---")
        
        # 메트릭 점수
        st.markdown("**📊 메트릭 점수**")
        metric_cols = st.columns(len(numeric_cols))
        for col_idx, col in enumerate(numeric_cols):
            with metric_cols[col_idx]:
                score = row[col]
                if pd.notna(score):
                    st.metric(col, f"{score:.3f}")
                else:
                    st.metric(col, "N/A")

# ==================== 모드별 실행 ====================
if st.session_state.mode == 'single':
    st.markdown("---")
    st.subheader("🧪 단일 테스트")
    
    col_input, col_k = st.columns([4, 1])
    query = col_input.text_input("검색 쿼리:", placeholder="예: 조기졸업 요건이 뭐야?")
    k = col_k.number_input("Top K", min_value=1, max_value=10, value=3)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_clicked = st.button("🔍 검색 실행", type="primary", use_container_width=True)
    
    if search_clicked:
        if query:
            display_single_results(query, k, selected_retrievers)
        else:
            st.warning("검색 쿼리를 입력해주세요.")

else:  # batch mode
    st.markdown("---")
    st.subheader("📈 배치 평가")
    
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.info(f"""
        **평가 설정:**
        - Retriever: {', '.join(selected_retrievers)}
        - 질문 수: {len(TEST_DATASET['questions'])}개
        - 메트릭: {len(selected_metrics)}개
        """)
    
    with col_right:
        batch_k = st.number_input("Top K", min_value=1, max_value=10, value=3, key="batch_k")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        eval_clicked = st.button("🚀 평가 시작", type="primary", use_container_width=True)
    
    if eval_clicked:
        run_batch_evaluation(selected_retrievers, selected_metrics, batch_k)