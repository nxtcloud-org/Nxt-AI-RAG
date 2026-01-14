import os
import io
import time
import boto3
import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from kb_management import KBManager


kb_manager = KBManager(file_path='kbs.json')

st.set_page_config(
    page_title="문서 기반 질의응답 시스템 (실무형 KB 관리)",
    page_icon="📚",
    layout="wide"
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_session_history(session_id):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    return st.session_state.chat_history

@st.cache_resource
def init_bedrock():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    llm = ChatBedrock(
        client=bedrock_client,
        model="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "temperature": 0.1
        },
        streaming=True
    )
    conversation = RunnableWithMessageHistory(
        llm,
        get_session_history,
        max_history=3
    ).with_config(configurable={"session_id": "default"})
    return conversation, bedrock_client

@st.cache_resource
def init_s3():
    return boto3.client('s3', region_name="us-east-1")

def retrieve_from_kb(query, knowledge_base_ids, k=3):
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name="us-east-1")
    all_retrieval_results = []
    
    # 검색 요청 생성 함수 (SEMANTIC 고정)
    def get_retrieval_config():
        config = {
            'vectorSearchConfiguration': {
                'numberOfResults': k,
                'overrideSearchType': 'SEMANTIC'
            }
        }
        return config

    for kb_id in knowledge_base_ids:
        if not kb_id: continue
        
        retrieval_configuration = get_retrieval_config()
        
        try:
            actual_method = 'SEMANTIC'
            response = bedrock_agent_runtime.retrieve(
                knowledgeBaseId=kb_id,
                retrievalConfiguration=retrieval_configuration,
                retrievalQuery={'text': query}
            )
            results = response.get('retrievalResults', [])
            
            # 검색 메타데이터 주입
            for r in results:
                r['actual_search_method'] = actual_method
                
            all_retrieval_results.extend(results)
            
        except Exception as e:
            st.error(f"KB 검색 중 오류 ({kb_id}): {str(e)}")
                
    return all_retrieval_results

def start_ingestion(kb_id, ds_id):
    client = boto3.client('bedrock-agent', region_name="us-east-1")
    try:
        response = client.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
        return response['ingestionJob']['ingestionJobId']
    except Exception as e:
        st.error(f"데이터 동기화 시작 실패: {str(e)}")
        return None

def check_ingestion_status(kb_id, ds_id, job_id):
    client = boto3.client('bedrock-agent', region_name="us-east-1")
    try:
        response = client.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)
        return response['ingestionJob']['status']
    except Exception:
        return "ERROR"

conversation, bedrock_client = init_bedrock()
registered_kbs = kb_manager.load_kbs()

st.title("🔍 문서 기반 질의응답 시스템")
st.caption("실무형 Knowledge Base 관리 및 다중 소스 RAG")

selected_kb_ids = []

with st.sidebar:
    st.header("⚙️ 지식 기반 관리")
    with st.expander("➕ 새 지식 기반 등록"):
        # st.form을 사용하여 입력 내용 자동 초기화 및 에러 방지
        with st.form("registration_form", clear_on_submit=True):
            new_name = st.text_input("KB 이름 (별칭)")
            new_kb_id = st.text_input("Knowledge Base ID")
            new_ds_id = st.text_input("Data Source ID")
            new_bucket = st.text_input("S3 버킷 이름")
            
            submit_button = st.form_submit_button("등록", use_container_width=True)
            
            if submit_button:
                if new_name and new_kb_id and new_ds_id and new_bucket:
                    success, msg = kb_manager.save_kb(new_name, new_kb_id, new_ds_id, new_bucket, "")
                    if success:
                        st.toast(f"✅ {msg}")
                        time.sleep(0.5)
                        st.rerun() # 탭을 닫고 목록을 업데이트하기 위해 재실행
                    else:
                        st.error(msg)
                else:
                    st.warning("모든 필드를 입력해주세요.")

    if registered_kbs:
        st.markdown("---")
        st.subheader("📚 검색 대상 설정")
        kb_names = [kb['name'] for kb in registered_kbs]
        selected_name = st.selectbox("검색할 KB 선택", kb_names)
        selected_kb_info = [kb for kb in registered_kbs if kb['name'] == selected_name]
        selected_kb_ids = [kb['kb_id'] for kb in selected_kb_info]
        
        st.markdown("---")
        st.subheader("📤 문서 추가")
        target_kb_name = st.selectbox("대상 지식 기반 선택", kb_names)
        target_kb = next(kb for kb in registered_kbs if kb['name'] == target_kb_name)
        uploaded_file = st.file_uploader("PDF 업로드", type=['pdf'])
        if st.button("업로드 및 데이터 동기화", use_container_width=True) and uploaded_file:
            try:
                s3 = init_s3()
                file_bytes = io.BytesIO(uploaded_file.getvalue())
                target_key = uploaded_file.name
                
                with st.spinner(f"S3 업로드 중... (Bucket: {target_kb['bucket']})"):
                    s3.upload_fileobj(
                        file_bytes, 
                        target_kb['bucket'], 
                        target_key, 
                        ExtraArgs={'ContentType': 'application/pdf'}
                    )
                st.success(f"S3 업로드 완료: {target_key}")
                
                with st.spinner("KB 데이터 동기화(Ingestion) 요청 중..."):
                    job_id = start_ingestion(target_kb['kb_id'], target_kb['ds_id'])
                    if job_id:
                        st.info(f"동기화 시작됨 (Job ID: {job_id})")
                        status_area = st.empty()
                        while True:
                            status = check_ingestion_status(target_kb['kb_id'], target_kb['ds_id'], job_id)
                            status_area.info(f"현재 동기화 상태: {status}")
                            if status in ['COMPLETE', 'FAILED', 'ERROR']: 
                                break
                            time.sleep(3)
                        
                        if status == 'COMPLETE': 
                            st.success("🎉 지식 기반 데이터 동기화가 성공적으로 완료되었습니다!")
                            st.balloons()
                        else: 
                            st.error(f"❌ 동기화 실패: {status}. AWS 콘솔에서 상세 에러를 확인하세요.")
                    else:
                        st.error("❌ 동기화 요청을 시작할 수 없습니다. KB ID와 Data Source ID를 확인하세요.")
            except Exception as e:
                st.error(f"🚨 치명적 오류 발생: {str(e)}")
                st.info("Tip: AWS 자격 증명(Access Key)과 S3 버킷 권한을 확인해주세요.")
    else:
        st.info("먼저 지식 기반을 등록해주세요.")

    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.chat_history.clear()
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.write(message["content"])

search_query = st.chat_input("질문을 입력하세요...")
if search_query:
    if not selected_kb_ids:
        st.warning("검색할 지식 기반을 선택해주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": search_query})
        with st.chat_message("user"): st.write(search_query)
        try:
            with st.chat_message("assistant"):
                with st.status("답변 생성 중...", expanded=True) as status:
                    results = retrieve_from_kb(
                        search_query, 
                        selected_kb_ids
                    )
                    context = "\n\n".join([r['content']['text'] for r in results])
                    status.update(label=f"검색 완료 ({len(results)}개)", state="complete", expanded=False)
                
                if not results:
                    full_resp = "검색된 결과가 없습니다. 다른 질문을 하시거나 필터를 확인해주세요."
                    st.write(full_resp)
                else:
                    prompt = HumanMessage(content=f"문서 내용을 바탕으로 질문에 답하세요.\n\n질문: {search_query}\n\n내용:\n{context}")
                    resp_placeholder = st.empty()
                    full_resp = ""
                    for chunk in conversation.stream([prompt], config={"configurable": {"session_id": "default"}}):
                        if hasattr(chunk, 'content') and chunk.content:
                            full_resp += chunk.content
                            resp_placeholder.markdown(full_resp)

                    st.session_state.messages.append({"role": "assistant", "content": full_resp})
                    if results:
                        with st.expander("📚 검색 분석 및 출처 확인"):
                            st.info(f"사용한 검색 전략: **SEMANTIC**")
                            for i, r in enumerate(results, 1):
                                score = r.get('score', 0)
                                method = r.get('actual_search_method', 'N/A')
                                st.markdown(f"---")
                                st.markdown(f"**[{i}] 관련도 점수: `{score:.4f}`** (방법: {method})")
                                st.write(r['content']['text'])
                                if 'location' in r: 
                                    st.caption(f"Source: {r['location']['s3Location']['uri']}")
        except Exception as e:
            st.error(f"오류: {str(e)}")
