import os
import io
import time
import json
import boto3
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(
    page_title="문서 기반 질의응답 시스템",
    page_icon="📚",
    layout="wide"
)

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()  
if 'messages' not in st.session_state:
    st.session_state.messages = [] 
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = True

def get_session_history(session_id):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    return st.session_state.chat_history

@st.cache_resource
def init_bedrock():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    
    llm = ChatBedrock(
        client=bedrock_client,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
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

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.autocommit = True  
    return conn

def get_embedding(text, bedrock_client):
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v1"
    )
    return embeddings.embed_query(text)

def find_similar_chunks(query_embedding, k=3):
    """검색"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT content, metadata,
                   1 - (embedding <=> %s::vector) as similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, k))
        
        results = cursor.fetchall()
        return [(row[0], row[1]) for row in results]
    finally:
        cursor.close()
        conn.close()

def check_base_documents():
    """문서 존재 여부 확인"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT EXISTS(SELECT 1 FROM documents LIMIT 1)")
        return cur.fetchone()[0]
    except Exception as e:
        print(f"Error checking base documents: {str(e)}")
        return False
    finally:
        cur.close()
        conn.close()

def check_recent_upload():
    """최근 업로드된 문서 확인"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 
                FROM documents 
                WHERE created_at >= NOW() - INTERVAL '2 minute'
            )
        """)
        return cur.fetchone()[0]
    except Exception as e:
        print(f"Error checking recent upload: {str(e)}")
        return False
    finally:
        cur.close()
        conn.close()

conversation, bedrock_client = init_bedrock()
docs_exist = check_base_documents()

st.title("🔍 문서 기반 질의응답 시스템")
st.caption("RAG(Retrieval-Augmented Generation) 기반 문서 검색")

with st.sidebar:
    with st.container():
        st.markdown("### 📋 사용 방법")
        st.markdown("""
        1. PDF 파일을 업로드하세요
        2. 처리 버튼을 클릭하세요
        3. 질문을 입력하여 답변을 받으세요
        """)
    
    st.markdown("---")
    
    if not st.session_state.show_upload:
        if st.button("📄 문서 추가하기", key="add_doc"):
            st.session_state.show_upload = True
    
    if not docs_exist or st.session_state.show_upload:
        with st.container():
            st.markdown("### 📤 문서 업로드")
            uploaded_file = st.file_uploader(
                "PDF 파일을 드래그하세요",
                type=['pdf'],
                help="PDF 형식만 가능"
            )
            
            if st.button("🚀 처리 시작", key="process_button", use_container_width=True):
                if uploaded_file:
                    try:
                        s3 = init_s3()
                        bucket_name = os.getenv('BUCKET_NAME')
                        file_name = uploaded_file.name
                        file_bytes = io.BytesIO(uploaded_file.getvalue())
                        
                        with st.spinner("문서를 처리하고 있습니다..."):
                            s3.upload_fileobj(
                                file_bytes,
                                bucket_name,
                                f"{file_name}",
                                ExtraArgs={'ContentType': 'application/pdf'}
                            )
                        
                        st.toast("✅ PDF가 성공적으로 업로드되었습니다!", icon="✅")
                        
                        with st.spinner("데이터베이스에 문서를 저장하고 있습니다...30초 정도 소요됩니다."):
                            time.sleep(30)

                        is_recent = check_recent_upload()
                        if is_recent:
                            st.session_state.show_upload = False
                            st.toast("✅ 문서 처리가 완료되었습니다! 질문을 입력해주세요.", icon="✅")
                            st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ 파일 업로드 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("⚠️ PDF 파일을 먼저 업로드해주세요.")

    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.chat_history.clear()
        st.session_state.messages = []
        st.toast("대화 기록이 초기화되었습니다.", icon="✅")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

search_query = st.chat_input("예: 졸업요건이 뭐야?")
if search_query:

    st.session_state.messages.append({"role": "user", "content": search_query})
    
    if not docs_exist:  
        st.warning("⚠️ 먼저 문서를 업로드해주세요!")
    else:
        # 사용자 질문만 화면에 표시
        with st.chat_message("user"):
            st.write(search_query)
        
        try:
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하고 있습니다..."):
                    # 1. 질문의 임베딩 생성
                    query_embedding = get_embedding(search_query, bedrock_client)
                    
                    # 2. 유사한 문서 검색
                    similar_chunks = find_similar_chunks(query_embedding)
                    
                    # 3. 컨텍스트 구성
                    context = "\n\n".join([chunk[0] for chunk in similar_chunks])
                    
                    # 4. 프롬프트 구성 
                    prompt = HumanMessage(content=f"""이전 대화 기록과 문서 내용을 참고하여 답변해주세요.
                    
                    질문: {search_query}
                    
                    관련 문서 내용:
                    {context}
                    
                    위 내용과 이전 대화 맥락을 바탕으로 질문에 대해 명확하고 친절하게 답변해주세요. 
                    문서에 없는 내용은 언급하지 말고, 확실한 정보만 답변에 포함해주세요.""")
                    
                    # 5. 답변 생성 및 표시
                    response = conversation.invoke(
                        [prompt],
                        config={"configurable": {"session_id": "default"}}
                    )
                    
                    response_content = response.content if hasattr(response, 'content') else str(response)
                    st.markdown(response_content)

                    # 6. 참고한 문서 표시
                    with st.expander("📚 참고한 문서"):
                        for i, (content, metadata) in enumerate(similar_chunks, 1):
                            st.markdown(f"**문서 {i}:**")
                            st.write(content)
                            if metadata:
                                st.caption(f"출처: {metadata.get('page', 'N/A')}페이지")

                    with st.expander("📊 상세 정보"):
                        st.json({
                            "모델": response.additional_kwargs.get("model_id", "N/A"),
                            "토큰 사용량": response.additional_kwargs.get("usage", {}),
                            "응답 ID": response.id
                        })

            st.session_state.messages.append({"role": "assistant", "content": response_content})
        except Exception as e:
            st.error(f"⚠️ 답변 생성 중 오류가 발생했습니다: {str(e)}")