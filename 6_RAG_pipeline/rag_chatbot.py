import os
import io
import time
import json
import boto3
import streamlit as st
from langchain_aws import ChatBedrock
import psycopg2
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="문서 기반 질의응답 시스템",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def init_bedrock():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    bedrock_embedding = boto3.client("bedrock-runtime", region_name="us-east-1")
    bedrock = ChatBedrock(
        client=bedrock_client,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
    )
    return bedrock, bedrock_embedding

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

def get_embedding(text, bedrock_embedding):
    response = bedrock_embedding.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({"inputText": text})
    )
    response_body = json.loads(response.get('body').read().decode())
    embedding = response_body['embedding']
    return embedding

def find_similar_chunks(query_embedding, k=3):
    """유사한 청크 검색"""
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

def check_documents_exist():
    """데이터베이스에 문서가 존재하는지 확인"""
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT EXISTS(SELECT 1 FROM documents LIMIT 1)")
        return cursor.fetchone()[0]

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("🔍 문서 기반 질의응답 시스템")
st.caption("RAG(Retrieval-Augmented Generation) 기반 문서 검색")

documents_exist = check_documents_exist()

with st.sidebar:
    
    with st.container():
        st.markdown("### 📋 사용 방법")
        st.markdown("""
        1. PDF 파일을 업로드하세요
        2. 처리 버튼을 클릭하세요
        3. 질문을 입력하여 답변을 받으세요
        """)
    
    st.markdown("---")
    
    # 문서가 없을 때만 업로드 UI 표시
    if not documents_exist:
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
                        
                        st.success("✅ PDF가 성공적으로 업로드되었습니다!")
                        
                        with st.spinner("데이터베이스에 문서를 저장하고 있습니다...30초 정도 소요됩니다."):
                            time.sleep(40)
                            
                        if check_documents_exist():
                            st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ 파일 업로드 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("⚠️ PDF 파일을 먼저 업로드해주세요.")
    else:
        st.success("✅ 문서 처리가 완료되었습니다! 질문을 입력해주세요.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("궁금한 내용을 입력하세요...")
if prompt:
    if not documents_exist:  
        st.warning("⚠️ 먼저 문서를 업로드해주세요!")
    else:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하고 있습니다..."):
                    bedrock, bedrock_embedding = init_bedrock()
                    
                    # 1. 질문의 임베딩 생성
                    query_embedding = get_embedding(prompt, bedrock_embedding)
                    
                    # 2. 유사한 문서 검색
                    similar_chunks = find_similar_chunks(query_embedding)
                    
                    # 3. 컨텍스트 구성
                    context = "\n\n".join([chunk[0] for chunk in similar_chunks])
                    
                    # 4. 프롬프트 구성
                    prompt_with_context = f"""다음은 학사 정보에 대한 질문과 관련 문서 내용입니다:
    
    질문: {prompt}
    
    관련 문서 내용:
    {context}
    
    위 내용을 바탕으로 질문에 대해 명확하고 친절하게 답변해주세요. 
    문서에 없는 내용은 언급하지 말고, 확실한 정보만 답변에 포함해주세요."""
                    
                    # 5. 답변 생성
                    response = bedrock.invoke(prompt_with_context)
                    
                    st.write(response.content)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.content
                    })
                    
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
                            "유사도 검색 결과 수": len(similar_chunks)
                        })
        except Exception as e:
            st.error(f"⚠️ 답변 생성 중 오류가 발생했습니다: {str(e)}")