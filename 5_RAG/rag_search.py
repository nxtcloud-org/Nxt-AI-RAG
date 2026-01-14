import boto3
import streamlit as st
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb

# 사이드바 자동 숨김 설정
st.set_page_config(initial_sidebar_state="collapsed")
st.title("🔍 학사 정보 검색 시스템")
st.caption("RAG(Retrieval-Augmented Generation) 기반 문서 검색")


# AWS Bedrock 클라이언트 초기화
@st.cache_resource
def init_bedrock():
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    bedrock = ChatBedrock(
        client=bedrock_client,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"anthropic_version": "bedrock-2023-05-31"},
    )
    embeddings = BedrockEmbeddings(region_name="us-east-1")
    return bedrock, embeddings


bedrock, embeddings = init_bedrock()


# PDF 로드 및 청크 분할
@st.cache_resource
def load_and_process_pdf():
    chroma_client = chromadb.PersistentClient(path="./vector_db")
    # 파일로드
    pdf_loader = PyMuPDFLoader("./data/univ-data.pdf")
    # 청크 분할
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
    )
    data = pdf_loader.load_and_split(text_splitter=splitter)
    # 벡터 스토어 구성
    vectorstore = Chroma.from_documents(
        documents=data,
        embedding=embeddings,
        persist_directory="./vector_db/",
        collection_name="university_docs",
    )
    return vectorstore


# 메인 검색 인터페이스
try:
    vectorstore = load_and_process_pdf()
    st.success("문서가 성공적으로 로드되었습니다.")
    st.header("📚 학사 정보 검색")

    search_query = st.text_input(
        "궁금한 내용을 자연어로 입력하세요:",
        placeholder="예: 졸업요건이 뭐야?",
        key="search_query",
    )

    if search_query:  # 검색어가 입력된 경우에만 검색 실행
        results = vectorstore.similarity_search(search_query, k=3)

        # 중복 제거
        seen_contents = set()
        unique_results = []
        for doc in results:
            content = doc.page_content.strip()
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append(doc)

        st.write(f"🎯 검색 결과: {len(unique_results)}개 관련 문서 발견")
        for i, doc in enumerate(unique_results, 1):
            with st.expander(f"검색 결과 #{i}"):
                st.markdown(f"**내용:**\n{doc.page_content}")
                st.caption(f"출처: {doc.metadata.get('page', 'N/A')}페이지")

        # AI 답변 생성
        st.write("---")
        st.subheader("🤖 AI 응답")

        with st.spinner("AI가 답변을 생성하고 있습니다..."):
            # 검색 결과를 하나의 컨텍스트로 결합
            context = "\n".join([doc.page_content for doc in unique_results])

            # 프롬프트 구성
            prompt = f"""다음은 학사 정보에 대한 질문과 관련 문서 내용입니다:
    
    질문: {search_query}
    
    관련 문서 내용:
    {context}
    
    위 내용을 바탕으로 질문에 대해 명확하고 친절하게 답변해주세요. 
    문서에 없는 내용은 언급하지 말고, 확실한 정보만 답변에 포함해주세요.
    """

            # AI 응답 생성 및 표시
            response = bedrock.invoke(prompt)

            # 응답 내용과 메타데이터 분리하여 표시
            st.markdown("**답변 내용:**")
            st.markdown(response.content)

            # 메타데이터를 접을 수 있는 expander로 표시
            with st.expander("📊 응답 메타데이터"):
                st.json(
                    {
                        "토큰 사용량": response.additional_kwargs["usage"],
                        "모델": response.additional_kwargs["model_id"],
                        "응답 ID": response.id,
                    }
                )

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.error("데이터베이스 초기화에 실패했습니다. 'vector_db' 디렉토리를 확인해주세요.")
