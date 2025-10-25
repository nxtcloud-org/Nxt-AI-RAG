import os
import json
import tempfile
import boto3
import psycopg2
import psycopg2.extras
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

s3_client = boto3.client('s3')
bedrock_client = boto3.client(service_name='bedrock-runtime',region_name='us-east-1')
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")

def lambda_handler(event, context):
    # 환경 변수
    DB_HOST = os.environ['DB_HOST']
    DB_NAME = os.environ['DB_NAME']
    DB_USER = os.environ['DB_USER']
    DB_PASSWORD = os.environ['DB_PASSWORD']
    
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    
    print(f"처리 시작 - 버킷이름: {bucket_name}")
    print(f"처리 파일: {file_key}")
    
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            s3_client.download_fileobj(bucket_name, file_key, tmp_file)
            pdf_path = tmp_file.name
        
        pdf_loader = PyPDFLoader(pdf_path)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=50,
            separator='\n'
        )
        chunks = pdf_loader.load_and_split(text_splitter=splitter)
        
        # PostgreSQL 연결
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        print(f"데이터베이스 연결 완료")
        print(f"데이터베이스 호스트: {DB_HOST}")
        print(f"데이터베이스 이름: {DB_NAME}")
        print(f"데이터베이스 사용자: {DB_USER}")
        print(f"총 처리할 청크 개수: {len(chunks)}")
        
        successful_chunks = 0
        
        for chunk in chunks:
            cleaned_content = chunk.page_content.encode().decode().replace("\x00", "").strip()

            if not cleaned_content:
                continue

            embedding_vector = embeddings.embed_query(cleaned_content)
            
            metadata = {
                'page': chunk.metadata.get('page', 0)+1
            }
            
            # 데이터베이스에 저장
            cursor.execute("""
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s, %s)
            """, (
                cleaned_content,
                embedding_vector,
                json.dumps(metadata)
            ))
            successful_chunks += 1
            
        conn.commit()
        
        print(f"문서 처리 완료")
        print(f"성공적으로 처리된 청크 개수: {successful_chunks}")
        
    except Exception as e:
        print(f"PDF 파일 처리 중 오류 발생: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        
    finally:
        if 'conn' in locals():
            conn.close()
        if 'pdf_path' in locals():
            os.remove(pdf_path)