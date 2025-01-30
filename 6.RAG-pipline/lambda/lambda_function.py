import os
import json
import boto3
import psycopg2
import psycopg2.extras
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile

bedrock = boto3.client(service_name='bedrock-runtime',region_name='us-east-1')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # 환경 변수
    DB_HOST = os.environ['DB_HOST']
    DB_NAME = os.environ['DB_NAME']
    DB_USER = os.environ['DB_USER']
    DB_PASSWORD = os.environ['DB_PASSWORD']
    
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    
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
        
        successful_chunks = 0
        
        for chunk in chunks:
            try:
                cleaned_content = chunk.page_content.encode().decode().replace("\x00", "").strip()

                if not cleaned_content:
                    continue

                embedding_response = bedrock.invoke_model(
                    modelId='amazon.titan-embed-text-v1',
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps({
                        "inputText": cleaned_content
                    })
                )
                
                response_body = json.loads(embedding_response.get('body').read().decode())
                embedding_vector = response_body['embedding']
                
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
                
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        conn.commit()
        
        print("Document processed successfully")
        print(f"Total chunks processed: {len(chunks)}")
        print(f"Successfully processed chunks: {successful_chunks}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        
    finally:
        if 'conn' in locals():
            conn.close()
        if 'pdf_path' in locals():
            os.remove(pdf_path)