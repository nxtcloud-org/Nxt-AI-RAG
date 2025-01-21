import json
import boto3
import streamlit as st

from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings

from langchain.store import LocalFileStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain.embeddings import CacheBackedEmbeddings

# from langchain_aws.vectorstores import InMemoryVectorStore

# AWS Bedrock 클라이언트 초기화
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# LangChain BedrockChat 초기화
bedrock = ChatBedrock(
    client=bedrock_client,  # Bedrock 클라이언트
    model_id="anthropic.claude-3-haiku-20240307-v1:0",  # Claude v3 모델 ID
    model_kwargs={"anthropic_version": "bedrock-2023-05-31"},  # Bedrock 모델 설정
)
embeddings = BedrockEmbeddings(region_name="us-east-1")

cache_dir = LocalFileStore("./.cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# vector_data = embeddings.embed_query("cloud")
# print(vector_data)
# print(len(vector_data))

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100,
)

pdf_loader = PyPDFLoader("./data/univ-data.pdf")
pdf = pdf_loader.load()

data = pdf_loader.load_and_split(text_splitter=splitter)
vectorstore = Chroma.from_documents(data, embeddings)
result = vectorstore.similarity_search("졸업요건이 뭐지?")
print(len(result))
print(result)
