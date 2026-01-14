"""
RAG 검색 품질 평가 스크립트

세 가지 RAG 시스템의 검색 성능을 비교 평가합니다:
- ChromaDB RAG (Local)
- PostgreSQL RAG (Production)
- AWS Knowledge Bases (Managed)

평가 메트릭: Context Recall, Context Precision, Faithfulness, Answer Relevancy
"""

import os
import time
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dotenv import load_dotenv
from typing import List, Dict
import chromadb

# LangChain imports
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import ContextRecall, ContextPrecision, Faithfulness, AnswerRelevancy

load_dotenv()

# 한글 폰트 설정
try:
    font_location = "../NanumGothic.ttf"
    if os.path.exists(font_location):
        fm.fontManager.addfont(font_location)
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
except:
    plt.rcParams["font.family"] = "DejaVu Sans"

# 테스트 데이터셋
TEST_DATASET = {
    "questions": [
        "당초입학자의 일반휴학 가능 최대학기는 몇 학기인가요?",
        "입학 후 첫 학기에 일반휴학을 할 수 없는 대상자는 누구인가요?",
        "복학 신청 기간은 언제인가요?",
        "입대휴학생은 제대 후 언제까지 복학해야 하나요?",
        "캠퍼스 내 소속 변경을 신청할 수 있는 자격은 무엇인가요?",
        "조기졸업을 하려면 어떤 요건을 충족해야 하나요?",
        "재입학이 가능한 대상자는 누구인가요?",
        "캠퍼스내 복수전공 신청은 언제부터 가능한가요?",
        "3학년 편입자의 학점 인정 범위는 어떻게 되나요?",
        "전적대학에서 이수한 과목을 우리 대학 과목으로 대체 인정받으려면 어떤 조건이 필요한가요?",
    ],
    "reference_answers": [
        "당초입학자는 6학기까지 일반휴학이 가능합니다. 단, 건축학(5년제)의 경우 7학기까지 가능합니다.",
        "신입생, 편입생, 졸업예정자 복수전공생, 재입학생은 입학 후 첫 학기에 일반휴학을 할 수 없습니다.",
        "복학 신청기간은 매년 1월 중순부터 2월 간 또는 7월 중순부터 8월에 있으며, 1월 또는 7월 초 학교 홈페이지 및 넥클포탈시스템 휴/복학 공지사항을 통해 안내됩니다.",
        "입대휴학생은 반드시 제대 후 1년 이내에 복학하여야 합니다. 예를 들어 2023년 8월 제대자의 경우 2023년 8월 또는 2024년 2월 중 정해진 복학기간 내에 복학해야 합니다.",
        "전공배정 후 3학기 이상 수료한 학생으로서 6학기 이내에 지원 가능합니다. 클라우드, AI, IT 관련 학과 학생의 경우 전공 간 소속 변경이 가능합니다.",
        "조기졸업을 위해서는 총 6학기 이상을 이수한 자로서 졸업에 필요한 모든 요건을 충족하고, 계절학기를 포함하여 취득한 모든 교과목을 통산한 성적의 평량평균이 3.75 이상이어야 합니다.",
        "당초입학생으로 미등록, 미복학, 성적불량(제적일로부터 2년 경과 후), 학기초과, 자원퇴학 제적자, 편입생(일반, 학사) 및 소속변경생으로서 제적자가 재입학을 신청할 수 있습니다.",
        "00학번 이후 학생으로서 3학기부터 졸업직전 학기까지 신청 가능합니다. 제1전공이 승인된 학생이어야 하며 휴학생도 신청 가능합니다.",
        "3학년 편입자는 졸업학점의 1/2이내에서 학점 인정이 가능합니다. 단, 학사편입과 졸업예정자 복수전공생은 과목인정 방식으로 처리됩니다.",
        "전적대학에서 이수한 과목이 우리대학에서 개설하고 있는 과목과 교과내용이 거의 유사해야 하며, 담당교수의 확인을 거쳐 대체 인정됩니다.",
    ],
    "gold_contexts": [
        ["재학 중 일반휴학 가능 최대학기\n당초입학자 : 6학기 (단, 건축학(5년제)는 7학기)"],
        ["입학 후 첫 학기 일반휴학 불가 대상자\n신입생, 편입생, 졸업예정자 복수전공생, 재입학생은 입학 후 첫 학기에 일반휴학을 불허함"],
        ["신청기간\n신청기간 : 매년 1월 중순2월 간 또는 7월 중순8월에 있으며, 1월 또는 7월 초 학교 홈페이지 및 넥클포탈시스템 휴/복학 공지사항을 통해 안내"],
        ["입대휴학생은 반드시 제대 후 1년 이내에 복학하여야 함(예: 2023. 8월 제대자의 경우 - 2023. 8월 또는 2024. 2월 중 정해진 복학기간 내에 복학해야 함)"],
        ["지원자격\n전공배정 후 3학기 이상 수료한 학생으로서 6학기 이내에 지원 가능\n클라우드, AI, IT 관련 학과 학생의 경우 전공 간 소속 변경 가능"],
        ["조기졸업 요건\n총 6학기 이상을 이수한 자로서 졸업에 필요한 모든 요건을 충족하는 자\n계절학기를 포함하여 취득한 모든 교과목을 통산한 성적의 평량평균이 3.75 이상인 자"],
        ["자격 대상\n당초입학생으로 미등록, 미복학, 성적불량(제적일로부터 2년 경과 후), 학기초과, 자원퇴학 제적자, 편입생(일반, 학사) 및 소속변경생으로서 제적자"],
        ["신청요건\n제1전공이 승인된 학생이어야 함(휴학생도 신청 가능)\n00학번 이후 학생으로서 3학기부터 졸업직전 학기까지 신청 가능"],
        ["인정범위\n2학년 편입자 : 졸업학점의 1/4이내 인정\n3학년 편입자 : 졸업학점의 1/2이내 인정(학사편입, 졸업예정자 복수전공생은 과목인정)"],
        ["학점 인정 시 참고사항\n전적대학에서 이수한 과목이 우리대학에서 개설하고 있는 과목과 교과내용이 거의 유사할 경우에 담당교수의 확인을 거쳐 대체 인정함"],
    ],
}


class ChromaDBRetriever:
    """ChromaDB 기반 RAG 시스템"""

    def __init__(self, pdf_path: str = None, vector_db_path: str = None):
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id="amazon.titan-embed-text-v1"
        )

        if pdf_path and vector_db_path:
            self.vectorstore = self._load_vectorstore(pdf_path, vector_db_path)

    def _load_vectorstore(self, pdf_path: str, vector_db_path: str):
        """PDF 로드 및 벡터 스토어 생성"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_loader = PyMuPDFLoader(pdf_path)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
        )
        documents = pdf_loader.load_and_split(text_splitter=splitter)

        try:
            vectorstore = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embeddings,
                collection_name="university_docs"
            )
            print(f"Loaded existing vector store from {vector_db_path}")
        except:
            print(f"Creating new vector store from {len(documents)} documents...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=vector_db_path,
                collection_name="university_docs",
            )

        return vectorstore

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """검색 수행"""
        results = self.vectorstore.similarity_search(query, k=k)

        # 중복 제거
        seen_contents = set()
        unique_results = []
        for doc in results:
            content = doc.page_content.strip()
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append(content)

        return unique_results[:k]

    def get_system_name(self) -> str:
        return "ChromaDB RAG"


class PostgreSQLRetriever:
    """PostgreSQL + pgvector 기반 RAG 시스템"""

    def __init__(self):
        import psycopg2

        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id="amazon.titan-embed-text-v1"
        )

        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "rag_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
        }

    def _get_connection(self):
        import psycopg2
        return psycopg2.connect(**self.db_config)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """검색 수행"""
        try:
            query_embedding = self.embeddings.embed_query(query)

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT content
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, k))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return [row[0] for row in results]

        except Exception as e:
            print(f"PostgreSQL retrieval error: {e}")
            return []

    def get_system_name(self) -> str:
        return "PostgreSQL RAG"


class AWSKnowledgeBaseRetriever:
    """AWS Knowledge Bases 기반 RAG 시스템"""

    def __init__(self, knowledge_base_ids: List[str]):
        self.kb_ids = knowledge_base_ids
        self.bedrock_agent_runtime = boto3.client(
            'bedrock-agent-runtime', 
            region_name="us-east-1"
        )

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """검색 수행"""
        all_retrieval_results = []

        retrieval_config = {
            'vectorSearchConfiguration': {
                'numberOfResults': k,
                'overrideSearchType': 'SEMANTIC'
            }
        }

        for kb_id in self.kb_ids:
            if not kb_id:
                continue

            try:
                response = self.bedrock_agent_runtime.retrieve(
                    knowledgeBaseId=kb_id,
                    retrievalConfiguration=retrieval_config,
                    retrievalQuery={'text': query}
                )

                results = response.get('retrievalResults', [])
                for result in results:
                    content = result.get('content', {}).get('text', '')
                    if content:
                        all_retrieval_results.append(content)

            except Exception as e:
                print(f"AWS KB retrieval error for {kb_id}: {e}")

        return all_retrieval_results[:k]

    def get_system_name(self) -> str:
        return "AWS Knowledge Bases"


class RetrievalEvaluator:
    """RAG 검색 품질 평가기"""

    def __init__(self, retrievers: List, questions: List[str], gold_contexts: List[List[str]]):
        self.retrievers = retrievers
        self.questions = questions
        self.gold_contexts = gold_contexts

        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={"temperature": 0.1},
        )
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client, 
            model_id="amazon.titan-embed-text-v1"
        )

    def collect_retrieval_results(self) -> Dict[str, List[List[str]]]:
        """모든 시스템에서 검색 결과 수집"""
        all_results = {}

        for retriever in self.retrievers:
            system_name = retriever.get_system_name()
            print(f"\nRetrieving from {system_name}...")

            system_results = []
            for i, question in enumerate(self.questions):
                start_time = time.time()
                contexts = retriever.retrieve(question, k=3)
                retrieval_time = time.time() - start_time

                print(f"  [{i+1}/{len(self.questions)}] {retrieval_time:.3f}s - {len(contexts)} contexts")
                system_results.append(contexts)

            all_results[system_name] = system_results

        return all_results

    def create_evaluation_dataset(
        self, 
        system_name: str, 
        retrieved_contexts: List[List[str]], 
        answers: List[str] = None
    ) -> Dataset:
        """RAGAS 평가용 데이터셋 생성"""
        ground_truths = [
            "\n".join(contexts) if contexts else "" 
            for contexts in self.gold_contexts
        ]

        data = {
            "question": self.questions,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths,
        }

        if answers is not None:
            data["answer"] = answers

        return Dataset.from_dict(data)

    def evaluate_system(
        self, 
        system_name: str, 
        retrieved_contexts: List[List[str]], 
        answers: List[str] = None
    ):
        """특정 시스템 평가"""
        print(f"\nEvaluating {system_name}...")

        eval_dataset = self.create_evaluation_dataset(system_name, retrieved_contexts, answers)

        # 답변이 있으면 생성 메트릭도 평가
        if answers is not None:
            metrics = [
                ContextRecall(),
                ContextPrecision(),
                Faithfulness(),
                AnswerRelevancy()
            ]
        else:
            metrics = [ContextRecall(), ContextPrecision()]

        try:
            result = evaluate(
                eval_dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return result.to_pandas()

        except Exception as e:
            print(f"Error evaluating {system_name}: {e}")
            return None

    def compare_systems(
        self, 
        all_results: Dict[str, List[List[str]]], 
        all_answers: Dict[str, List[str]] = None
    ):
        """모든 시스템 비교 평가"""
        all_dfs = {}

        for system_name, contexts in all_results.items():
            system_answers = None
            if all_answers and system_name in all_answers:
                system_answers = all_answers[system_name]

            df = self.evaluate_system(system_name, contexts, system_answers)
            if df is not None:
                all_dfs[system_name] = df

        return all_dfs

    def create_comparison_report(self, all_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """시스템 비교 리포트 생성"""
        comparison_data = []

        for system_name, df in all_dfs.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                comparison_data.append({
                    "System": system_name,
                    "Metric": col,
                    "Mean": df[col].mean(),
                    "Std": df[col].std(),
                    "Min": df[col].min(),
                    "Max": df[col].max(),
                })

        return pd.DataFrame(comparison_data)

    def visualize_comparison(self, comparison_df: pd.DataFrame, output_path: str = None):
        """비교 결과 시각화"""
        pivot_df = comparison_df.pivot(index="Metric", columns="System", values="Mean")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 막대 그래프
        x = np.arange(len(pivot_df.index))
        width = 0.25

        for i, system in enumerate(pivot_df.columns):
            axes[0].bar(x + i * width, pivot_df[system], width, label=system)

        axes[0].set_xlabel('Metric')
        axes[0].set_ylabel('Score')
        axes[0].set_title('RAG Retrieval Performance Comparison')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(pivot_df.index, rotation=45)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1])

        # 히트맵
        im = axes[1].imshow(pivot_df.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        axes[1].set_xticks(np.arange(len(pivot_df.columns)))
        axes[1].set_yticks(np.arange(len(pivot_df.index)))
        axes[1].set_xticklabels(pivot_df.columns)
        axes[1].set_yticklabels(pivot_df.index)
        axes[1].set_title('RAG Performance Heatmap')

        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                axes[1].text(
                    j, i, f'{pivot_df.values[i, j]:.2f}',
                    ha="center", va="center", color="black"
                )

        fig.colorbar(im, ax=axes[1])
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {output_path}")

        plt.show()

    def save_results(
        self, 
        all_dfs: Dict[str, pd.DataFrame], 
        comparison_df: pd.DataFrame, 
        output_dir: str = "."
    ):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        for system_name, df in all_dfs.items():
            filename = f"{output_dir}/eval_{system_name.replace(' ', '_')}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")

        comparison_filename = f"{output_dir}/comparison.csv"
        comparison_df.to_csv(comparison_filename, index=False)
        print(f"Saved {comparison_filename}")


def main():
    print("RAG Retrieval Quality Evaluation\n")

    retrievers = []

    # ChromaDB
    try:
        chroma_retriever = ChromaDBRetriever(
            pdf_path="../5_RAG/data/univ-data.pdf",
            vector_db_path="../5_RAG/vector_db"
        )
        retrievers.append(chroma_retriever)
        print("✓ ChromaDB Retriever initialized")
    except Exception as e:
        print(f"✗ ChromaDB: {e}")

    # PostgreSQL
    if os.getenv("DB_HOST"):
        try:
            pg_retriever = PostgreSQLRetriever()
            retrievers.append(pg_retriever)
            print("✓ PostgreSQL Retriever initialized")
        except Exception as e:
            print(f"✗ PostgreSQL: {e}")

    # AWS Knowledge Bases
    kb_ids = [kb_id.strip() for kb_id in os.getenv("AWS_KB_IDS", "").split(",") if kb_id.strip()]
    if kb_ids:
        try:
            kb_retriever = AWSKnowledgeBaseRetriever(knowledge_base_ids=kb_ids)
            retrievers.append(kb_retriever)
            print(f"✓ AWS KB Retriever initialized ({len(kb_ids)} KBs)")
        except Exception as e:
            print(f"✗ AWS KB: {e}")

    if not retrievers:
        print("Error: No retrievers initialized.")
        return

    evaluator = RetrievalEvaluator(
        retrievers=retrievers,
        questions=TEST_DATASET["questions"],
        gold_contexts=TEST_DATASET["gold_contexts"],
    )

    all_results = evaluator.collect_retrieval_results()
    all_dfs = evaluator.compare_systems(all_results)

    if not all_dfs:
        print("Error: No evaluation results generated.")
        return

    comparison_df = evaluator.create_comparison_report(all_dfs)

    print("\nEVALUATION SUMMARY")
    print(comparison_df.to_string(index=False))

    evaluator.visualize_comparison(comparison_df, output_path="retrieval_comparison.png")
    evaluator.save_results(all_dfs, comparison_df, output_dir="./evaluation_results")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()