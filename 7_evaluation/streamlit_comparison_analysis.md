# RAGAS vs OpenEvals Streamlit 앱 비교 분석

## 개요

이 문서는 `ragas_streamlit.py`와 `openevals_streamlit.py` 두 평가 시스템의 공통점과 차이점을 분석합니다.

## 📊 기본 정보

| 항목            | RAGAS              | OpenEvals                   |
| --------------- | ------------------ | --------------------------- |
| 파일명          | ragas_streamlit.py | openevals_streamlit.py      |
| 총 라인 수      | 708줄              | 761줄                       |
| 주요 라이브러리 | RAGAS, AWS Bedrock | OpenEvals, Anthropic Claude |
| 평가 방식       | LLM + 임베딩       | LLM + 임베딩 하이브리드     |

## 🤝 공통점

### 1. 기본 구조 및 UI

- **Streamlit 기반**: 두 앱 모두 Streamlit을 사용한 웹 대시보드
- **3탭 구조**: "평가 결과", "평가 실행", "설정" 탭으로 구성
- **한국어 UI**: 모든 인터페이스가 한국어로 제공
- **반응형 레이아웃**: `layout="wide"` 설정으로 넓은 화면 활용

### 2. 공통 라이브러리

```python
# 공통 임포트
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import tempfile
import importlib
```

### 3. 데이터 처리

- **CSV 파일 저장**: 평가 결과를 CSV 형태로 저장
- **모델별 결과 관리**: 모델명을 포함한 파일명으로 결과 구분
- **세션 상태 관리**: Streamlit session_state를 활용한 상태 관리

### 4. 시각화 기능

- **바 차트**: 평균 메트릭 점수 표시
- **히트맵**: 질문별 메트릭 점수 매트릭스
- **상세 결과 테이블**: 질문별 상세 평가 결과
- **확장 가능한 섹션**: expander를 활용한 상세 정보 표시

### 5. 테스트 데이터 관리

- **기본 샘플 데이터**: 5개의 한국어 질문-답변 쌍
- **동적 항목 수 조정**: 1-10개 항목까지 조정 가능
- **데이터 초기화 기능**: 전체 지우기/기본 데이터 불러오기

## 🔄 차이점

### 1. 평가 라이브러리 및 모델

| 구분                | RAGAS             | OpenEvals                  |
| ------------------- | ----------------- | -------------------------- |
| **평가 라이브러리** | RAGAS             | OpenEvals                  |
| **LLM 제공자**      | AWS Bedrock       | Anthropic Claude           |
| **모델 종류**       | Claude 3, Llama 3 | Claude 3 Haiku/Sonnet/Opus |
| **임베딩 모델**     | Amazon Titan      | SentenceTransformer        |

### 2. 임포트 차이점

#### RAGAS 전용

```python
import boto3
from datasets import Dataset
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy
```

#### OpenEvals 전용

```python
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import openevals
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

### 3. 평가 메트릭

#### RAGAS (4개 메트릭)

- **Faithfulness**: 충실도 - 컨텍스트 기반 사실성
- **Context Recall**: 컨텍스트 재현율 - 필요 정보 검색률
- **Context Precision**: 컨텍스트 정밀도 - 관련 정보 비율
- **Answer Relevancy**: 응답 관련성 - 질문-답변 관련성

#### OpenEvals (5개 메트릭)

- **Accuracy**: 정확성 - 참조 답변과의 일치도 (하이브리드)
- **Completeness**: 완전성 - 답변의 충분성 (LLM)
- **Relevance**: 관련성 - 질문과의 관련성 (하이브리드)
- **Clarity**: 명확성 - 답변의 명확성 (LLM)
- **Semantic Similarity**: 의미적 유사도 - 임베딩 기반 유사도

### 4. 평가 방식

#### RAGAS

```python
# 기존 RAGAS 라이브러리 사용
result = evaluate(
    eval_dataset,
    metrics=metrics,
    llm=bedrock_model,
    embeddings=bedrock_embeddings,
)
```

#### OpenEvals

```python
# 하이브리드 평가 시스템
# 1. LLM 기반 평가
judge = openevals.create_llm_as_judge(prompt=prompt, judge=model)
llm_score = judge(question, response, reference)

# 2. 임베딩 기반 평가
embedding_score = calculate_semantic_similarity(text1, text2, embedding_model)

# 3. 하이브리드 점수 (특정 메트릭)
hybrid_score = 0.7 * llm_score + 0.3 * embedding_score
```

### 5. 폰트 처리

#### RAGAS

```python
# 한글 폰트 다운로드 및 설정
def setup_korean_font():
    font_path = "./NanumGothic.ttf"
    # 폰트 다운로드 로직
    plt.rcParams["font.family"] = "NanumGothic"
```

#### OpenEvals

```python
# 기본 폰트 사용 (영어 차트)
def setup_korean_font():
    plt.rcParams["font.family"] = "DejaVu Sans"
```

### 6. 환경 설정

#### RAGAS

- AWS 자격 증명 필요
- Bedrock 서비스 접근 권한
- 리전 설정 (us-east-1)

#### OpenEvals

- Anthropic API 키 필요
- `.env` 파일 환경 변수 관리
- `python-dotenv` 사용

### 7. 데이터 구조

#### RAGAS

```python
# Dataset 객체 사용
eval_dataset = Dataset.from_dict({
    "question": [...],
    "response": [...],
    "reference": [...],
    "contexts": [...]  # 컨텍스트 필수
})
```

#### OpenEvals

```python
# 딕셔너리 리스트 사용
eval_data = [
    {
        "question": "...",
        "response": "...",
        "reference": "..."  # 컨텍스트 불필요
    }
]
```

## 🎯 핵심 차이점 요약

### 1. 평가 철학

- **RAGAS**: RAG 시스템 전용, 컨텍스트 기반 평가 중심
- **OpenEvals**: 범용 QA 평가, 직접적인 질문-답변 평가

### 2. 기술적 접근

- **RAGAS**: 기존 검증된 라이브러리 활용
- **OpenEvals**: 커스텀 하이브리드 평가 시스템 구현

### 3. 임베딩 활용

- **RAGAS**: 라이브러리 내부에서 자동 처리
- **OpenEvals**: 명시적 임베딩 계산 및 하이브리드 점수

### 4. 확장성

- **RAGAS**: RAGAS 라이브러리 업데이트에 의존
- **OpenEvals**: 독립적인 메트릭 추가/수정 가능

## 📈 성능 및 사용성

### RAGAS 장점

- ✅ 검증된 RAG 평가 메트릭
- ✅ 학술적 근거가 있는 평가 방식
- ✅ 컨텍스트 품질 평가 가능
- ✅ 안정적인 라이브러리

### OpenEvals 장점

- ✅ 유연한 커스텀 평가 시스템
- ✅ 하이브리드 평가로 더 정교한 점수
- ✅ 빠른 API 응답 (Anthropic)
- ✅ 의미적 유사도 직접 측정

### 각각의 한계

- **RAGAS**: 컨텍스트 필수, AWS 의존성
- **OpenEvals**: 학술적 검증 부족, 커스텀 구현

## 🔮 결론

두 시스템은 서로 다른 평가 철학을 가지고 있습니다:

- **RAGAS**는 RAG 시스템의 전체적인 성능을 평가하는 데 특화
- **OpenEvals**는 질문-답변의 직접적인 품질을 다각도로 평가

실제 사용 시에는 평가 목적에 따라 선택하거나, 두 시스템을 병행하여 더 종합적인 평가를 수행하는 것이 권장됩니다.
