import boto3
from datasets import Dataset
from langchain_aws import ChatBedrock, BedrockEmbeddings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy

# 폰트 파일 등록
font_location = "./NanumGothic.ttf"
# 폰트 등록
fm.fontManager.addfont(font_location)

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# 한글 텍스트 출력 시 사용할 폰트 명시적 지정
plt.rcParams["font.sans-serif"] = ["NanumGothic"]

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Bedrock 모델 설정
bedrock_model = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.1},
)

# Bedrock 임베딩 설정
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_client, model_id="amazon.titan-embed-text-v1"
)

# 현실적인 테스트 데이터
data = {
    "question": [
        "한국의 수도는 어디인가요?",
        "인공지능의 종류는 무엇이 있나요?",
        "최근 5년간 한국 경제 성장률은 어떻게 변했나요?",
        "영화 '기생충'의 감독은 누구인가요?",
        "2023년 미국 주식 시장은 어땠나요?",
    ],
    "response": [
        "한국의 수도는 서울입니다.",
        "인공지능은 크게 강인공지능과 약인공지능으로 나눌 수 있습니다. 현재 개발된 AI는 대부분 약인공지능에 속합니다.",
        "한국의 경제 성장률은 2019년 2.0%, 2020년 -1.0%, 2021년 4.0%, 2022년 2.5%를 기록했습니다. 2023년은 아직 집계되지 않았습니다. (2024년 5월 15일 기준)",
        "영화 '기생충'은 봉준호 감독이 연출했습니다.",
        "2023년 미국 주식 시장은 상승세를 보였습니다. 하지만 금리 인상 등의 영향으로 변동성이 큰 한 해였습니다.",
    ],
    "reference": [
        "대한민국의 수도는 서울특별시이다.",
        "인공지능은 목표나 능력에 따라 강인공지능과 약인공지능으로 분류할 수 있다.",
        "한국은행 경제통계시스템에 따르면, 한국의 경제 성장률은 2019년 2.0%, 2020년 -1.0%, 2021년 4.0%, 2022년 2.5%를 기록했다.",
        "영화 '기생충'은 봉준호 감독이 각본과 연출을 맡았다.",
        "2023년 미국 주식 시장은 S&P 500 기준으로 20% 이상 상승했다.",
    ],
    "contexts": [
        [
            "대한민국의 수도는 서울특별시이다.",
            "서울은 대한민국의 정치, 경제, 사회, 문화의 중심지이다.",
        ],
        [
            "인공지능은 다양한 기준에 따라 여러 가지로 분류될 수 있다.",
            "일반적으로는 강인공지능과 약인공지능으로 나눌 수 있다.",
        ],
        [
            "한국의 경제 성장률은 한국은행 경제통계시스템에서 확인할 수 있다.",
            "경제 성장률은 국가 경제의 건전성을 나타내는 중요한 지표 중 하나이다.",
        ],
        [
            "봉준호 감독은 한국을 대표하는 영화감독 중 한 명이다.",
            "영화 '기생충'은 2019년 칸 영화제에서 황금종려상을 수상했다.",
        ],
        [
            "미국 주식 시장은 세계 경제에 큰 영향을 미친다.",
            "2023년 미국 주식 시장은 금리 인상, 인플레이션 등의 변수에 영향을 받았다.",
        ],
    ],
}

# Dataset 생성
eval_dataset = Dataset.from_dict(data)

# 평가 메트릭 정의
metrics = [Faithfulness(), ContextRecall(), ContextPrecision(), AnswerRelevancy()]

# 평가 실행
result = evaluate(
    eval_dataset,
    metrics=metrics,
    llm=bedrock_model,
    embeddings=bedrock_embeddings,
)

print(result)

# 결과를 DataFrame으로 변환하고 숫자 데이터만 선택
df = result.to_pandas()
numeric_df = df.select_dtypes(include=[np.number])

# 결과 저장
df.to_csv("evaluation_results.csv")

# 시각화
plt.figure(figsize=(12, 6))
scores = numeric_df.mean()  # 숫자 데이터의 평균만 계산
plt.bar(range(len(scores)), scores.values)
plt.title("평균 메트릭 점수", fontsize=14, pad=20)
plt.xlabel("메트릭", fontsize=12)
plt.ylabel("점수", fontsize=12)
plt.xticks(range(len(scores)), scores.index, rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n=== 평가 결과 ===")
print(df)
print("\n시각화 파일 생성 완료: metrics_bar.png")
