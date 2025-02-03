레이어 파일 크기 제한: 압축 전 기준 250MB 이하

레이어 생성 방법
# 파이썬 폴더 생성
mkdir python
# 폴더 안에 패키지 설치
pip install --target ./python -r requirements.txt
# 압축
zip -r package.zip python

