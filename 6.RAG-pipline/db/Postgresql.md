#Cloud9에 설치
sudo yum install -y postgresql15

# 생성 DB 전체 확인
SELECT datname FROM pg_database;

# DB 삭제
DROP database db_001;

# DB 변경
\c db_001;

# 테이블 조회
SELECT * FROM documents;
SELECT COUNT(*) FROM documents;

# 유저 삭제
REVOKE ALL ON SCHEMA public FROM user_001;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON TABLES FROM user_001;
DROP USER IF EXISTS user_001;

# DB 연결
psql -h 호스트경로 -U 유저이름 -d 디비이름

# 벡터 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;

# 테이블 생성
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);