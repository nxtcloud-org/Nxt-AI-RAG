import psycopg2

# 연결 정보
rds_host = ""
db_username = ''
db_password = ''

# 연결
try:
    conn = psycopg2.connect(host=rds_host, database='postgres', user=db_username, password=db_password)
    conn.autocommit = True
except Exception as e:
    print(f"Error connecting to RDS: {e}")
    exit()
    
def create_database_and_user(user_index):
    user_name = f"user_{user_index:03d}"
    db_name = f"db_{user_index:03d}"
    user_password = f"pw_{user_index:03d}"
    with conn.cursor() as cursor:

        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created")
        
        cursor.execute(f"CREATE USER {user_name} WITH PASSWORD %s", (user_password,))
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user_name}")
        cursor.execute(f"GRANT ALL ON SCHEMA public TO {user_name}")
        cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {user_name}")
    
    # 새로 생성된 데이터베이스에 연결하여 vector 설치
    db_conn = psycopg2.connect(
        host=rds_host,
        database=db_name,
        user=db_username,
        password=db_password
    )
    db_conn.autocommit = True
    
    with db_conn.cursor() as cursor:
        cursor.execute(f"GRANT ALL ON SCHEMA public TO {user_name}")
        cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {user_name}")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute('''
           CREATE TABLE IF NOT EXISTS documents (
               id SERIAL PRIMARY KEY,
               content TEXT,
               embedding vector(1536),
               metadata JSONB,
               created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
           )
        ''')
        cursor.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {user_name}")
        
        print(f"Vector extension installed in {db_name}")
    
    db_conn.close()

# 사용자 및 데이터베이스를 원하는 횟수만큼 생성
number_of_users = 2  # 생성할 사용자 및 데이터베이스 수
for i in range(1, number_of_users + 1):
    print(f'{i} 번째 유저의 데이터베이스 정보 생성중...\n\n')
    create_database_and_user(i)
    
# cursor = conn.cursor()
# cursor.execute("SELECT datname FROM pg_database;")
# result = cursor.fetchall()
# print("All databases:", result)

# 연결 종료
conn.close()
