from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# check_same_thread 옵션은 sqlite에서만 필요
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# SessionLocal의 인스턴스를 생성하면 데이터베이스의 세션이 생성됨
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base를 상속하여 모델을 만들게 된다.
Base = declarative_base()
