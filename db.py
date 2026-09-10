import os
from pathlib import Path
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# 載入專案根目錄的 .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "找不到 DATABASE_URL，請在 .env 設定 PostgreSQL 連線字串"
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def init_db():
    """
    啟動 API 時自動建立資料表。
    正式環境之後可以改 Alembic migration。
    """
    import log_store  # 確保 ApiUsageLog 被註冊到 Base.metadata

    Base.metadata.create_all(bind=engine)