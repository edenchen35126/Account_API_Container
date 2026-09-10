"""
log_store.py — PostgreSQL Log 記錄（透過 SQLAlchemy 介接 db.py）
提供與 api.py 互動的 create_api_log / update_api_log / read_api_log 。
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError

from db import Base, SessionLocal, engine


# =========================
# ORM Model
# =========================
class ApiUsageLog(Base):
    __tablename__ = "api_usage_logs"

    id              = Column(Integer, primary_key=True, index=True)
    job_id          = Column(String(64),  unique=True, index=True, nullable=False)
    status          = Column(String(30),  nullable=False, default="PROCESSING")
    endpoint        = Column(String(255), nullable=True)
    method          = Column(String(20),  nullable=True)
    client_ip       = Column(String(100), nullable=True)
    filename        = Column(Text,        nullable=True)
    file_size       = Column(BigInteger,  nullable=True)
    content_type    = Column(String(255), nullable=True)
    api_key_alias   = Column(String(255), nullable=True)
    http_status     = Column(Integer,     nullable=True)
    elapsed_ms      = Column(Integer,     nullable=True)
    saved_path      = Column(Text,        nullable=True)
    result_summary  = Column(JSONB,       nullable=True)
    error_message   = Column(Text,        nullable=True)
    error_traceback = Column(Text,        nullable=True)
    created_at      = Column(DateTime,    nullable=False, default=datetime.now)
    updated_at      = Column(DateTime,    nullable=False, default=datetime.now)
    finished_at     = Column(DateTime,    nullable=True)


# =========================
# 建立資料表
# =========================
def init_db():
    """Create all tables if not exist."""
    Base.metadata.create_all(bind=engine)


# =========================
# Helper
# =========================
def _now() -> datetime:
    return datetime.now()


def _model_to_dict(log: ApiUsageLog) -> Dict[str, Any]:
    return {
        "id":              log.id,
        "job_id":          log.job_id,
        "status":          log.status,
        "endpoint":        log.endpoint,
        "method":          log.method,
        "client_ip":       log.client_ip,
        "filename":        log.filename,
        "file_size":       log.file_size,
        "content_type":    log.content_type,
        "api_key_alias":   log.api_key_alias,
        "http_status":     log.http_status,
        "elapsed_ms":      log.elapsed_ms,
        "saved_path":      log.saved_path,
        "result_summary":  log.result_summary,
        "error_message":   log.error_message,
        "error_traceback": log.error_traceback,
        "created_at":      log.created_at.isoformat(timespec="seconds") if log.created_at else None,
        "updated_at":      log.updated_at.isoformat(timespec="seconds") if log.updated_at else None,
        "finished_at":     log.finished_at.isoformat(timespec="seconds") if log.finished_at else None,
    }


# =========================
# CRUD
# =========================
def create_api_log(data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new log row with status=PROCESSING."""
    db = SessionLocal()
    try:
        log = ApiUsageLog(
            job_id          = data["job_id"],
            status          = data.get("status", "PROCESSING"),
            endpoint        = data.get("endpoint"),
            method          = data.get("method"),
            client_ip       = data.get("client_ip"),
            filename        = data.get("filename"),
            file_size       = data.get("file_size"),
            content_type    = data.get("content_type"),
            api_key_alias   = data.get("api_key_alias"),
            http_status     = data.get("http_status"),
            elapsed_ms      = data.get("elapsed_ms"),
            saved_path      = data.get("saved_path"),
            result_summary  = data.get("result_summary"),
            error_message   = data.get("error_message"),
            error_traceback = data.get("error_traceback"),
            created_at      = data.get("created_at") or _now(),
            updated_at      = _now(),
            finished_at     = data.get("finished_at"),
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return _model_to_dict(log)
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def update_api_log(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing log row by job_id. Creates a new row if not found."""
    db = SessionLocal()
    try:
        log = db.query(ApiUsageLog).filter(ApiUsageLog.job_id == job_id).first()
        if not log:
            log = ApiUsageLog(
                job_id     = job_id,
                status     = updates.get("status", "PROCESSING"),
                created_at = _now(),
            )
            db.add(log)

        allowed_fields = {
            "status", "endpoint", "method", "client_ip", "filename",
            "file_size", "content_type", "api_key_alias", "http_status",
            "elapsed_ms", "saved_path", "result_summary",
            "error_message", "error_traceback", "finished_at",
        }
        for key, val in updates.items():
            if key in allowed_fields:
                setattr(log, key, val)

        log.updated_at = _now()

        # SUCCESS / FAILED 時自動填 finished_at
        if log.status in ("SUCCESS", "FAILED") and not log.finished_at:
            log.finished_at = _now()

        db.commit()
        db.refresh(log)
        return _model_to_dict(log)
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def read_api_log(job_id: str) -> Optional[Dict[str, Any]]:
    """Read a log row by job_id, return as dict or None."""
    db = SessionLocal()
    try:
        log = db.query(ApiUsageLog).filter(ApiUsageLog.job_id == job_id).first()
        if not log:
            return None
        return _model_to_dict(log)
    finally:
        db.close()