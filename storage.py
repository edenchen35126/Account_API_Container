"""
storage.py — MinIO 物件儲存整合

只把每個 job 底下這幾個資料夾的檔案上傳到 MinIO：
    jpg_pages/, json_output/, output/, vlm_crop_debug/
上傳檔案本身（job_dir/input.*）與其他暫存內容不會上傳。

設計理念：
    OCR / TSR / LLM / VLM 流程中大量套件（cv2、PIL、pdf2image/poppler）
    都需要「本地檔案路徑」才能運作，因此無法整段改成直接寫入 MinIO。
    做法改為：
        1. 處理過程仍先寫到本地暫存目錄 results/{job_id}/work（維持不變）。
        2. 該 job 處理完成後，把上述 4 個資料夾上傳到 MinIO。
        3. 可選擇上傳成功後刪除本地這 4 個資料夾
           （CLEANUP_LOCAL_AFTER_UPLOAD=true），上傳檔案本身則保留在本地。

環境變數（於 .env 設定）：
    MINIO_ENDPOINT              例如 "mis-4142:9000"（不含 http(s)://）
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    MINIO_BUCKET                預設 "invoice-results"
    MINIO_SECURE                "true" 使用 https，預設 "false"
    CLEANUP_LOCAL_AFTER_UPLOAD  "true" 上傳成功後刪除本地這 4 個資料夾，預設 "false"
"""

import os
import shutil
import io
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from minio.helpers import check_bucket_name

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _normalize_endpoint(raw_endpoint: str) -> tuple[str, Optional[bool]]:
    """
    去除使用者誤填的 http(s):// 開頭，並依 scheme 推斷 secure。
    回傳 (endpoint, secure_override)；secure_override 為 None 表示沒有 scheme、
    改採 MINIO_SECURE 設定值。
    """
    endpoint = raw_endpoint.strip()
    if endpoint.startswith("https://"):
        return endpoint[len("https://"):], True
    if endpoint.startswith("http://"):
        return endpoint[len("http://"):], False
    return endpoint, None


_raw_endpoint = os.getenv("MINIO_ENDPOINT", "")
MINIO_ENDPOINT, _secure_from_scheme = _normalize_endpoint(_raw_endpoint)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "invoice-results")
MINIO_SECURE     = (
    _secure_from_scheme
    if _secure_from_scheme is not None
    else os.getenv("MINIO_SECURE", "false").lower() == "true"
)

CLEANUP_LOCAL_AFTER_UPLOAD = os.getenv("CLEANUP_LOCAL_AFTER_UPLOAD", "false").lower() == "true"

# 只有這幾個資料夾要上傳到 MinIO
UPLOAD_SUBDIRS = ["jpg_pages", "json_output"#, "output", "vlm_crop_debug"
                  ]

# 上傳的原始檔案（job_dir/input.*）是否也要上傳 MinIO
UPLOAD_INPUT_FILE = True

_client: Optional[Minio] = None


def is_configured() -> bool:
    """是否已在 .env 設定 MinIO 連線資訊。"""
    return bool(MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY)


def get_client() -> Minio:
    """取得（並快取）MinIO client。"""
    global _client
    if _client is None:
        if not is_configured():
            raise RuntimeError(
                "找不到 MinIO 連線設定，請在 .env 設定 "
                "MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY"
            )
        try:
            check_bucket_name(MINIO_BUCKET, strict=True)
        except ValueError as e:
            raise RuntimeError(
                f"MINIO_BUCKET={MINIO_BUCKET!r} 不是合法的 bucket 名稱"
                f"（僅能小寫字母/數字/連字號，3-63字）：{e}"
            ) from e
        _client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _client


def ensure_bucket() -> None:
    """確保目標 bucket 存在，不存在則自動建立。"""
    client = get_client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def put_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """
    直接把記憶體中的 bytes 上傳到 MinIO，不需要先寫入本地磁碟。
    適合用來上傳處理過程中產生的裁切圖片等中間檔案。
    """
    client = get_client()
    ensure_bucket()
    client.put_object(
        MINIO_BUCKET,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def upload_job_folders(job_dir: Path, job_id: str, subdirs: List[str] = None) -> List[str]:
    """
    只把 job_dir 底下指定的資料夾（預設 UPLOAD_SUBDIRS）遞迴上傳到 MinIO。

    object name 格式為 "{job_id}/{子資料夾}/相對路徑"，
    例如 "abc123/json_output/all_pages_result.json"。

    回傳已成功上傳的 object name 清單。
    """
    client = get_client()
    ensure_bucket()

    job_dir = Path(job_dir)
    subdirs = subdirs or UPLOAD_SUBDIRS

    uploaded: List[str] = []
    for sub in subdirs:
        sub_dir = job_dir / sub
        if not sub_dir.is_dir():
            continue
        for file_path in sub_dir.rglob("*"):
            if file_path.is_file():
                relative_name = file_path.relative_to(job_dir).as_posix()
                object_name = f"{job_id}/{relative_name}"
                client.fput_object(MINIO_BUCKET, object_name, str(file_path))
                uploaded.append(object_name)

    return uploaded


def object_exists(object_name: str) -> bool:
    client = get_client()
    try:
        client.stat_object(MINIO_BUCKET, object_name)
        return True
    except S3Error:
        return False


def get_object_bytes(object_name: str) -> bytes:
    """讀取單一 object 的內容（用於結果查詢等場景）。"""
    client = get_client()
    response = client.get_object(MINIO_BUCKET, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_presigned_url(object_name: str, expires_seconds: int = 3600) -> str:
    """產生可直接下載的臨時網址（預設 1 小時有效）。"""
    client = get_client()
    return client.presigned_get_object(
        MINIO_BUCKET, object_name, expires=timedelta(seconds=expires_seconds)
    )


def cleanup_job_folders(job_dir: Path, subdirs: List[str] = None) -> None:
    """刪除本地這幾個已上傳的資料夾（上傳檔案本身 job_dir/input.* 不會被刪除）。"""
    job_dir = Path(job_dir)
    subdirs = subdirs or UPLOAD_SUBDIRS
    for sub in subdirs:
        shutil.rmtree(job_dir / sub, ignore_errors=True)
