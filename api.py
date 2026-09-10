"""
api.py — FastAPI 路由管理
...
"""

import os
import json
import time
import uuid
import traceback                          # ← 新增
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse, StreamingResponse

import processor
import log_store                                                         # ← 新增
import storage                                                           # ← 新增：MinIO 物件儲存

from dotenv import load_dotenv

import asyncio

load_dotenv()

API_KEY = os.getenv("INVOICE_API_KEY")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str = Security(api_key_header)
):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    return api_key

app = FastAPI(title="Invoice OCR API", version="1.0.0")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    log_store.init_db()                                                  # ← 新增
    # 已停用 Excel 標準答案載入。
    # standard_dict = processor.load_excel_standard(processor.EXCEL_PATH)


# ← 新增 /health
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "invoice-recognition-api",
        "version": "1.0.0",
        "mode": "legacy_app_compare_only",
    }


# ← 新增 job 查詢
@app.get("/api/jobs/{job_id}")
def get_job_log(job_id: str):
    log = log_store.read_api_log(job_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"找不到 job_id：{job_id}")
    return {"success": True, "job": log}


# ← 新增 result 查詢
@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    result_path = RESULTS_DIR / job_id / "work" / "json_output" / "all_pages_result.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        return {"success": True, "job_id": job_id, "result": result}

    # 本地檔案已不存在（例如已上傳並清除），改從 MinIO 讀取
    if storage.is_configured():
        object_name = f"{job_id}/json_output/all_pages_result.json"
        if storage.object_exists(object_name):
            result = json.loads(storage.get_object_bytes(object_name).decode("utf-8"))
            return {"success": True, "job_id": job_id, "result": result, "source": "minio"}

    raise HTTPException(status_code=404, detail=f"找不到辨識結果：{job_id}")


# @app.post("/api/process")
# async def process_invoice(
#     request: Request,                                                    # ← 加 Request
#     file: UploadFile = File(...)
# ):
#     job_id    = uuid.uuid4().hex

@app.post("/api/process")
async def process_invoice(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    job_id = uuid.uuid4().hex
    start_ms = time.time() * 1000
    client_ip = request.client.host if request.client else None

    # =====================================================
    # 建立 API Log
    # =====================================================
    log_store.create_api_log({
        "job_id":        job_id,
        "endpoint":      "/api/process",
        "method":        "POST",
        "client_ip":     client_ip,
        "filename":      file.filename,
        "content_type":  file.content_type,
        "api_key_alias": request.headers.get("x-api-key-alias"),
        "status":        "PROCESSING",
        "http_status":   None,
    })

    try:
        # =====================================================
        # 1. 建立工作目錄
        # =====================================================
        job_dir         = RESULTS_DIR / job_id / "work"
        json_output_dir = job_dir / "json_output"
        jpg_dir         = job_dir / "jpg_pages"

        json_output_dir.mkdir(parents=True, exist_ok=True)
        jpg_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # 2. 儲存上傳檔案
        # =====================================================
        file_suffix = Path(
            file.filename or "upload.pdf"
        ).suffix.lower()

        upload_path = job_dir / f"input{file_suffix}"

        content = await file.read()

        with open(upload_path, "wb") as f_out:
            f_out.write(content)

        file_size = len(content)

        log_store.update_api_log(job_id, {
            "file_size":  file_size,
            "saved_path": str(upload_path),
        })

    except Exception as e:
        # =====================================================
        # Streaming 尚未開始前發生錯誤
        # 這時還可以正常回 HTTP 500
        # =====================================================
        elapsed_ms = int(time.time() * 1000 - start_ms)

        log_store.update_api_log(job_id, {
            "status":          "FAILED",
            "http_status":     500,
            "elapsed_ms":      elapsed_ms,
            "error_message":   str(e),
            "error_traceback": traceback.format_exc(),
        })

        raise HTTPException(
            status_code=500,
            detail=f"建立辨識工作失敗：{str(e)}"
        )

    # =========================================================
    # 3. 真正的同步耗時工作
    #
    # 注意：
    # 這個函式會被 asyncio.to_thread() 執行
    # 所以不會卡住 FastAPI Event Loop
    # =========================================================
    def run_job_sync():

        try:
            # =================================================
            # 原始上傳檔案上傳 MinIO
            # =================================================
            if (
                storage.is_configured()
                and storage.UPLOAD_INPUT_FILE
            ):
                try:
                    input_object_name = (
                        f"{job_id}/input{file_suffix}"
                    )

                    storage.put_bytes(
                        input_object_name,
                        content,
                        content_type=(
                            file.content_type
                            or "application/octet-stream"
                        ),
                    )

                    print(
                        f"[MinIO] 已上傳原始檔案："
                        f"{input_object_name}"
                    )

                except Exception as upload_err:
                    # 原始檔 MinIO 上傳失敗
                    # 不影響主辨識流程
                    print(
                        f"[MinIO] job_id={job_id} "
                        f"原始檔案上傳失敗：{upload_err}"
                    )

            # =================================================
            # OCR → TSR → LLM → VLM → Retry
            # =================================================
            all_pages_result = processor.process_document(
                file_path=str(upload_path),
                work_dir=str(job_dir),
            )

            result_path = str(
                json_output_dir
                / "all_pages_result.json"
            )

            # =================================================
            # 全頁是否通過
            # =================================================
            all_pages_passed = all(
                p.get(
                    "compare_result",
                    {}
                ).get(
                    "全部比對通過",
                    False
                )
                for p in all_pages_result
                if p.get("status") == "processed"
            )

            # =================================================
            # API 輸出整理
            # =================================================
            pages_output = []

            for p in all_pages_result:

                if p.get("status") == "processed":

                    pages_output.append({
                        "page": p["page"],
                        "compare_result":
                            processor.build_api_compare_result(
                                p["compare_result"],
                                p.get(
                                    "field_confidence",
                                    {}
                                ),
                                p.get(
                                    "api_extra_fields",
                                    {}
                                ),
                            ),
                    })

                else:

                    pages_output.append({
                        "page":   p["page"],
                        "status": p.get("status"),
                        "reason": p.get("reason"),
                    })

            # =================================================
            # MinIO 上傳
            # =================================================
            minio_uploaded = False
            minio_object_count = 0

            if storage.is_configured():

                try:
                    uploaded_objects = (
                        storage.upload_job_folders(
                            job_dir,
                            job_id
                        )
                    )

                    minio_uploaded = True
                    minio_object_count = len(
                        uploaded_objects
                    )

                    if storage.CLEANUP_LOCAL_AFTER_UPLOAD:
                        storage.cleanup_job_folders(
                            job_dir
                        )

                except Exception as upload_err:

                    minio_uploaded = False

                    print(
                        f"[MinIO] job_id={job_id} "
                        f"上傳失敗：{upload_err}"
                    )

                    log_store.update_api_log(
                        job_id,
                        {
                            "error_message":
                                f"MinIO 上傳失敗："
                                f"{upload_err}",
                        }
                    )

            # =================================================
            # 完成
            # =================================================
            elapsed_ms = int(
                time.time() * 1000
                - start_ms
            )

            final_result = {
                "success": True,
                "job_id": job_id,
                "elapsed_ms": elapsed_ms,
                "mode": "legacy_app_compare_only",

                "result": {
                    "job_id": job_id,
                    "page_count": len(
                        all_pages_result
                    ),
                    "all_pages_passed":
                        all_pages_passed,
                    "legacy_result_path":
                        os.path.abspath(
                            result_path
                        ),
                    "pages": pages_output,
                },

                "storage": {
                    "minio_uploaded":
                        minio_uploaded,
                    "minio_object_count":
                        minio_object_count,
                },
            }

            # =================================================
            # Success Log
            #
            # 放在 Worker 裡面：
            # 就算 Client 中途斷線，
            # 背景工作完成後還是會更新 DB Log
            # =================================================
            log_store.update_api_log(
                job_id,
                {
                    "status": "SUCCESS",
                    "http_status": 200,
                    "elapsed_ms": elapsed_ms,

                    "result_summary": {
                        "page_count":
                            len(all_pages_result),

                        "all_pages_passed":
                            all_pages_passed,

                        "legacy_result_path":
                            os.path.abspath(
                                result_path
                            ),

                        "minio_uploaded":
                            minio_uploaded,

                        "minio_object_count":
                            minio_object_count,
                    },
                }
            )

            return final_result

        except Exception as e:

            # =================================================
            # Worker 執行失敗
            # =================================================
            elapsed_ms = int(
                time.time() * 1000
                - start_ms
            )

            log_store.update_api_log(
                job_id,
                {
                    "status": "FAILED",
                    "http_status": 500,
                    "elapsed_ms": elapsed_ms,
                    "error_message": str(e),
                    "error_traceback":
                        traceback.format_exc(),
                }
            )

            raise

    # =========================================================
    # 4. Streaming Generator
    # =========================================================
    async def generate_response():

        # 把整個長工作業丟到 Thread
        task = asyncio.create_task(
            asyncio.to_thread(
                run_job_sync
            )
        )

        try:
            # =================================================
            # 第一個 heartbeat
            #
            # JSON 前面允許 whitespace
            # =================================================
            yield b"\n"

            # =================================================
            # 每 20 秒確認工作是否完成
            # =================================================
            while not task.done():

                try:
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=20
                    )

                except asyncio.TimeoutError:

                    # =========================================
                    # heartbeat
                    # =========================================
                    yield b"\n"

            # =================================================
            # 工作完成
            # =================================================
            final_result = await task

            # =================================================
            # 最終 JSON
            # =================================================
            yield json.dumps(
                final_result,
                ensure_ascii=False,
                default=str
            ).encode("utf-8")

        except Exception as e:

            # =================================================
            # 注意：
            # Streaming 已開始
            # HTTP Status 已經是 200
            #
            # 所以這裡只能用 JSON 告訴 Client 失敗
            # =================================================
            error_result = {
                "success": False,
                "job_id": job_id,
                "error": f"辨識失敗：{str(e)}",
            }

            yield json.dumps(
                error_result,
                ensure_ascii=False,
                default=str
            ).encode("utf-8")

    # =========================================================
    # 5. Streaming Response
    # =========================================================
    return StreamingResponse(
        generate_response(),
        media_type="application/json",
        headers={
            "Cache-Control": "no-cache",

            # 告訴 Nginx 不要把 Response Buffer 起來
            "X-Accel-Buffering": "no",

            # Client 可以直接知道目前是哪一個 Job
            "X-Job-Id": job_id,
        }
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
