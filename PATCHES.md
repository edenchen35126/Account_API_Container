# 容器化前建議的最小程式修改

這些修改不改發票辨識流程，只把「環境差異」抽成環境變數。

## 1. processor.py：設定改成 env 優先

把原本：

```python
EXCEL_PATH   = config["excel_path"]
POPPLER_PATH = config["poppler_path"]
FONT_PATH    = config["font_path"]
DPI          = config.get("dpi", 700)
```

改成：

```python
EXCEL_PATH = os.getenv(
    "EXCEL_PATH",
    config["excel_path"],
)

POPPLER_PATH = (
    os.getenv("POPPLER_PATH")
    or config.get("poppler_path")
    or None
)

FONT_PATH = os.getenv(
    "FONT_PATH",
    config["font_path"],
)

DPI = int(
    os.getenv(
        "DPI",
        config.get("dpi", 700),
    )
)
```

## 2. processor.py：PaddleOCR URL / timeout 改成 env 優先

把原本：

```python
OCR_API_URL = config.get(
    "paddleocr_api_url",
    "http://mis-4141:8080/paddleocr/ocr"
)
OCR_API_TIMEOUT = config.get("paddleocr_api_timeout", 300)
```

改成：

```python
OCR_API_URL = os.getenv(
    "PADDLEOCR_API_URL",
    config.get(
        "paddleocr_api_url",
        "http://mis-4141:8080/paddleocr/ocr",
    ),
)

OCR_API_TIMEOUT = int(
    os.getenv(
        "PADDLEOCR_API_TIMEOUT",
        config.get("paddleocr_api_timeout", 300),
    )
)
```

## 3. processor.py：字軌 PostgreSQL 改成 env 優先

把 `get_database_connection()` 改成：

```python
def get_database_connection():
    """建立字軌規則資料庫連線。"""

    database_url = os.getenv(
        "INVOICE_TRACK_DATABASE_URL"
    )

    if database_url:
        return psycopg2.connect(database_url)

    # 保留原本 config fallback，方便舊開發環境
    db_config = config["database"]

    return psycopg2.connect(
        host=db_config["host"],
        port=db_config.get("port", 5432),
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
    )
```

正式環境確認 `INVOICE_TRACK_DATABASE_URL` 可用後，
即可從 config.json 移除 database 帳密。

## 4. vlm.py：移除 Windows Poppler 硬編路徑

把：

```python
POPPLER_PATH = "Release-25.12.0-0/poppler-25.12.0/Library/bin"
```

改成：

```python
POPPLER_PATH = os.getenv("POPPLER_PATH") or None
```

Linux image 已安裝 `poppler-utils`，
`pdf2image` 在 `poppler_path=None` 時直接從 PATH 找 pdftoppm / pdftocairo。

## 5. config.json

可改用本範本內的 `config.container.json` 內容，
正式專案檔名仍要叫做：

```text
config.json
```

## 6. source 檔名

正式 build context 內應確保是：

```text
api.py
processor.py
llm.py
vlm.py
db.py
log_store.py
config.json
requirements-container.txt
Containerfile
compose.yaml
```

不要使用上傳時的 `(10)`、日期等檔名，因為 Python import 與程式碼目前是依上述標準檔名。

## 7. Excel

目前 API startup 會載入：

```text
/app/file/會計憑證POC.xlsx
```

因此 host 專案目錄必須存在：

```text
./file/會計憑證POC.xlsx
```

更新 Excel 後需 restart API，因為 standard_dict 是 startup 時載入。
