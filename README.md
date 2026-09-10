# Invoice Recognition API - Podman 部署

## 建議架構

此 compose 只負責 Invoice API。

外部依賴：
- PostgreSQL：mis-4142:5432
- PaddleOCR API：mis-4141:8080
- LLM/VLM API：mis-4142:8190
- Tavily：若保留目前 import 初始化，需可連外

## 專案目錄

```text
invoice-api/
├─ Containerfile
├─ compose.yaml
├─ .containerignore
├─ .env.production
├─ requirements-container.txt
├─ api.py
├─ processor.py
├─ llm.py
├─ vlm.py
├─ db.py
├─ log_store.py
├─ config.json
└─ file/
   └─ 會計憑證POC.xlsx
```

## 第一次部署

```bash
cp .env.production.example .env.production
# 編輯 .env.production，填入正式環境帳密與 API Key

podman compose version
podman compose build
podman compose up -d
podman compose ps
podman logs -f invoice-recognition-api
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Swagger

瀏覽器開：

```text
http://<正式主機>:8000/docs
```

## 測試容器內 DNS

```bash
podman exec invoice-recognition-api \
  python -c "import socket; print(socket.gethostbyname('mis-4141')); print(socket.gethostbyname('mis-4142'))"
```

如果解析不到 mis-4141 / mis-4142，
需要處理正式機 DNS，或在 compose 中設定 extra_hosts。

## 重建

```bash
podman compose down
podman compose build
podman compose up -d
```

`invoice-results` 是 named volume，正常 down/up 不會刪除。
不要使用 `podman compose down -v`，除非確定要刪除 results volume。

## 注意

目前 `/health` 只代表 FastAPI 本身活著，
不代表 PostgreSQL / PaddleOCR / LLM/VLM 都正常。

正式上線後建議另外新增 `/ready`，
檢查外部依賴是否可連線。
