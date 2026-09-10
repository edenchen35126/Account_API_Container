FROM docker.io/library/python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Taipei

WORKDIR /app

# pdf2image 需要 poppler-utils
# 中文 debug/標註圖片使用 Noto CJK，避免依賴 Windows 字型
# libgomp1 給 OpenCV / NumPy runtime 使用
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates \
       poppler-utils \
       fontconfig \
       fonts-noto-cjk \
       tzdata \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-container.txt /app/requirements-container.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements-container.txt

# .containerignore 會排除 .env / results / Windows Poppler 等不應進 image 的內容
COPY . /app

RUN mkdir -p /app/results /app/file

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
