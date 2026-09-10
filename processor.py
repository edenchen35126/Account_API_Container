# =========================
# 套件引入
# =========================
from pdf2image import convert_from_path  # PDF 轉圖片
from opencc import OpenCC                # 簡體轉繁體
import cv2                               # 圖像處理
import numpy as np                       # 數值運算
import os                                # 檔案系統操作
import io                                # 記憶體內 bytes 緩衝（上傳 MinIO 用）
import json                              # JSON 讀寫
import re                                # 正則表達式
import mimetypes                         # HTTP 上傳檔案 MIME type
import requests                          # 呼叫 PaddleOCR 容器 API
from PIL import Image, ImageDraw, ImageFont  , ImageEnhance, ImageFilter # 圖片繪製與字型
import pandas as pd                      # Excel 讀取
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

import storage                           # ← 新增：MinIO 物件儲存

from PIL import Image, ImageOps, ImageEnhance

# ✅ 引入 VLM 判斷函式
from vlm import (
    detect_multi_invoice, extract_fields_from_image_region, crop_image_region,
    detect_total_ntd_text, extract_invoice_number_from_image,
    extract_tax_type_from_image,
    verify_company_existence_by_model,   # ✅ 新增
)
# ✅ 引入 LLM 擷取函式
from llm import (
    extract_invoice_fields_by_llm,
    reextract_specific_fields,
    locate_field_region_by_llm,
    determine_tax_type_by_llm,
    compare_chinese_amount_meaning_by_llm,
    verify_seller_name_matches_tax_id,
    repair_leading_invalid_amount_slot,  # ✅ 新增
)



# =========================
# 讀取設定檔（API 模式只使用辨識設定，不使用固定輸入檔）
# =========================
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

EXCEL_PATH   = config["excel_path"]
POPPLER_PATH = config["poppler_path"]
FONT_PATH    = config["font_path"]
DPI          = config.get("dpi", 700)

# =========================
# 簡體 → 繁體轉換器
# =========================
cc = OpenCC('s2t')

# =========================
# 明細欄位預設與外部服務設定
# =========================
_DETAIL_PRESETS = config.get("detail_field_presets", {
    "FULL":   ["品名", "數量", "單價", "金額"],
    "AMOUNT": ["品名", "金額"]
})

OCR_API_URL = config.get(
    "paddleocr_api_url",
    "http://mis-4141:8080/paddleocr/ocr"
)
OCR_API_TIMEOUT = config.get("paddleocr_api_timeout", 300)


def get_period_start_month(month: int) -> int:
    """將月份換算為統一發票雙月期別的起始月份。"""
    return ((month - 1) // 2) * 2 + 1


def get_database_connection():
    """建立字軌規則資料庫連線（連線設定來自 .env 的 INVOICE_RULE_DB_*）。"""
    host     = os.getenv("INVOICE_RULE_DB_HOST")
    port     = os.getenv("INVOICE_RULE_DB_PORT", "5432")
    database = os.getenv("INVOICE_RULE_DB_NAME")
    user     = os.getenv("INVOICE_RULE_DB_USER")
    password = os.getenv("INVOICE_RULE_DB_PASSWORD")

    if not host or not database or not user:
        raise RuntimeError(
            "找不到字軌規則資料庫連線設定，請在 .env 設定 "
            "INVOICE_RULE_DB_HOST / INVOICE_RULE_DB_NAME / "
            "INVOICE_RULE_DB_USER / INVOICE_RULE_DB_PASSWORD"
        )

    return psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )


def load_invoice_prefix_rules(invoice_year: int) -> dict:
    """
    從 PostgreSQL 載入指定年度的字軌規則。

    回傳格式維持 {prefix: rule_config}，讓後續辨識邏輯不需改寫。
    """
    sql = """
        SELECT
            TRIM(p.prefix) AS prefix,
            r.rule_config
        FROM invoice_prefix_period p
        INNER JOIN invoice_rule_profile r
            ON r.rule_code = p.rule_code
        WHERE p.invoice_year = %s
          AND r.is_active = TRUE
    """

    with get_database_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, (invoice_year,))
            rows = cursor.fetchall()

    prefix_rules = {}
    for row in rows:
        prefix = row["prefix"].strip().upper()
        prefix_rules[prefix] = row["rule_config"]

    print(
        f"[字軌資料庫] 已載入 {invoice_year} 年，"
        f"共 {len(prefix_rules)} 個字軌"
    )
    return prefix_rules


# 僅作函式預設值；API 實際處理時會依每頁發票年月動態載入。
INVOICE_PREFIX_RULES = {}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def convert_file_to_pages(file_path: str) -> list[Image.Image]:
    """將 API 上傳的 PDF 或圖片轉為逐頁 PIL Image。"""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return convert_from_path(file_path, dpi=DPI, poppler_path=POPPLER_PATH)
    if suffix in IMAGE_EXTENSIONS:
        with Image.open(file_path) as image:
            return [image.convert("RGB")]
    raise ValueError(f"不支援的檔案格式：{suffix}")


def _norm_for_match(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "")
    return s

def get_ocr_confidence_for_value(ocr_items: list, target_value: str) -> float | None:
    """
    用最小侵入方式估 OCR 信心值：
    - 先找完全匹配
    - 找不到再找包含關係
    - 回傳匹配到的最高 score
    """
    tgt = _norm_for_match(target_value)
    if not tgt:
        return None

    best = None
    for item in ocr_items:
        txt = _norm_for_match(item.get("text", ""))
        score = item.get("score")
        if score is None:
            continue

        if txt == tgt or (tgt in txt) or (txt in tgt):
            best = score if best is None else max(best, score)

    return round(best, 4) if best is not None else None


def estimate_field_confidence(ocr_items: list, field_name: str, field_value, source: str) -> float | None:
    """
    統一估算欄位信心值（不影響原本判斷流程）：
    1) 只針對 OCR 欄位估算
    2) LLM/VLM 欄位不在此估算，改由模型 prompt 直接回傳
    """
    if field_value is None:
        return None

    source = (source or "").upper()

    if source != "OCR":
        return None

    # 明細項目是 list[dict]，取可匹配到 OCR 的分數平均
    if field_name == "明細項目" and isinstance(field_value, list):
        matched_scores = []
        for item in field_value:
            if not isinstance(item, dict):
                continue
            for key in ["品名", "數量", "單價", "金額"]:
                val = item.get(key)
                if val:
                    sc = get_ocr_confidence_for_value(ocr_items, str(val))
                    if sc is not None:
                        matched_scores.append(sc)

        if matched_scores:
            return round(sum(matched_scores) / len(matched_scores), 4)
        return None

    # 一般欄位：先試 OCR 對齊
    conf = get_ocr_confidence_for_value(ocr_items, str(field_value))
    if conf is not None:
        return conf

    return None


def format_confidence(conf_value) -> str:
    if conf_value is None:
        return "N/A"
    try:
        return f"{float(conf_value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

def compare_detail_items(ocr_items: list, std_items: list, active_detail_fields: list = None) -> dict:
    if active_detail_fields is None:
        active_detail_fields = ["品名", "數量", "單價", "金額"]

    if not std_items:
        return {
            "是否一致":  None,
            "檢核欄位":  active_detail_fields,
            "比對細節":  "標準答案未填寫明細項目，略過比對",
            "失敗項目摘要": []
        }

    if not ocr_items:
        return {
            "是否一致":  False,
            "檢核欄位":  active_detail_fields,
            "比對細節":  "資料缺失",
            "失敗項目摘要": []
        }

    if len(ocr_items) != len(std_items):
        return {
            "是否一致":  False,
            "檢核欄位":  active_detail_fields,
            "比對細節":  f"筆數不一致（標準:{len(std_items)}筆, OCR:{len(ocr_items)}筆）",
            "失敗項目摘要": []
        }

    detail_results = []
    all_match      = True
    failed_items   = []

    # ✅ 每個欄位的整體一致性（所有品項都一致才算True）
    field_all_match = {key: True for key in active_detail_fields}

    for idx, (ocr_item, std_item) in enumerate(zip(ocr_items, std_items)):
        item_result = {}
        item_match  = True

        for key in active_detail_fields:
            ocr_val = str(ocr_item.get(key) or "").replace(",", "").strip()
            std_val = str(std_item.get(key) or "").replace(",", "").strip()

            if key in ["數量", "單價", "金額"]:
                ocr_num_str = re.search(r'[\d.]+', ocr_val)
                std_num_str = re.search(r'[\d.]+', std_val)
                try:
                    ocr_num = float(ocr_num_str.group()) if ocr_num_str else None
                    std_num = float(std_num_str.group()) if std_num_str else None
                    if ocr_num is not None and std_num is not None:
                        match   = (ocr_num == std_num)
                        ocr_val = str(int(ocr_num)) if ocr_num == int(ocr_num) else str(ocr_num)
                        std_val = str(int(std_num)) if std_num == int(std_num) else str(std_num)
                    else:
                        match = (ocr_val == std_val)
                except ValueError:
                    match = (ocr_val == std_val)
            else:
                match = (ocr_val == std_val)

            if not match:
                item_match = False
                field_all_match[key] = False  # ✅ 該欄位有一筆失敗就標記

            item_result[key] = {
                "標準答案": std_val,
                "OCR結果":  ocr_val,
                "是否一致": match
            }

        item_result["此筆一致"] = item_match

        if not item_match:
            all_match = False
            failed_fields_in_item = [
                k for k in active_detail_fields
                if not item_result.get(k, {}).get("是否一致", True)
            ]
            failed_items.append({
                "第幾筆":   idx + 1,
                "品名":     std_item.get("品名", ""),
                "失敗欄位": failed_fields_in_item,
                "詳細":     {k: item_result[k] for k in failed_fields_in_item}
            })

        detail_results.append(item_result)

    # ✅ 每個欄位的整體摘要（跟未稅金額格式一樣）
    field_summary = {}
    for key in active_detail_fields:
        std_vals = [str(item.get(key, "")).strip() for item in std_items]
        ocr_vals = [str(item.get(key, "")).strip() for item in ocr_items]
        field_summary[key] = {
            "標準答案": std_vals if len(std_vals) > 1 else std_vals[0],
            "OCR結果":  ocr_vals if len(ocr_vals) > 1 else ocr_vals[0],
            "是否一致": field_all_match[key]
        }

    return {
        "是否一致":    all_match,
        "檢核欄位":    active_detail_fields,
        "欄位摘要":    field_summary,      # ✅ 每個欄位整體一致性
        "比對細節":    detail_results,     # 每筆明細詳細結果
        "失敗項目摘要": failed_items        # 哪幾筆哪些欄位失敗
    }


def compare_detail_items_has_value(ocr_items: list, active_detail_fields: list = None) -> dict:
    """
    明細項目新邏輯：只要每筆品項中各欄位有值就通過，不需與 Excel 比對。

    Args:
        ocr_items:            OCR 彙整後的明細項目 list
        active_detail_fields: 要檢核的欄位清單（品名、數量、單價、金額）
    Returns:
        dict: 檢核結果
    """
    if active_detail_fields is None:
        active_detail_fields = ["品名", "數量", "單價", "金額"]

    if not ocr_items:
        return {
            "是否一致":  False,
            "檢核欄位":  active_detail_fields,
            "比對細節":  "OCR未擷取到明細項目",
            "失敗項目摘要": []
        }

    all_pass    = True
    failed_items = []
    detail_results = []

    for idx, ocr_item in enumerate(ocr_items):
        item_result = {}
        item_pass   = True

        for key in active_detail_fields:
            val = str(ocr_item.get(key) or "").replace(",", "").strip()
            has_value = bool(val)
            if not has_value:
                item_pass = False
            item_result[key] = {
                "OCR結果":  val,
                "是否一致": has_value
            }

        item_result["此筆一致"] = item_pass

        if not item_pass:
            all_pass = False
            empty_fields = [k for k in active_detail_fields if not item_result[k]["是否一致"]]
            failed_items.append({
                "第幾筆":   idx + 1,
                "品名":     str(ocr_item.get("品名", "")),
                "失敗欄位": empty_fields
            })

        detail_results.append(item_result)

    # 欄位摘要（彙整每個欄位的 OCR 值清單）
    field_summary = {}
    for key in active_detail_fields:
        ocr_vals = [str(item.get(key, "")).strip() for item in ocr_items]
        field_all_pass = all(bool(v) for v in ocr_vals)
        field_summary[key] = {
            "OCR結果":  ocr_vals if len(ocr_vals) > 1 else (ocr_vals[0] if ocr_vals else ""),
            "是否一致": field_all_pass
        }

    return {
        "是否一致":    all_pass,
        "檢核欄位":    active_detail_fields,
        "欄位摘要":    field_summary,
        "比對細節":    detail_results,
        "失敗項目摘要": failed_items
    }


def build_ocr_text_with_position(ocr_items: list) -> str:
    """
    將 OCR 結果轉成帶位置資訊的文字，傳給 LLM 輔助判斷
    格式：[x1,y1,x2,y2] 文字內容（bbox 外接矩形）
    同一行（Y 座標相近）的文字會排在同一行
    """
    if not ocr_items:
        return ""

    # 依 Y 座標排序後分行
    sorted_items = sort_bbox_texts(ocr_items, y_tolerance=15)

    lines = []
    current_line = []
    current_y    = None

    for item in sorted_items:
        y1 = item["bbox"][1]
        if current_y is None or abs(y1 - current_y) <= 15:
            current_line.append(item)
            current_y = y1
        else:
            lines.append(current_line)
            current_line = [item]
            current_y    = y1

    if current_line:
        lines.append(current_line)

    # 每行組成「[x,y] 文字」格式
    result_lines = []
    for line in lines:
        line_parts = []
        for item in line:
            x1, y1, x2, y2 = item["bbox"]
            line_parts.append(f"[{x1},{y1},{x2},{y2}]{item['text']}")
        result_lines.append("  ".join(line_parts))

    return "\n".join(result_lines)

def get_validation_rules_by_prefix(invoice_no: str, page_text_clean: str = "", page_image=None, prefix_rules: dict = None) -> dict:
    """根據發票號碼前兩碼英文，決定要檢核的項目。"""
    if prefix_rules is None:
        prefix_rules = INVOICE_PREFIX_RULES

    if not invoice_no or len(invoice_no) < 2:
        print(f"⚠️  無法取得發票前綴，轉人工審核")
        return {"prefix": None, "rules": None, "detail_fields": None, "聯式": None, "unknown": True}

    prefix = invoice_no[:2].upper()

    if prefix not in prefix_rules:
        print(f"⚠️  發票前綴 [{prefix}] 找不到對應檢核規則，轉人工審核")
        return {"prefix": prefix, "rules": None, "detail_fields": None, "聯式": None, "unknown": True}

    prefix_config = prefix_rules[prefix]

    if "condition" in prefix_config:
        keywords = prefix_config["condition"]["keywords"]
        matched = any(keyword in page_text_clean for keyword in keywords)
        condition_source = "OCR"

        if not matched and page_image is not None:
            # 第一次：原圖 VLM 判斷
            vlm_condition = detect_total_ntd_text(page_image)
            vlm_status = vlm_condition.get("status")

            condition_source = (
                f"VLM1({vlm_condition.get('reason', '')})"
            )

            # 先以第一次結果作為目前最終結果
            final_vlm_status = vlm_status

            # ============================================================
            # 第一次只要是 absent 或 uncertain，都放大再判斷一次
            # ============================================================
            if vlm_status in {"absent", "uncertain"}:
                print(
                    f"[總計新臺幣判斷] 第一次結果={vlm_status}，"
                    "放大圖片後重新判斷..."
                )

                w, h = page_image.size

                retry_image = page_image.resize(
                    (w * 2, h * 2),
                    Image.Resampling.LANCZOS
                )

                retry_condition = detect_total_ntd_text(
                    retry_image
                )

                retry_status = retry_condition.get("status")

                condition_source = (
                    f"{condition_source} → "
                    f"VLM2({retry_condition.get('reason', '')})"
                )

                # 第二次結果作為最終 VLM 結果
                final_vlm_status = retry_status

            # ============================================================
            # 所有 VLM 判斷完成後，
            # 如果最終仍為 absent，才做 OCR 最後保底
            # ============================================================
            if final_vlm_status == "absent":
                ocr_text_for_check = str(page_text_clean or "")

                if (
                    "元" in ocr_text_for_check
                    or "整" in ocr_text_for_check
                ):
                    print(
                        "[總計新臺幣判斷][OCR最終保底] "
                        "兩次 VLM 最終結果=absent，"
                        "但 OCR 文字中發現「元」或「整」，"
                        "改為 uncertain"
                    )

                    final_vlm_status = "uncertain"

            # ============================================================
            # 最後才轉成 matched
            # ============================================================
            if final_vlm_status == "present":
                matched = True

            elif final_vlm_status == "absent":
                matched = False

            else:
                matched = None

        if matched is True:
            selected = prefix_config["when_true"]

        elif matched is False:
            selected = prefix_config["when_false"]


        else:
            # 兩次 VLM 都無法確認「總計新臺幣」
            # 為避免影響後續既有檢核流程，內部規則維持原本 when_false，
            # 但聯式不做判定，輸出 None
            print(
                "⚠️  [總計新臺幣判斷] "
                "兩次 VLM 仍無法確認，聯式輸出 None"
            )

            selected = prefix_config["when_false"].copy()
            selected["form_type"] = None
    else:
        selected = prefix_config

    detail_fields_key = selected.get("detail_fields", "FULL")
    return {
        "prefix": prefix,
        "rules": selected["rules"],
        "detail_fields": _DETAIL_PRESETS.get(detail_fields_key, _DETAIL_PRESETS["FULL"]),
        "聯式": selected.get("form_type"),
        "unknown": False
    }
    # prefix = invoice_no[:2].upper()

    # if prefix not in INVOICE_PREFIX_RULES:
    #     print(f"⚠️  發票前綴 [{prefix}] 找不到對應檢核規則，轉人工審核")
    #     return {"prefix": prefix, "rules": None, "detail_fields": None, "unknown": True}

    # prefix_config = INVOICE_PREFIX_RULES[prefix]
    # return {
    #     "prefix":        prefix,
    #     "rules":         prefix_config["rules"],
    #     "detail_fields": prefix_config["detail_fields"],  # ✅ 新增
    #     "unknown":       False
    # }

def cv2_imwrite_unicode(path: str, img):
    """支援中文路徑的 cv2.imwrite 替代函式"""
    ext = os.path.splitext(path)[1]  # 取副檔名，如 .jpg
    result, encoded = cv2.imencode(ext, img)
    if result:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    return False

def normalize_company_name(name: str) -> str:
    """正規化公司名稱，處理常見異體字與全半形差異"""
    if not name:
        return name
    name = re.sub(r"\s+", "", name)
    # ✅ 台 / 臺 視為相同
    name = name.replace("臺", "台")
    # ✅ 其他常見異體字
    name = name.replace("說", "説")
    name = name.replace("著", "着")
    return name

def is_chinese_amount_match(ocr_amount: str, std_amount: str) -> tuple[bool, str]:
    """
    比對中文大寫金額，處理前綴零和「元整」的差異
    """
    if not ocr_amount or not std_amount:
        return False, "資料缺失"

    ocr = re.sub(r"\s+", "", ocr_amount)
    std = re.sub(r"\s+", "", std_amount)

    # 1. 完全相等
    if ocr == std:
        return True, "完全相符"

    # 2. OCR結果包含於標準答案中（標準答案有前綴零或元整）
    if len(ocr) >= 2 and ocr in std:
        return True, "核心金額相符（標準答案含前綴零或元整）"

    # 3. 去除「元整」後比對
    ocr_clean = re.sub(r'元整$', '', ocr)
    std_clean  = re.sub(r'元整$', '', std)
    if ocr_clean and ocr_clean in std_clean:
        return True, "去除元整後核心金額相符"

    # 4. 去除前綴零（零仟零佰零拾零萬）後比對
    std_no_zero = re.sub(r'^[零仟佰拾萬]+', '', std_clean)
    if ocr_clean and ocr_clean == std_no_zero:
        return True, "去除前綴零後完全相符"

    return False, "不相符"

# =========================
# 讀取 Excel 標準答案
# =========================
def load_excel_standard(excel_path):
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = df.columns.str.strip()
    print(f"[DEBUG] Excel 欄位清單: {list(df.columns)}")  # ✅ 加這行確認欄位名稱
    df = df.fillna("")

    standard_dict = {}
    for _, row in df.iterrows():
        invoice_no = str(row.get("發票號碼", "")).strip()
        if not invoice_no:
            continue

        # ✅ 解析明細項目 JSON 字串
        detail_items = []
        raw_detail = str(row.get("明細項目", "")).strip()

        print(f"[DEBUG] 發票 [{invoice_no}] 明細項目原始值: '{raw_detail}'")  # ✅ 加這行

        if raw_detail:
            try:
                parsed = json.loads(raw_detail)
                if isinstance(parsed, list):
                    # 數字欄位統一轉字串並去除逗號
                    for item in parsed:
                        detail_items.append({
                            "品名": str(item.get("品名", "")).strip(),
                            "數量": str(item.get("數量", "")).replace(",", "").strip(),
                            "單價": str(item.get("單價", "")).replace(",", "").strip(),
                            "金額": str(item.get("金額", "")).replace(",", "").strip(),
                        })
            except json.JSONDecodeError:
                print(f"⚠️  發票 [{invoice_no}] 明細項目 JSON 解析失敗：{raw_detail}")

        standard_dict[invoice_no] = {
            "發票號碼":    invoice_no,
            "金額大寫中文": str(row.get("金額大寫中文", "")).strip(),
            "年度期間":    str(row.get("年度期間", "")).strip(),
            "廠商統編":    str(row.get("廠商統編", "")).strip(),
            "廠商名稱":    str(row.get("廠商名稱", "")).strip(),
            "未稅金額":    str(row.get("未稅金額", "")).replace(",", "").strip(),
            "稅額":       str(row.get("稅額", "")).replace(",", "").strip(),
            "合計金額":    str(row.get("合計金額", "")).replace(",", "").strip(),
            "明細項目":    detail_items   # ✅ 解析後的 list
        }

    print(f"已載入標準答案，共 {len(standard_dict)} 筆")
    return standard_dict

# 買方統編固定值（本系統的買方永遠是此統編）
BUYER_TAX_ID_FIXED = "05637971"
BUYER_COMPANY_NAME_FIXED = "燿華電子股份有限公司"

BUYER_COMPANY_NAME_ACCEPTED = [
    "燿華電子股份有限公司",
    "燿華電子(股)公司",
]

# 發票日期格式（民國 / 西元），供格式驗證與年月解析共用
_DATE_PATTERNS = [
    r'\d{2,3}年\d{1,2}月\d{1,2}日',
    r'\d{4}-\d{1,2}-\d{1,2}',
    r'\d{4}/\d{1,2}/\d{1,2}',
    r'\d{2,3}/\d{1,2}/\d{1,2}',
    r'\d{2,3}-\d{1,2}-\d{1,2}',
    r'\d{4}年\d{1,2}月\d{1,2}日',
    r'\d{2,3}\.\d{1,2}\.\d{1,2}',
    r'\d{4}\.\d{1,2}\.\d{1,2}',
]


def parse_invoice_year_month(date_str: str):
    """從發票日期解析西元年份與月份；民國年會自動加 1911。"""
    if not date_str:
        return None

    match = re.match(r'(\d{2,4})[年/\-\.](\d{1,2})', str(date_str).strip())
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    if year < 200:
        year += 1911
    return year, month

def extract_invoice_year(date_str: str):
    """從各種發票日期字串中萃取西元年份。"""
    if not date_str:
        return None

    text = str(date_str).strip()

    # 1. 優先抓 4 碼西元年，例如 2026
    match = re.search(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', text)
    if match:
        return int(match.group(1))

    # 1.5 處理前導 0 的民國日期，例如 0115.08.03、0115/08/03、0115-08-03
    match = re.search(
        r'(?<!\d)0(\d{3})[\/\-.]\d{1,2}[\/\-.]\d{1,2}(?!\d)',
        text
    )

    if match:
        roc_year = int(match.group(1))

        if 1 <= roc_year <= 199:
            return roc_year + 1911

    # 2. 再抓民國年，例如 115年、民國115年、中華民國115年
    match = re.search(r'(?:中華民國|民國)?\s*(\d{2,3})\s*年', text)
    if match:
        roc_year = int(match.group(1))

        # 避免把奇怪的小數字誤當年份
        if 1 <= roc_year <= 199:
            return roc_year + 1911

    # 3. 處理 115/07/15、115-07-15、115.07.15
    match = re.search(
        r'(?<!\d)(\d{2,3})[\/\-.]\d{1,2}[\/\-.]\d{1,2}(?!\d)',
        text
    )

    if match:
        roc_year = int(match.group(1))

        if 1 <= roc_year <= 199:
            return roc_year + 1911

    return None

# =========================
# 與 Excel 標準答案比對
# =========================
def compare_with_standard(buyer_tax_id, seller_tax_id, buyer_company_name, seller_company_name,
                           amount_validation, extracted_invoice_no, llm_fields=None,
                           active_rules=None, active_detail_fields=None,
                           tax_type=None, tax_found=False,
                           std_sales_amount=None, std_tax_amount=None, std_total_amount=None,invoice_date=None,remark_text=None):
    """
    將 OCR 擷取結果與標準答案逐欄比對（新版邏輯）

    Args:
        buyer_tax_id:        OCR 辨識到的買方統編
        seller_tax_id:       OCR 辨識到的賣方統編
        buyer_company_name:  OCR 辨識到的買方公司名稱
        seller_company_name: OCR 辨識到的賣方公司名稱
        amount_validation:   金額檢核結果 dict
        extracted_invoice_no: OCR 辨識到的發票號碼
        llm_fields:          LLM 擷取的欄位結果
        active_rules:        本張發票適用的檢核項目清單
        active_detail_fields: 明細項目要比對的欄位清單
        tax_type:            稅別字串（"應稅" / "零稅率" / "免稅" / None）
        tax_found:           發票上是否有勾選其中一項（True=已勾選，False=找不到）
        std_sales_amount:    標準答案未稅金額（由明細加總計算）
        std_tax_amount:      標準答案稅額（由稅別計算）
        std_total_amount:    標準答案合計金額（未稅+稅額）
    Returns:
        dict: 各欄位比對結果
    """
    # # 找不到對應標準答案時，直接回傳錯誤訊息
    # if standard is None:
    #     return {"比對結果": f"找不到發票號碼 [{extracted_invoice_no}] 對應的標準答案"}

    compare = {}

    # ✅ 每個比對項目前先確認是否在 active_rules 內
    if "發票號碼" in active_rules:
        # --- 0. 發票號碼：比對 Excel 標準答案 ---
        # std_invoice_no = standard.get("發票號碼", "").strip()
        # compare["發票號碼"] = {
        #     "標準答案": std_invoice_no,
        #     "OCR結果":  extracted_invoice_no or "",
        #     "是否一致": (extracted_invoice_no == std_invoice_no)
        # }
        compare["發票號碼"] = {
        "OCR結果":  extracted_invoice_no or "",
        "說明":     "只要有值即通過",
        "是否一致": bool(extracted_invoice_no)
    }

    if "買方統編" in active_rules:
        # --- 1. 買方統編：與固定值 05637971 比對 ---
        compare["買方統編"] = {
            "標準答案": BUYER_TAX_ID_FIXED,
            "OCR結果":  buyer_tax_id or "",
            "是否一致": (buyer_tax_id == BUYER_TAX_ID_FIXED)
        }

    if "買方公司名稱" in active_rules:
        # --- 2. 買方公司名稱：與固定值比對 ---
        is_match, match_method = is_buyer_company_name_match(
            buyer_company_name
        )
        compare["買方公司名稱"] = {
            "標準答案": BUYER_COMPANY_NAME_FIXED,
            "OCR結果":  buyer_company_name or "",
            "比對方式": match_method,
            "是否一致": is_match
        }

    if "賣方統編" in active_rules:
        # --- 3. 賣方統編：必須是完整 8 位數字才通過（不比對 Excel）---
        ocr_seller_tax = (seller_tax_id or "").strip()

        seller_tax_valid = bool(
            re.fullmatch(r"\d{8}", ocr_seller_tax)
        )

        compare["賣方統編"] = {
            "標準答案": "（8位數字即通過）",
            "OCR結果": ocr_seller_tax,
            "是否一致": seller_tax_valid
        }

    if "賣方公司名稱" in active_rules:
        # --- 4. 賣方公司名稱：只要 OCR 有值就通過（不比對 Excel）---
        ocr_seller_name = (seller_company_name or "").strip()
        compare["賣方公司名稱"] = {
            "標準答案": "（有值即通過）",
            "OCR結果":  ocr_seller_name,
            "是否一致": bool(ocr_seller_name)
        }

    # --- ✅ 營業稅稅別判斷：有勾選其中一項才通過 ---
    if "營業稅稅別判斷" in active_rules:
        if tax_found and tax_type is not None:
            compare["營業稅稅別判斷"] = {
                "結果":   tax_type,
                "說明":   f"發票上勾選項目為「{tax_type}」",
                "是否一致": True
            }
        elif tax_type == "":
            # 補救流程已窮盡仍無法判斷，輸出空字串並視為通過（不再重試）
            compare["營業稅稅別判斷"] = {
                "結果":   "未找到勾選",
                "說明":   "補救流程已窮盡，發票上仍無法判斷稅別",
                "是否一致": True
            }
        else:
            # 找不到勾選：輸出失敗
            compare["營業稅稅別判斷"] = {
                "結果":   "未找到勾選",
                "說明":   "發票上找不到應稅/零稅率/免稅的勾選",
                "是否一致": False
            }

    # --- ✅ LLM 欄位比對 ---
    if llm_fields:
        if "年度期間" in active_rules:
            # 年度期間：不再與 Excel 比對，只要有辨識值即通過
            ocr_period = (llm_fields.get("年度期間") or "").strip()
            compare["年度期間"] = {
                "OCR結果":  ocr_period,
                "說明":     "只要有值即通過",
                "是否一致": bool(ocr_period)
            }

        if "未稅金額" in active_rules:
            # --- 5. 未稅金額：標準答案=明細金額加總，OCR結果維持原邏輯 ---
            ocr_sales = (llm_fields.get("未稅金額") or "").replace(",", "").strip()
            std_sales_str = str(std_sales_amount) if std_sales_amount is not None else ""
            # 比對時去除小數點後多餘的零
            try:
                ocr_sales_num = int(float(ocr_sales)) if ocr_sales else None
                std_sales_num = int(std_sales_amount) if std_sales_amount is not None else None
                is_sales_match = (ocr_sales_num == std_sales_num) if (ocr_sales_num is not None and std_sales_num is not None) else False
            except (ValueError, TypeError):
                is_sales_match = (ocr_sales == std_sales_str)
            compare["未稅金額"] = {
                "標準答案": std_sales_str,
                "OCR結果":  ocr_sales,
                "說明":     "標準答案為明細金額加總",
                "是否一致": is_sales_match
            }

        if "稅額" in active_rules:
            # --- 6. 稅額：標準答案依稅別計算，OCR結果維持原邏輯 ---
            ocr_tax = (llm_fields.get("稅額") or "").replace(",", "").strip()
            std_tax_str = str(std_tax_amount) if std_tax_amount is not None else ""
            try:
                ocr_tax_num = int(float(ocr_tax)) if ocr_tax else None
                std_tax_num = int(std_tax_amount) if std_tax_amount is not None else None
                is_tax_match = (ocr_tax_num == std_tax_num) if (ocr_tax_num is not None and std_tax_num is not None) else False
            except (ValueError, TypeError):
                is_tax_match = (ocr_tax == std_tax_str)
            tax_note = "應稅（未稅金額×5%）" if tax_type == "應稅" else "免稅（固定為0）"
            compare["稅額"] = {
                "標準答案": std_tax_str,
                "OCR結果":  ocr_tax,
                "說明":     f"標準答案依{tax_note}計算",
                "是否一致": is_tax_match
            }

        if "合計金額" in active_rules:
            # --- 7. 合計金額：標準答案=未稅+稅額，OCR結果維持原邏輯 ---
            ocr_total = (llm_fields.get("合計金額") or "").replace(",", "").strip()
            std_total_str = str(std_total_amount) if std_total_amount is not None else ""
            try:
                ocr_total_num = int(float(ocr_total)) if ocr_total else None
                std_total_num = int(std_total_amount) if std_total_amount is not None else None
                is_total_match = (ocr_total_num == std_total_num) if (ocr_total_num is not None and std_total_num is not None) else False
            except (ValueError, TypeError):
                is_total_match = (ocr_total == std_total_str)
            compare["合計金額"] = {
                "標準答案": std_total_str,
                "OCR結果":  ocr_total,
                "說明":     "標準答案為未稅金額+稅額",
                "是否一致": is_total_match
            }

        if "金額大寫中文" in active_rules:
            original_chinese = (
                llm_fields.get("金額大寫中文") or ""
            ).strip()

            # ✅ 修復字首「非法字元 + 金額單位」
            repaired_chinese = repair_leading_invalid_amount_slot(
                original_chinese
            )

            # ✅ 關鍵：把修正結果寫回 llm_fields
            if repaired_chinese != original_chinese:
                llm_fields["金額大寫中文"] = repaired_chinese

                print(
                    "[中文金額回寫] 已更新 llm_fields："
                    f"{original_chinese!r} → {repaired_chinese!r}"
                )

            # ✅ 後續一律使用修正後的內容
            ocr_chinese = repaired_chinese

            is_chinese_match, chinese_match_method = (
                compare_chinese_amount_meaning_by_llm(
                    ocr_chinese,
                    std_total_amount
                )
            )

            compare["金額大寫中文"] = {
                "標準答案": f"合計金額 {std_total_amount} 的中文大寫",
                "OCR結果": ocr_chinese,
                "比對方式": chinese_match_method,
                "是否一致": is_chinese_match
            }

        if "明細項目" in active_rules:
            # --- 9. 明細項目：只要每筆的各欄位有值就通過 ---
            ocr_items = llm_fields.get("明細項目", [])
            compare["明細項目"] = compare_detail_items_has_value(
                ocr_items,
                active_detail_fields=active_detail_fields
            )
    if "發票日期" in active_rules:
        # ✅ 發票日期加入 compare_result
        compare["發票日期"] = {
            "OCR結果":  invoice_date,
            "說明":     "只要有值即通過",
            "是否一致":  bool(invoice_date)
        }
    if "備註" in active_rules:
        compare["備註"] = {
            "OCR結果":  remark_text,
            "說明":     "只有原文出現「備註」關鍵字時才需要有值，否則視為一致",
            "是否一致": bool(remark_text) #(not remark_expected) or 
        }
    # --- 整體通過判斷：所有欄位都一致才算通過 ---
    compare["全部比對通過"] = all(
        v["是否一致"] for v in compare.values() if isinstance(v, dict)
    )

    return compare

# =========================
# OCR 擷取買方公司名稱的函式
# =========================
def extract_buyer_company_name(page_text_clean):
    lines = page_text_clean.split("\n")
    normalized_lines = [re.sub(r"\s+", "", l) for l in lines]

    for idx, line in enumerate(normalized_lines):

        is_buyer_line  = False
        content_idx    = idx

        # 正常情況：「買方」在同一行
        if re.search(r'買方|購買人', line):
            is_buyer_line = True

        # ✅ 跨行情況：「方:」單獨成行
        # 往前 5 行找是否有含「買」字的行（處理「統買地\n編:\n方:」的情況）
        elif re.search(r'^方[:：]', line):
            context_before = normalized_lines[max(0, idx-5):idx]
            if any('買' in l for l in context_before):
                is_buyer_line = True

        if not is_buyer_line:
            continue

        # 同行找公司名稱
        m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', line)
        if m:
            return m.group(1)

        # 往後 3 行找
        for next_line in normalized_lines[idx+1:idx+4]:
            m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', next_line)
            if m:
                return m.group(1)

    # Fallback：直接掃全文（處理 OCR 首字誤判，如燿→耀）
    full_text = "".join(normalized_lines)
    # 找所有出現的「XX電子股份有限公司」類型
    m = re.search(r'[\u4e00-\u9fff]{1,2}華電子股份有限公司', full_text)
    if m:
        return m.group()

    return None

# =========================
# 抓買方/賣方統一編號（上下文判斷）
# =========================
def extract_tax_id_by_context(page_text_clean):
    buyer_tax_id  = None
    seller_tax_id = None

    lines = page_text_clean.split("\n")
    normalized_lines = [re.sub(r"\s+", "", l) for l in lines]
    full_text = "".join(normalized_lines)

    # ✅ 買方統編：固定已知，直接比對全文
    if BUYER_TAX_ID_FIXED in full_text:
        buyer_tax_id = BUYER_TAX_ID_FIXED

    # ✅ 賣方統編：策略1 - 「統一編號:XXXXXXXX」格式（同行）
    for line in normalized_lines:
        m = re.search(r'統一?編號[:：](\d{8})', line)
        if m:
            candidate = m.group(1)
            if candidate != BUYER_TAX_ID_FIXED:
                seller_tax_id = candidate
                break

    # ✅ 賣方統編：策略2 - 跨行「統一編號\n:12345678」
    if seller_tax_id is None:
        for idx, line in enumerate(normalized_lines):
            if not re.search(r'統一?編號', line):
                continue
            for next_line in normalized_lines[idx+1:idx+4]:
                m = re.search(r'^[:：]?(\d{8})$', next_line)
                if m:
                    candidate = m.group(1)
                    if candidate != BUYER_TAX_ID_FIXED:
                        seller_tax_id = candidate
                    break
            if seller_tax_id:
                break

    # ✅ 賣方統編：策略3 - 「買方 賣方」同行，下一行緊接兩個統編
    # 處理「買方  賣方\n05637971  04406559」的格式
    if seller_tax_id is None:
        for idx, line in enumerate(normalized_lines):
            if not (re.search(r'買方', line) and re.search(r'賣方', line)):
                continue
            # 往後 3 行找含兩個 8 位數字的行
            for next_line in normalized_lines[idx+1:idx+4]:
                tax_ids = re.findall(r'\d{8}', next_line)
                if len(tax_ids) >= 2:
                    for tid in tax_ids:
                        if tid != BUYER_TAX_ID_FIXED:
                            seller_tax_id = tid
                            break
                elif len(tax_ids) == 1 and tax_ids[0] != BUYER_TAX_ID_FIXED:
                    # 只有一個且非買方統編
                    seller_tax_id = tax_ids[0]
                if seller_tax_id:
                    break
            if seller_tax_id:
                break

    # ✅ 賣方統編：策略4 - 「賣方」單獨行，往後幾行找 8 位數字
    # 處理「賣方\n04406559」的格式
    if seller_tax_id is None:
        for idx, line in enumerate(normalized_lines):
            if not re.search(r'^賣方$|^賣\s*方$', line):
                continue
            for next_line in normalized_lines[idx+1:idx+5]:
                m = re.search(r'(\d{8})', next_line)
                if m:
                    candidate = m.group(1)
                    if candidate != BUYER_TAX_ID_FIXED:
                        seller_tax_id = candidate
                        break
            if seller_tax_id:
                break

    # ✅ 賣方統編：策略5 - 「賣方XXXXXXXX」或「賣方:XXXXXXXX」直接在同行
    if seller_tax_id is None:
        for line in normalized_lines:
            m = re.search(r'賣方[:：]?(\d{8})', line)  # ✅ 冒號設為可選
            if m:
                candidate = m.group(1)
                if candidate != BUYER_TAX_ID_FIXED:
                    seller_tax_id = candidate
                    print(f"[DEBUG] 策略5 找到賣方統編: {seller_tax_id}")
                    break

    print(f"[DEBUG] 買方統編={buyer_tax_id}, 賣方統編={seller_tax_id}")
    return buyer_tax_id, seller_tax_id

# 買方公司名稱比對（忽略第一個字，因為 OCR 常誤判罕見字）
def is_company_name_match(ocr_name, standard_name):
    if not ocr_name or not standard_name:
        return False, "資料缺失"

    # ocr = re.sub(r"\s+", "", ocr_name)
    # std = re.sub(r"\s+", "", standard_name)
    # ✅ 先正規化再比對
    ocr = normalize_company_name(ocr_name)
    std = normalize_company_name(standard_name)

    # 1. 完全相等
    if ocr == std:
        return True, "完全相符"

    # # 2. 忽略第一個字比對
    # if len(ocr) >= 2 and len(std) >= 2:
    #     if ocr[1:] == std[1:]:
    #         return True, f"忽略首字相符（OCR首字:{ocr[0]} 標準首字:{std[0]}）"

    # 3. 標準答案包含於 OCR 結果中（OCR多抓了前後雜訊）
    if len(std) >= 4 and std in ocr:
        return True, "標準答案包含於OCR結果中"

    # # 4. 忽略首字後，標準答案包含於 OCR 結果中
    # if len(std) >= 4 and std[1:] in ocr:
    #     return True, f"忽略首字後包含於OCR結果中（標準首字:{std[0]}）"
    

    return False, "不相符"

def is_buyer_company_name_match(ocr_name):
    if not ocr_name:
        return False, "資料缺失"

    for accepted_name in BUYER_COMPANY_NAME_ACCEPTED:
        is_match, match_method = is_company_name_match(
            ocr_name,
            accepted_name
        )

        if is_match:
            return True, f"{match_method}：{accepted_name}"

    return False, "不相符"


def extract_seller_company_name(page_text_clean):
    VALID_EXTENSION_PATTERN = r'[\u4e00-\u9fff]{1,8}(分公司|辦事處|營業所|工廠|廠|事業部|物流中心|倉儲中心|貨櫃集散站|集散站|貨櫃場|營業處|服務中心|配送中心)'
    TAX_CATEGORY_PATTERN = r'^(應稅|免稅|零稅率|營業稅|銷售額合計|合計|總計|銷售額)$'
    MAX_FRAGMENT_LEN = 12

    lines = page_text_clean.split("\n")
    normalized_lines = [re.sub(r"\s+", "", l) for l in lines]

    print("=== [DEBUG] extract_seller_company_name ===")

    for idx, line in enumerate(normalized_lines):
        is_seller_line = False
        seller_content_idx = idx

        if (re.search(r'賣方|賣\s*方', line) or
            ('賣' in line and '方' in line and re.search(r'有限公司', line))):
            is_seller_line = True
        elif line == '賣':
            next_line = normalized_lines[idx+1] if idx+1 < len(normalized_lines) else ''
            if next_line.startswith('方'):
                is_seller_line = True
                seller_content_idx = idx + 1
        elif re.search(r'^營業人[:：]', line):
            is_seller_line = True

        print(f"  [{idx}] normalized='{line}' | is_seller={is_seller_line}")

        if not is_seller_line:
            continue

        company_name = ""
        base_idx = seller_content_idx
        trigger_line = normalized_lines[seller_content_idx]

        m = re.search(r'([\u4e00-\u9fff]{2,10}(?:股份)?有限公司[\u4e00-\u9fff]*)', trigger_line)
        if m:
            company_name = m.group(1)
            print(f"  → 同行找到: '{company_name}'")
        else:
            accumulated = re.sub(r'^[賣方:：\s]+', '', re.sub(r'[^\u4e00-\u9fff]', '', trigger_line))
            print(f"  → 累積起點: '{accumulated}'，往後掃描...")

            for offset in range(1, 10):
                next_idx = seller_content_idx + offset
                if next_idx >= len(normalized_lines):
                    break
                next_norm = normalized_lines[next_idx]

                if re.search(r'\d{4,}', next_norm):
                    continue
                if re.search(TAX_CATEGORY_PATTERN, next_norm):
                    print(f"    [{next_idx}] 跳過稅率分類詞: '{next_norm}'")
                    continue

                chinese_part = re.sub(r'[^\u4e00-\u9fff]', '', next_norm)

                if len(chinese_part) > MAX_FRAGMENT_LEN:
                    print(f"    [{next_idx}] 片段過長({len(chinese_part)}字)，停止: '{next_norm}'")
                    break

                if chinese_part:
                    accumulated += chinese_part
                    print(f"    [{next_idx}] 累積: '{accumulated}'")

                m = re.search(r'([\u4e00-\u9fff]{2,10}(?:股份)?有限公司[\u4e00-\u9fff]*)', accumulated)
                if m:
                    company_name = m.group(1)
                    base_idx = next_idx
                    print(f"  → 累積後找到: '{company_name}'")
                    break

        if not company_name:
            print(f"  → 找不到公司名稱，continue")
            continue

        for next_line in normalized_lines[base_idx+1:base_idx+6]:
            if not next_line:
                continue
            if re.search(VALID_EXTENSION_PATTERN, next_line):
                company_name += next_line
                continue
            if re.search(r'\d', next_line):
                break
            if len(re.findall(r'[A-Za-z]', next_line)) >= 2:
                break
            continue

        print(f"  → 最終結果: '{company_name}'")
        print("===========================================")
        return company_name

    # ✅ Fallback：正常流程找不到時，掃全文找所有「有限公司」
    # 排除買方公司名稱，剩下的就是賣方
    print("  → Fallback：掃全文找賣方公司名稱")
    buyer_normalized = re.sub(r"\s+", "", BUYER_COMPANY_NAME_FIXED)
    for line in normalized_lines:
        m = re.search(r'([\u4e00-\u9fff]{2,}(?:股份)?有限公司[\u4e00-\u9fff]*)', line)
        if m:
            candidate = m.group(1)
            # 排除買方（允許首字誤判，用忽略首字比對排除）
            is_buyer = (
                candidate == buyer_normalized or
                (len(candidate) >= 2 and len(buyer_normalized) >= 2 and
                 candidate[1:] == buyer_normalized[1:])
            )
            if not is_buyer:
                print(f"  → Fallback 找到: '{candidate}'")
                print("===========================================")
                return candidate

    print("  → 未找到賣方行，回傳 None")
    print("===========================================")
    return None

# =========================
# 中文字型（視覺化用）
# =========================
# font_path = r"C:\Windows\Fonts\msjh.ttc"
font       = ImageFont.truetype(FONT_PATH, 28)   # 大字（摘要資訊）
small_font = ImageFont.truetype(FONT_PATH, 20)   # 小字（OCR 標注）

# =========================
# 工具函式
# =========================
def clean_text(text: str) -> str:
    """將全形標點符號統一轉成半形，方便後續 Regex 比對"""
    text = text.replace("：", ":")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    return text

def normalize_text(text: str) -> str:
    """clean_text 後再移除所有空白"""
    text = clean_text(text)
    text = re.sub(r"\s+", "", text)
    return text

def poly_to_bbox(poly):
    """將多邊形頂點座標轉換為 [x1, y1, x2, y2] 的外接矩形"""
    arr = np.array(poly)
    x1 = int(np.min(arr[:, 0]))
    y1 = int(np.min(arr[:, 1]))
    x2 = int(np.max(arr[:, 0]))
    y2 = int(np.max(arr[:, 1]))
    return [x1, y1, x2, y2]

def point_in_bbox(px, py, bbox):
    """判斷點 (px, py) 是否在 bbox 內"""
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2

def bbox_area(bbox):
    """計算 bbox 面積"""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def bbox_intersection_area(b1, b2):
    """計算兩個 bbox 的交集面積"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0  # 無交集
    return (x2 - x1) * (y2 - y1)

def overlap_ratio(inner_bbox, outer_bbox):
    """
    計算 inner_bbox 有多少比例落在 outer_bbox 內
    回傳值介於 0.0 ~ 1.0
    """
    inter = bbox_intersection_area(inner_bbox, outer_bbox)
    area = bbox_area(inner_bbox)
    if area == 0:
        return 0.0
    return inter / area

def sort_bbox_texts(items, y_tolerance=10):
    """
    將 OCR items 依照閱讀順序排序（先由上到下分行，同行內由左到右）
    
    Args:
        items:       OCR item 清單，每個 item 需有 bbox 欄位
        y_tolerance: 同一行的 Y 座標允許誤差（像素）
    Returns:
        排序後的 item 清單
    """
    if not items:
        return []

    items = sorted(items, key=lambda x: x["bbox"][1])  # 先依 Y 排序
    lines = []

    for item in items:
        y1 = item["bbox"][1]
        placed = False

        # 找是否有已存在的行可以歸入（Y 差距在容忍範圍內）
        for line in lines:
            line_y = int(np.mean([t["bbox"][1] for t in line]))
            if abs(y1 - line_y) <= y_tolerance:
                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])  # 建立新行

    # 每行內依 X 座標由左到右排序
    for line in lines:
        line.sort(key=lambda x: x["bbox"][0])

    # 行與行之間依最上方 Y 座標排序
    lines.sort(key=lambda line: min(t["bbox"][1] for t in line))

    result = []
    for line in lines:
        result.extend(line)

    return result

# =========================
# OCR 執行與解析
# =========================
def run_ocr(image_path: str = None, image_bytes: bytes = None, filename: str = None) -> list[dict]:
    """
    呼叫 PaddleOCR 容器 API，並轉成原流程使用的 OCR item 格式。

    - image_path：本地圖片路徑（原行為）。
    - image_bytes + filename：記憶體中的圖片 bytes，不需要本地檔案
      （MinIO-only 模式使用，圖片不落地本機磁碟）。
    """
    file_obj = None
    if image_bytes is not None:
        filename = filename or "page.jpg"
        content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            content_type = "image/jpeg"
        file_obj = io.BytesIO(image_bytes)
    else:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"OCR 輸入圖片不存在：{image_path}")
        filename = filename or os.path.basename(image_path)
        content_type, _ = mimetypes.guess_type(image_path)
        if not content_type:
            content_type = "application/octet-stream"

    print(f"[PaddleOCR API] 開始辨識：{filename}")
    print(f"[PaddleOCR API] URL：{OCR_API_URL}")

    response = None
    try:
        if file_obj is not None:
            response = requests.post(
                OCR_API_URL,
                files={"file": (filename, file_obj, content_type)},
                timeout=(10, OCR_API_TIMEOUT),
            )
        else:
            with open(image_path, "rb") as image_file:
                response = requests.post(
                    OCR_API_URL,
                    files={"file": (filename, image_file, content_type)},
                    timeout=(10, OCR_API_TIMEOUT),
                )
        response.raise_for_status()
    except requests.exceptions.ConnectTimeout as exc:
        raise RuntimeError(f"PaddleOCR API 連線逾時：{OCR_API_URL}") from exc
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            f"PaddleOCR API 推論逾時，超過 {OCR_API_TIMEOUT} 秒"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"無法連線 PaddleOCR API：{OCR_API_URL}，"
            "請確認 Podman 容器及 API 服務是否啟動"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        response_text = response.text[:1000] if response is not None else ""
        status_code = response.status_code if response is not None else "unknown"
        raise RuntimeError(
            f"PaddleOCR API HTTP 錯誤：{status_code}，內容：{response_text}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"呼叫 PaddleOCR API 失敗：{exc}") from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"PaddleOCR API 回傳內容不是合法 JSON：{response.text[:1000]}"
        ) from exc

    if result.get("status") != "ok":
        raise RuntimeError(
            f"PaddleOCR API 辨識失敗：{result.get('error', '未知錯誤')}"
        )

    raw_ocr_items = result.get("ocr_items")
    if not isinstance(raw_ocr_items, list):
        raise RuntimeError("PaddleOCR API 回傳格式錯誤：ocr_items 不是 list")

    ocr_items = []
    for index, item in enumerate(raw_ocr_items):
        if not isinstance(item, dict):
            print(f"⚠️ [PaddleOCR API] 第 {index + 1} 筆不是 dict，略過")
            continue

        text_value = cc.convert(str(item.get("text") or ""))
        poly = item.get("poly") or []
        bbox = item.get("bbox")
        score = item.get("score")

        if not bbox and poly:
            try:
                bbox = poly_to_bbox(poly)
            except Exception:
                bbox = None

        if not bbox or len(bbox) != 4:
            print(
                f"⚠️ [PaddleOCR API] 第 {index + 1} 筆缺少合法 bbox，略過："
                f"text={text_value!r}"
            )
            continue

        try:
            normalized_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            normalized_score = None

        ocr_items.append({
            "text": text_value,
            "score": normalized_score,
            "poly": poly,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        })

    print(
        f"[PaddleOCR API] 辨識完成：device={result.get('device')}，"
        f"API回傳={result.get('text_count')} 筆，有效資料={len(ocr_items)} 筆"
    )
    return ocr_items

def extract_value_by_keyword(text: str, keyword_pattern: str, value_pattern: str):
    """
    在文字中找「關鍵字 + 值」的組合，回傳值的部分
    例如：「統一編號:12345678」→ 回傳「12345678」
    """
    pattern = rf'{keyword_pattern}\s*[:：]?\s*({value_pattern})'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_multiline_digits_after_keyword(text: str, keyword_pattern: str, digit_count: int = 8):
    """
    關鍵字後方跨行找指定位數的數字
    （因 OCR 有時候會把關鍵字和數字拆到不同行）
    """
    pattern = rf'{keyword_pattern}[\s:：\n]*([\d\s\n]{{{digit_count},{digit_count * 4}}})'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None

    raw_value = match.group(1)
    digits_only = re.sub(r'\D', '', raw_value)  # 只保留數字

    if len(digits_only) >= digit_count:
        return digits_only[:digit_count]  # 取前 N 碼

    return None

def extract_fields_from_ocr_text(page_text_clean):
    field_rules = {
        "統一編號": {
            "keyword_pattern": r'統一\s*編號',
            "value_pattern": r'\d{8}'
        },
        "發票號碼": {
            "keyword_pattern": r'發票號碼|發票號',
            "value_pattern": r'[A-Z]{2}[-\s]?\d{8}'  # ✅ 允許中間有「-」或空白
        }
    }

    extracted_data = {}

    for field_name, rule in field_rules.items():
        value = extract_value_by_keyword(
            page_text_clean,
            rule["keyword_pattern"],
            rule["value_pattern"]
        )
        extracted_data[field_name] = value

    if extracted_data["統一編號"] is None:
        extracted_data["統一編號"] = extract_multiline_digits_after_keyword(
            page_text_clean, r'統一\s*編號', 8
        )

    # ✅ Fallback：掃全文找發票號碼，限定 2 碼英文 + 8 碼數字
    if extracted_data["發票號碼"] is None:
        m = re.search(r'[A-Z]{2}-?\d{8}', page_text_clean)
        if m:
            raw = m.group()
            cleaned = raw.replace("-", "").replace(" ", "")
            extracted_data["發票號碼"] = cleaned
            print(f"[DEBUG] 發票號碼原始辨識: '{raw}' → 清除後: '{cleaned}'")

    return extracted_data

# =========================
# TSR（表格結構辨識）
# =========================
def merge_nearby_lines(lines, axis="x", gap_threshold=15):
    """
    將相近的線條合併為一條，避免重複偵測
    
    Args:
        lines:         線條清單（bbox 格式）
        axis:          合併方向，"x" 為垂直線，"y" 為水平線
        gap_threshold: 距離小於此值視為同一條線（像素）
    Returns:
        合併後的座標清單
    """
    if not lines:
        return []

    # 取線條中心座標
    if axis == "x":
        coords = sorted([int((l[0] + l[2]) / 2) for l in lines])
    else:
        coords = sorted([int((l[1] + l[3]) / 2) for l in lines])

    merged = []
    group = [coords[0]]

    for c in coords[1:]:
        if abs(c - group[-1]) <= gap_threshold:
            group.append(c)  # 距離夠近，歸入同一群組
        else:
            merged.append(int(sum(group) / len(group)))  # 取平均作為合併後座標
            group = [c]
    merged.append(int(sum(group) / len(group)))

    return merged

def detect_table_lines(image):
    """
    使用形態學運算偵測圖片中的表格水平線與垂直線
    
    Returns:
        merged_x:       垂直線的 X 座標清單
        merged_y:       水平線的 Y 座標清單
        horizontal_img: 水平線遮罩圖
        vertical_img:   垂直線遮罩圖
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)  # 二值化，深色線條變白色

    h, w = binary.shape

    # 水平線：長寬比大的矩形核
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    # 垂直線：長高比大的矩形核
    vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 30)))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel,   iterations=1)

    horizontal_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_contours, _   = cv2.findContours(vertical,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 過濾太短的線條（可能是雜訊）
    h_lines = []
    for cnt in horizontal_contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww > w * 0.15:  # 水平線長度須超過圖寬 15%
            h_lines.append((x, y, x + ww, y + hh))

    v_lines = []
    for cnt in vertical_contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if hh > h * 0.10:  # 垂直線高度須超過圖高 10%
            v_lines.append((x, y, x + ww, y + hh))

    # 合併相近線條
    merged_y = merge_nearby_lines(h_lines, axis="y", gap_threshold=12)
    merged_x = merge_nearby_lines(v_lines, axis="x", gap_threshold=12)

    return merged_x, merged_y, horizontal, vertical

def build_cells_from_lines(xs, ys, min_cell_w=20, min_cell_h=15):
    """
    根據水平線 Y 座標與垂直線 X 座標，建立表格 Cell 清單
    
    Args:
        xs:         垂直線 X 座標清單
        ys:         水平線 Y 座標清單
        min_cell_w: 最小 Cell 寬度（過濾雜訊）
        min_cell_h: 最小 Cell 高度（過濾雜訊）
    Returns:
        cells list: [{ row, col, bbox, texts, text }, ...]
    """
    cells = []
    if len(xs) < 2 or len(ys) < 2:
        return cells  # 線條不足，無法建立表格

    row_idx = 0
    for r in range(len(ys) - 1):
        col_idx = 0
        for c in range(len(xs) - 1):
            x1, x2 = xs[c], xs[c + 1]
            y1, y2 = ys[r], ys[r + 1]

            if (x2 - x1) >= min_cell_w and (y2 - y1) >= min_cell_h:
                cells.append({
                    "row":   row_idx,
                    "col":   col_idx,
                    "bbox":  [x1, y1, x2, y2],
                    "texts": [],
                    "text":  ""
                })
                col_idx += 1
        row_idx += 1

    return cells

def run_tsr(image):
    """
    對圖片執行完整表格結構辨識（偵測線條 → 建立 Cells）
    
    Returns:
        tsr_result:    { vertical_lines, horizontal_lines, cells }
        horizontal_img: 水平線遮罩
        vertical_img:   垂直線遮罩
    """
    xs, ys, horizontal_img, vertical_img = detect_table_lines(image)
    cells = build_cells_from_lines(xs, ys)
    tsr_result = {
        "vertical_lines":   xs,
        "horizontal_lines": ys,
        "cells":            cells
    }
    return tsr_result, horizontal_img, vertical_img

# =========================
# Cell Alignment（OCR 文字對齊到 Cell）
# =========================
def split_ocr_item_by_chars(item):
    """
    將一個 OCR item 拆成每個字元一個 item
    用於處理一段文字橫跨多個 Cell 的情況
    """
    text = item["text"]
    bbox = item["bbox"]

    if not text or len(text) <= 1:
        return [item]

    x1, y1, x2, y2 = bbox
    total_w    = x2 - x1
    char_count = len(text)

    if total_w <= 0 or char_count <= 1:
        return [item]

    char_w = total_w / char_count  # 每個字元平均寬度
    split_items = []

    for i, ch in enumerate(text):
        sx1 = int(round(x1 + i * char_w))
        sx2 = int(round(x1 + (i + 1) * char_w))
        split_items.append({
            "text": ch,
            "bbox": [sx1, y1, sx2, y2],
            "poly": [[sx1, y1], [sx2, y1], [sx2, y2], [sx1, y2]]
        })

    return split_items

def get_overlapping_cell_indices(item_bbox, cells, threshold=0.15):
    """找出與指定 bbox 重疊超過 threshold 的所有 Cell 索引"""
    matched = []
    for idx, cell in enumerate(cells):
        score = overlap_ratio(item_bbox, cell["bbox"])
        if score >= threshold:
            matched.append((idx, score))
    return matched

def align_ocr_to_cells(ocr_items, cells):
    """
    將 OCR items 對齊到 TSR 的 Cell 中
    
    策略：
    1. 若 OCR item 橫跨多個 Cell（overlap >= 2），拆成字元再對齊
    2. 先用 overlap_ratio 找最佳 Cell
    3. overlap 不夠時用中心點 fallback
    
    Returns:
        aligned_cells: 每個 Cell 附帶對齊的 OCR 文字
    """
    def merge_cell_texts(cell_texts):
        """將 Cell 內多個 OCR item 依閱讀順序合併成字串"""
        sorted_texts = sort_bbox_texts(cell_texts, y_tolerance=10)
        return "".join([t["text"] for t in sorted_texts]).strip()

    # 初始化對齊後的 Cell 清單
    aligned_cells = []
    for cell in cells:
        aligned_cells.append({
            "row":   cell["row"],
            "col":   cell["col"],
            "bbox":  cell["bbox"],
            "texts": [],
            "text":  ""
        })

    # 第一步：對跨格文字進行字元拆分
    expanded_items = []
    for item in ocr_items:
        overlaps = get_overlapping_cell_indices(item["bbox"], aligned_cells, threshold=0.15)
        if len(overlaps) >= 2 and len(item["text"]) > 1:
            expanded_items.extend(split_ocr_item_by_chars(item))  # 拆成字元
        else:
            expanded_items.append(item)

    # 第二步：將每個 item 對齊到最佳 Cell
    for item in expanded_items:
        bbox = item["bbox"]
        cx = (bbox[0] + bbox[2]) / 2  # 中心 X
        cy = (bbox[1] + bbox[3]) / 2  # 中心 Y

        best_idx   = None
        best_score = -1.0

        # 用 overlap_ratio 找重疊最高的 Cell
        for idx, cell in enumerate(aligned_cells):
            score = overlap_ratio(bbox, cell["bbox"])
            if score > best_score:
                best_score = score
                best_idx   = idx

        # overlap 太低時，改用中心點判斷
        if best_score < 0.3:
            for idx, cell in enumerate(aligned_cells):
                if point_in_bbox(cx, cy, cell["bbox"]):
                    best_idx = idx
                    break

        if best_idx is not None:
            aligned_cells[best_idx]["texts"].append(item)

    # 第三步：將 Cell 內文字排序後合併成字串
    for cell in aligned_cells:
        cell["texts"] = sort_bbox_texts(cell["texts"], y_tolerance=10)
        cell["text"]  = merge_cell_texts(cell["texts"])

    return aligned_cells

def build_table_from_cells(cells):
    """
    將 aligned cells 轉成二維表格（list of list）
    
    Returns:
        table: [ [row0col0, row0col1, ...], [row1col0, ...], ... ]
    """
    if not cells:
        return []

    max_row = max(cell["row"] for cell in cells)
    max_col = max(cell["col"] for cell in cells)

    table = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for cell in cells:
        table[cell["row"]][cell["col"]] = cell.get("text", "")

    return table

def safe_int(text):
    """將文字安全轉換成整數，失敗回傳 None"""
    if text is None:
        return None
    text = str(text).replace(",", "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None

def normalize_cell_text(text):
    """移除 Cell 文字中的所有空白"""
    if text is None:
        return ""
    return re.sub(r"\s+", "", str(text))

def find_header_row_and_columns(table_data):
    """
    在表格中找「品名、數量、單價、金額」的標頭行，並回傳各欄的 col index
    
    Returns:
        (header_row_idx, col_map) 或 (None, None)
    """
    for row_idx, row in enumerate(table_data):
        normalized_row = [normalize_cell_text(c) for c in row]

        col_map = {
            "品名": None,
            "數量": None,
            "單價": None,
            "金額": None
        }

        for col_idx, cell in enumerate(normalized_row):
            if "品名" in cell or cell in ["品名", "名品"]:
                col_map["品名"] = col_idx
            elif "數量" in cell:
                col_map["數量"] = col_idx
            elif "單價" in cell or "價單" in cell:
                col_map["單價"] = col_idx
            elif "金額" in cell or "額金" in cell:
                col_map["金額"] = col_idx

        # 四個欄位都找到才算找到標頭
        if all(v is not None for v in col_map.values()):
            return row_idx, col_map

    return None, None

# =========================
# 可視化
# =========================
def draw_ocr_boxes(img, ocr_items):
    """在圖片上畫出 OCR 文字框（綠色多邊形）"""
    for item in ocr_items:
        pts = np.array(item["poly"]).astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    return img

def draw_cells(img, cells):
    """在圖片上畫出 TSR Cell 邊框（藍色矩形）"""
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img

def draw_texts_with_pil(img, ocr_items, summary_lines, aligned_cells=None):
    """
    用 PIL 在圖片上疊加中文文字（OpenCV 不支援中文）
    
    Args:
        img:           原始圖片（BGR numpy array）
        ocr_items:     OCR 結果，用紅色標注在框框上方
        summary_lines: 頁面摘要文字，顯示在左上角（藍色）
        aligned_cells: 對齊後的 Cell，顯示對齊文字（橘色）
    Returns:
        疊加文字後的圖片（BGR numpy array）
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 紅色：OCR 辨識文字標注
    for item in ocr_items:
        x1, y1, x2, y2 = item["bbox"]
        draw.text(
            (x1, max(y1 - 22, 0)),
            item["text"],
            font=small_font,
            fill=(255, 0, 0)
        )

    # 橘色：Cell 對齊後的文字
    if aligned_cells:
        for cell in aligned_cells:
            if cell["text"]:
                x1, y1, x2, y2 = cell["bbox"]
                draw.text(
                    (x1 + 2, y1 + 2),
                    cell["text"],
                    font=small_font,
                    fill=(0, 128, 255)
                )

    # 藍色：左上角摘要資訊
    summary_text = "\n".join(summary_lines)
    draw.text((20, 20), summary_text, font=font, fill=(0, 0, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================
# 金額檢核
# =========================
def validate_amounts(table_data):
    """
    從表格資料中檢核金額邏輯：
    1. 明細列：數量 × 單價 = 金額
    2. 明細金額加總 = 銷售額合計
    3. 銷售額合計 + 營業稅 = 總計
    
    Returns:
        dict: 各項檢核結果
    """
    result = {
        "detail_checks":             [],    # 每列明細的檢核結果
        "detail_sum":                None,  # 明細金額加總
        "sales_amount":              None,  # 銷售額合計
        "tax_amount":                None,  # 營業稅
        "total_amount":              None,  # 總計
        "detail_sum_match_sales":    None,  # 明細加總 == 銷售額合計？
        "sales_plus_tax_match_total": None, # 銷售額 + 稅 == 總計？
        "header_row_index":          None,  # 標頭行位置
        "detail_col_map":            None   # 各欄對應 col index
    }

    # 找標頭行
    header_row_idx, col_map = find_header_row_and_columns(table_data)
    result["header_row_index"] = header_row_idx
    result["detail_col_map"]   = col_map

    detail_sum   = 0
    detail_count = 0

    # 逐列處理明細資料（標頭行以下，遇到合計列就停止）
    if header_row_idx is not None and col_map is not None:
        for row in table_data[header_row_idx + 1:]:
            merged = "".join(normalize_cell_text(c) for c in row)

            # 遇到合計相關列，停止讀取明細
            if "銷售額合計" in merged or "營業稅" in merged or "營傢業稅" in merged or "總計" in merged:
                break

            max_idx = max(col_map.values())
            if len(row) <= max_idx:
                continue  # 欄數不足，跳過

            item_name  = normalize_cell_text(row[col_map["品名"]])
            qty        = safe_int(normalize_cell_text(row[col_map["數量"]]))
            unit_price = safe_int(normalize_cell_text(row[col_map["單價"]]))
            amount     = safe_int(normalize_cell_text(row[col_map["金額"]]))

            # 空白列跳過
            if not item_name and qty is None and unit_price is None and amount is None:
                continue

            if item_name and qty is not None and unit_price is not None and amount is not None:
                row_ok = (qty * unit_price == amount)  # 數量 × 單價 = 金額
                result["detail_checks"].append({
                    "品名": item_name,
                    "數量": qty,
                    "單價": unit_price,
                    "金額": amount,
                    "檢核結果": row_ok
                })
                detail_sum   += amount
                detail_count += 1

    if detail_count > 0:
        result["detail_sum"] = detail_sum

    # 掃全表找銷售額合計、營業稅、總計
    for row in table_data:
        merged = "".join(normalize_cell_text(c) for c in row)

        if "銷售額合計" in merged or "銷售額" in merged:
            for cell in row:
                val = safe_int(normalize_cell_text(cell))
                if val is not None:
                    result["sales_amount"] = val

        elif "營業稅" in merged or "營傢業稅" in merged or "稅" in merged:
            for cell in row:
                val = safe_int(normalize_cell_text(cell))
                if val is not None:
                    result["tax_amount"] = val

        elif "總計" in merged:
            for cell in row:
                val = safe_int(normalize_cell_text(cell))
                if val is not None:
                    result["total_amount"] = val

    # 最終邏輯驗證
    if result["detail_sum"] is not None and result["sales_amount"] is not None:
        result["detail_sum_match_sales"] = (result["detail_sum"] == result["sales_amount"])

    if (
        result["sales_amount"] is not None and
        result["tax_amount"]   is not None and
        result["total_amount"] is not None
    ):
        result["sales_plus_tax_match_total"] = (
            result["sales_amount"] + result["tax_amount"] == result["total_amount"]
        )

    return result



def process_document(file_path: str, work_dir: str) -> list[dict]:
    """
    以完整 app_old 流程處理 API 上傳文件。

    僅將原本固定的輸入/輸出路徑改為每個 job 的 work 目錄；
    OCR、TSR、LLM、VLM、重試、裁切、比對與人工審核邏輯皆沿用原程式。
    """
    work_dir_path = Path(work_dir)
    jpg_pages_dir = work_dir_path / "jpg_pages"
    output_dir = work_dir_path / "output"
    json_output_dir = work_dir_path / "json_output"
    vlm_crop_debug_dir = work_dir_path / "vlm_crop_debug"

    for directory in (jpg_pages_dir, output_dir, json_output_dir, vlm_crop_debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    jpg_pages_prefix = str(jpg_pages_dir) + os.sep
    output_prefix = str(output_dir) + os.sep
    json_output_prefix = str(json_output_dir) + os.sep
    vlm_crop_debug_prefix = str(vlm_crop_debug_dir) + os.sep

    # ← 新增：job_id（work_dir 為 results/{job_id}/work，取上一層資料夾名稱）
    job_id = work_dir_path.parent.name

    def save_or_upload(data: bytes, local_path: str, content_type: str):
        """
        統一的「落地或上傳」邏輯：
        該檔案所在的第一層資料夾（jpg_pages/output/json_output/vlm_crop_debug）
        沒有列在 storage.UPLOAD_SUBDIRS 裡 → 完全跳過，不寫本地也不上傳 MinIO；
        有列在裡面 → MinIO 已設定時直接以記憶體 bytes 上傳，不寫入本地磁碟；
        MinIO 未設定時才寫入本地檔案（方便本機除錯）。
        """
        relative_name = Path(local_path).relative_to(work_dir_path).as_posix()
        top_level_folder = relative_name.split("/", 1)[0]

        if top_level_folder not in storage.UPLOAD_SUBDIRS:
            print(f"[略過] {top_level_folder} 未列在 UPLOAD_SUBDIRS，不儲存：{local_path}")
            return

        if storage.is_configured():
            object_name = f"{job_id}/{relative_name}"
            storage.put_bytes(object_name, data, content_type=content_type)
            print(f"[MinIO] 已上傳：{object_name}")
        else:
            with open(local_path, "wb") as f_out:
                f_out.write(data)
            print(f"已儲存（本地）：{local_path}")

    def save_debug_image(image: Image.Image, local_path: str):
        """儲存 VLM 裁切除錯圖片（PIL Image），透過 save_or_upload 落地或上傳。"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        save_or_upload(buffer.getvalue(), local_path, content_type="image/png")

    def encode_jpg_bytes(img_bgr) -> bytes:
        """把 OpenCV BGR numpy array 編碼成 JPEG bytes（支援中文路徑情境下沿用同一套編碼）。"""
        ok, encoded = cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("JPEG 編碼失敗")
        return encoded.tobytes()

    PDF_NAME = Path(file_path).stem
    pages = convert_file_to_pages(file_path)
    all_pages_result = []

    for i, page in enumerate(pages):
        field_confidence = {}   # 欄位 -> 信心值
        field_source = {}       # 欄位 -> "OCR" / "LLM" / "VLM"

        def set_field_meta(field_name: str, field_value, source_name: str, model_confidence: float | None = None):
            """更新欄位來源與信心值（只做紀錄，不改既有判斷邏輯）。"""
            if field_value is None:
                return
            if isinstance(field_value, str) and not field_value.strip():
                return
            field_source[field_name] = source_name
            if model_confidence is not None:
                field_confidence[field_name] = model_confidence
                return
            if source_name == "OCR":
                field_confidence[field_name] = estimate_field_confidence(
                    ocr_items, field_name, field_value, source_name
                )
    

        # --- 圖片轉換與儲存 ---
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        jpg_filename = f"page_{i+1}_{PDF_NAME}.jpg"
        jpg_path = f"{jpg_pages_prefix}{jpg_filename}"
        jpg_bytes = encode_jpg_bytes(img)
        save_or_upload(jpg_bytes, jpg_path, content_type="image/jpeg")
        print(f"已轉換: {jpg_path}")

        # ---------------------------------
        # ✅ 0. VLM 前置判斷：是否有多張發票
        # ---------------------------------
        print(f"\n===== 第 {i+1} 頁 VLM 多發票偵測 =====")
        detection = detect_multi_invoice(page)
        print(f"  → 發票張數: {detection.get('invoice_count')}")
        print(f"  → 信心度:   {detection.get('confidence')}")
        print(f"  → 原因:     {detection.get('reason')}")

        if detection.get("has_multiple_invoices"):
            print(f"⚠️  第 {i+1} 頁偵測到多張發票，跳過辨識，請人工處理")

            _multi_invoice_fields = [
                "發票號碼", "買方統編", "買方公司名稱", "賣方統編", "賣方公司名稱",
                "營業稅稅別判斷", "未稅金額", "稅額", "合計金額", "金額大寫中文",
                "明細項目", "發票日期", "備註",
            ]
            print(f"\n第 {i+1} 頁 輸出OCR+LLM結果摘要:")
            compare_result = {}
            for _field in _multi_invoice_fields:
                print(f"  {_field}: {{'OCR結果': None, '信心值': None, '來源': None}}")
                compare_result[_field] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  折讓單日期: {{'OCR結果': None, '信心值': None}}")
            compare_result["折讓單日期"] = {"OCR結果": None, "信心值": None}
            print(f"  聯式: {{'OCR結果': None, '信心值': None, '來源': None}}")
            compare_result["聯式"] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  是否有多張發票: {{'OCR結果': True, '信心值': {repr(detection.get('confidence'))}}}")
            compare_result["是否有多張發票"] = {"OCR結果": True, "信心值": detection.get("confidence")}

            all_pages_result.append({
                "page":   i + 1,
                "status": "processed",
                "reason": "偵測到多張發票，請人工處理",
                "vlm_detection": detection,
                "compare_result": compare_result
            })
            continue  # ✅ 跳過此頁，不進行 OCR

        print(f"✅  第 {i+1} 頁為單張發票，繼續辨識")


        # ---------------------------------
        # 1. OCR branch：執行文字辨識
        # ---------------------------------
        if storage.is_configured():
            # MinIO 模式：圖片不落地，直接以記憶體 bytes 呼叫 OCR API
            ocr_items = run_ocr(image_bytes=jpg_bytes, filename=jpg_filename)
        else:
            ocr_items = run_ocr(image_path=jpg_path)

        # 將所有辨識文字合併成純文字字串
        page_text       = "\n".join([item["text"] for item in ocr_items])
        page_text_clean = clean_text(page_text)

        print(f"\n===== 第 {i+1} 頁 OCR 純文字 =====")
        print(page_text_clean)
        print("=================================\n")

        # 從 OCR 文字擷取關鍵欄位
        extracted_data = extract_fields_from_ocr_text(page_text_clean)

        # ✅ 發票號碼格式驗證：必須是 2 碼英文 + 8 碼數字
        _inv_no_raw = extracted_data.get("發票號碼")
        if _inv_no_raw:
            if not re.fullmatch(r'[A-Za-z]{2}\d{8}', _inv_no_raw):
                print(f"⚠️  發票號碼格式不符: '{_inv_no_raw}'（應為 2 碼英文 + 8 碼數字），清除並觸發 LLM/VLM 重抓")
                extracted_data["發票號碼"] = None

        # ---------------------------------
        # 2. TSR branch：表格結構辨識
        # ---------------------------------
        tsr_result, horizontal_img, vertical_img = run_tsr(img)
        cells = tsr_result["cells"]

        # ---------------------------------
        # 3. Alignment：OCR 文字對齊到 Cell
        # ---------------------------------
        aligned_cells = align_ocr_to_cells(ocr_items, cells)
        table_data    = build_table_from_cells(aligned_cells)

        # ---------------------------------
        # 3.5 金額檢核
        # ---------------------------------
        amount_validation = validate_amounts(table_data)



        # ---------------------------------
        # 4. Regex 檢核：統一編號驗證
        # ---------------------------------
        buyer_tax_id, seller_tax_id = extract_tax_id_by_context(page_text_clean)
        buyer_company_name = extract_buyer_company_name(page_text_clean)
        # seller_company_name = extract_seller_company_name(page_text_clean)
        seller_company_name = None

        seller_tax_id = None

        if buyer_company_name is not None:
            buyer_name = ['燿華電子股份有限公司','燿華電子(股)公司']
            if buyer_company_name not in buyer_name:
                print(f"⚠️  [OCR] 買方公司名稱未擷取到，觸發 LLM/VLM 重抓")
                extracted_data["買方公司名稱"] = None
                buyer_company_name = None

        # # ✅ OCR 擷取到賣方公司名稱後也做「存在性」保守檢查：
        # # 只有 exists=True 才保留，其餘（false / null）一律視為失敗並清空。
        # val_str = str(seller_company_name or "").strip()
        # if val_str and val_str.lower() not in ["null", "none", "未找到"]:
        #     verdict = verify_company_existence_by_model(val_str)
        #     if verdict.get("exists") is True:
        #         seller_company_name = val_str
        #     else:
        #         print(f"⚠️  [公司存在性檢查][OCR] '{val_str}' 不存在或不確定（{verdict}），視為 None")
        #         seller_company_name = None



        validation_result = {}
        validation_result["買方統一編號"]          = buyer_tax_id
        validation_result["買方公司名稱"]          = buyer_company_name
        validation_result["賣方統一編號"]          = seller_tax_id
        validation_result["賣方公司名稱"]    = seller_company_name
        validation_result["買方統編固定值檢核"]     = (buyer_tax_id == BUYER_TAX_ID_FIXED)  # 是否等於 05637971
        validation_result["統一編號格式正確"]       = bool(buyer_tax_id and re.fullmatch(r"\d{8}", buyer_tax_id))
        validation_result["金額檢核"]              = amount_validation
        validation_result["明細加總是否等於銷售額合計"]   = amount_validation["detail_sum_match_sales"]
        validation_result["銷售額合計加營業稅是否等於總計"] = amount_validation["sales_plus_tax_match_total"]
    

        # ---------------------------------
        # OCR 階段先填 OCR 欄位的信心值
        # ---------------------------------
        # OCR來源欄位（先給初值，後續若被 LLM/VLM 覆蓋再更新）
        ocr_field_values = {
            "發票號碼": extracted_data.get("發票號碼"),
            "買方統編": buyer_tax_id,
            "賣方統編": seller_tax_id,
            "買方公司名稱": buyer_company_name,
            "賣方公司名稱": seller_company_name,
        }

        for k, val in ocr_field_values.items():
            conf = get_ocr_confidence_for_value(ocr_items, val)
            if conf is not None:
                field_confidence[k] = conf
                field_source[k] = "OCR"
            elif val:
                field_source[k] = "OCR"


        # ---------------------------------
        # 4.5 與 Excel 標準答案比對
        # ---------------------------------
        extracted_invoice_no = extracted_data.get("發票號碼")
        lookup_invoice_no    = extracted_invoice_no
        ocr_text_with_position = build_ocr_text_with_position(ocr_items)

        def clean_invoice_no_candidate(value):
            """清理 LLM/VLM 回傳的發票號碼候選值，並驗證格式為 2 碼英文 + 8 碼數字"""
            if value is None:
                return None
            cleaned = re.sub(r'[^A-Za-z0-9]', '', str(value)).upper()
            if cleaned in ["", "NULL", "NONE", "未找到"]:
                return None
            if re.fullmatch(r'[A-Z]{2}\d{8}', cleaned):
                return cleaned
            print(f"⚠️  LLM/VLM 發票號碼格式不符: '{cleaned}'（應為 2 碼英文 + 8 碼數字）")
            return None

        # ---------------------------------
        # ✅ VLM 辨識發票年月，依年度／雙月期別動態載入字軌規則
        # ---------------------------------
        _invoice_year_for_rules = None
        _invoice_month_for_rules = None

        print(f"\n===== 第 {i+1} 頁 發票年月辨識（字軌規則用） =====")

        # # Step 1：VLM 以整張圖辨識發票日期
        # _date_vlm_result = extract_fields_from_image_region(page, ["發票日期"])
        # _date_vlm_str = re.sub(r'\s+', '', str((_date_vlm_result or {}).get("發票日期") or "").strip())

        # if _date_vlm_str and any(re.fullmatch(p, _date_vlm_str) for p in _DATE_PATTERNS):
        #     _parsed = parse_invoice_year_month(_date_vlm_str)
        #     if _parsed:
        #         _invoice_year_for_rules, _invoice_month_for_rules = _parsed
        #         print(
        #             f"  [VLM] 發票日期: {_date_vlm_str} → "
        #             f"年: {_invoice_year_for_rules}, 月: {_invoice_month_for_rules}"
        #         )
        #     else:
        #         print(f"⚠️  [VLM] 發票日期格式解析失敗: {_date_vlm_str!r}")
        # else:
        #     print(f"⚠️  [VLM] 無法辨識有效發票日期（回傳: {_date_vlm_str!r}）")

        # ---------------------------------
        # ✅ LLM 定位 + VLM 辨識發票年月
        # ---------------------------------
        _invoice_year_for_rules = None
        _invoice_month_for_rules = None

        print(f"\n===== 第 {i+1} 頁 發票年月辨識（LLM定位 + VLM） =====")

        # Step 1：LLM 根據 OCR 文字 + bbox 定位發票日期區域
        print("  [Step 1] LLM 定位發票日期區域...")

        _date_bbox = locate_field_region_by_llm(
            ocr_text_with_position,
            ["發票日期"]
        )

        if (
            _date_bbox
            and all(k in _date_bbox for k in ["x1", "y1", "x2", "y2"])
        ):
            # Step 2：裁切日期區域
            _date_crop = crop_image_region(
                page,
                [
                    _date_bbox["x1"],
                    _date_bbox["y1"],
                    _date_bbox["x2"],
                    _date_bbox["y2"],
                ],
                padding=30,
            )

            _date_crop_path = (
                f"{vlm_crop_debug_prefix}"
                f"page{i+1}_invoice_date_for_rules.png"
            )

            save_debug_image(_date_crop, _date_crop_path)

            print(
                f"  [LLM定位] 裁切圖片已儲存：{_date_crop_path}，"
                f"bbox=["
                f"{_date_bbox['x1']},"
                f"{_date_bbox['y1']},"
                f"{_date_bbox['x2']},"
                f"{_date_bbox['y2']}"
                f"]"
            )

            # Step 3：VLM 辨識裁切圖片中的發票日期
            print("  [Step 2] VLM 辨識發票日期...")

            _date_vlm_result = extract_fields_from_image_region(
                _date_crop,
                ["發票日期"]
            )

            _date_vlm_conf_map = (
                _date_vlm_result.get("__field_confidence__", {})
                if isinstance(_date_vlm_result, dict)
                else {}
            )

            # ✅ 發票日期唯一來源：最前面的 LLM定位 + VLM
            invoice_date = (
                (_date_vlm_result or {}).get("發票日期")
            )

            print(f"  [VLM] 回傳結果：{_date_vlm_result}")

            _date_vlm_str = re.sub(
                r'\s+',
                '',
                str(invoice_date or "").strip()
            )

            # Step 4：解析年份
            if _date_vlm_str:

                _parsed = extract_invoice_year(
                    _date_vlm_str
                )

                if _parsed:
                    _invoice_year_for_rules = _parsed

                    print(
                        f"✅ [LLM+VLM] 發票日期: {_date_vlm_str} → "
                        f"年: {_invoice_year_for_rules}"
                    )

                else:
                    print(
                        f"⚠️  [LLM+VLM] 發票日期格式解析失敗: "
                        f"{_date_vlm_str!r}"
                    )

            else:
                print(
                    "⚠️  [LLM+VLM] VLM 無法辨識發票日期"
                )
                # ---------------------------------
                # ✅ 發票日期補救：前面辨識失敗時，改用整張圖再跑一次 VLM
                # ---------------------------------
                if _invoice_year_for_rules is None:
                    print(
                        "⚠️  [發票日期補救] 前面無法取得有效發票日期，"
                        "改用整張圖重新進行 VLM 辨識..."
                    )

                    _date_full_vlm_result = extract_fields_from_image_region(
                        page,
                        ["發票日期"]
                    )

                    print(
                        f"  [發票日期補救][整張圖 VLM] 回傳結果："
                        f"{_date_full_vlm_result}"
                    )

                    _date_full_vlm_str = re.sub(
                        r'\s+',
                        '',
                        str(
                            (_date_full_vlm_result or {}).get("發票日期")
                            or ""
                        ).strip()
                    )

                    if _date_full_vlm_str:
                        _parsed = extract_invoice_year(
                            _date_full_vlm_str
                        )

                        if _parsed:
                            invoice_date = _date_full_vlm_str
                            _invoice_year_for_rules = _parsed

                            print(
                                f"✅ [發票日期補救][整張圖 VLM] "
                                f"發票日期: {invoice_date} → "
                                f"年: {_invoice_year_for_rules}"
                            )
                        else:
                            print(
                                f"⚠️  [發票日期補救][整張圖 VLM] "
                                f"日期格式解析失敗: {_date_full_vlm_str!r}"
                            )
                    else:
                        print(
                            "⚠️  [發票日期補救][整張圖 VLM] "
                            "仍無法辨識發票日期"
                        )

        else:
            print(
                "⚠️  [LLM+VLM] LLM 無法定位發票日期區域"
            )
            
        # ---------------------------------
        # ✅ 發票年份最終補救：從 OCR 純文字抓「中華民國XXX年」
        # ---------------------------------
        if _invoice_year_for_rules is None:
            _ocr_roc_year_match = re.search(
                r'中華民國\s*(\d{2,3})\s*年',
                str(page_text_clean or "")
            )

            if _ocr_roc_year_match:
                _ocr_roc_year = int(
                    _ocr_roc_year_match.group(1)
                )

                if 1 <= _ocr_roc_year <= 199:
                    _invoice_year_for_rules = (
                        _ocr_roc_year + 1911
                    )

                    print(
                        "✅ [發票日期補救][OCR年份] "
                        f"找到中華民國{_ocr_roc_year}年 "
                        f"→ 西元 {_invoice_year_for_rules} 年"
                    )

                    # 如果前面的 VLM 已經有抓到月日，例如「8月14」
                    # 就組成完整日期
                    if invoice_date:
                        _month_day_match = re.search(
                            r'(\d{1,2})月(\d{1,2})日?',
                            str(invoice_date)
                        )

                        if _month_day_match:
                            _month = int(
                                _month_day_match.group(1)
                            )
                            _day = int(
                                _month_day_match.group(2)
                            )

                            invoice_date = (
                                f"中華民國{_ocr_roc_year}年"
                                f"{_month}月{_day}日"
                            )

                            print(
                                "✅ [發票日期補救][OCR年份+VLM月日] "
                                f"完整日期: {invoice_date}"
                            )

        # # Step 3：仍無法取得年月時，維持 API 的人工審核狀態語意
        # if _invoice_year_for_rules is None or _invoice_month_for_rules is None:
        # Step 3：仍無法取得年份時，維持 API 的人工審核狀態語意
        if _invoice_year_for_rules is None:
            print(f"⚠️  第 {i+1} 頁無法辨識發票年月，轉人工審核")
            _multi_invoice_fields = [
                "發票號碼", "買方統編", "買方公司名稱", "賣方統編", "賣方公司名稱",
                "營業稅稅別判斷", "未稅金額", "稅額", "合計金額", "金額大寫中文",
                "明細項目", "發票日期", "備註",
            ]
            print(f"\n第 {i+1} 頁 輸出OCR+LLM結果摘要:")
            compare_result = {}
            for _field in _multi_invoice_fields:
                print(f"  {_field}: {{'OCR結果': None, '信心值': None, '來源': None}}")
                compare_result[_field] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  折讓單日期: {{'OCR結果': None, '信心值': None}}")
            compare_result["折讓單日期"] = {"OCR結果": None, "信心值": None}
            print(f"  聯式: {{'OCR結果': None, '信心值': None, '來源': None}}")
            compare_result["聯式"] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  是否有多張發票: {{'OCR結果': False, '信心值': {repr(detection.get('confidence'))}}}")
            compare_result["是否有多張發票"] = {"OCR結果": False, "信心值": detection.get("confidence")}

            all_pages_result.append({
                "page": i + 1,
                "status": "processed",
                "reason": "無法辨識發票年月，無法載入字軌規則",
                "compare_result": compare_result
            })
            continue

        _page_prefix_rules = load_invoice_prefix_rules(
            invoice_year=_invoice_year_for_rules,
        )
        print(
            f"  [字軌規則] 年={_invoice_year_for_rules}, "
            f"共 {len(_page_prefix_rules)} 個字軌"
        )

        # ✅ 依發票前兩碼決定檢核規則
        prefix_rule = get_validation_rules_by_prefix(
            extracted_invoice_no,
            page_text_clean,
            page,
            prefix_rules=_page_prefix_rules,
        )
        # # ✅ 發票前綴找不到/未設定時：先用 LLM 補抓發票號碼，再用 VLM 保底
        # if prefix_rule["prefix"] is None or prefix_rule.get("unknown"):
        #     print(f"⚠️  發票前綴無法使用，先改用 LLM 重新擷取發票號碼")
        #     llm_invoice_retry = reextract_specific_fields(ocr_text_with_position, ["發票號碼"])
        #     llm_invoice_conf_map = llm_invoice_retry.get("__field_confidence__", {}) if llm_invoice_retry else {}
        #     llm_invoice_no = clean_invoice_no_candidate(llm_invoice_retry.get("發票號碼") if llm_invoice_retry else None)

        #     if llm_invoice_no:
        #         extracted_invoice_no = llm_invoice_no
        #         lookup_invoice_no    = llm_invoice_no
        #         extracted_data["發票號碼"] = llm_invoice_no
        #         set_field_meta("發票號碼", llm_invoice_no, "LLM", llm_invoice_conf_map.get("發票號碼"))
        #         prefix_rule = get_validation_rules_by_prefix(extracted_invoice_no, page_text_clean, page)
        #         print(f"[前綴補救][LLM] 發票號碼更新為 {extracted_invoice_no}，前綴: [{prefix_rule['prefix']}] → 檢核項目: {prefix_rule['rules']}")
        #     else:
        #         print(f"⚠️  [前綴補救][LLM] 仍無法取得可用發票號碼")

        if prefix_rule["prefix"] is None or prefix_rule.get("unknown"):
            # print(f"⚠️  發票前綴仍無法使用，改用 VLM 圖片理解重新擷取發票號碼")
            # vlm_invoice_retry = extract_fields_from_image_region(page, ["發票號碼"])
            # vlm_invoice_conf_map = vlm_invoice_retry.get("__field_confidence__", {}) if vlm_invoice_retry else {}
            # vlm_invoice_no = clean_invoice_no_candidate(vlm_invoice_retry.get("發票號碼") if vlm_invoice_retry else None)
            print(f"⚠️  發票前綴仍無法使用，改用 VLM 圖片理解重新擷取發票號碼")

            # ✅ 新增：先用 LLM 根據 OCR 座標文字定位發票號碼區域，裁切後再辨識
            vlm_invoice_no = None
            vlm_invoice_conf_map = {}
            bbox_result = locate_field_region_by_llm(ocr_text_with_position, ["發票號碼"])
            if bbox_result and all(k in bbox_result for k in ["x1", "y1", "x2", "y2"]):
                invoice_no_crop = crop_image_region(
                    page,
                    [bbox_result["x1"], bbox_result["y1"], bbox_result["x2"], bbox_result["y2"]],
                    padding=30
                )
                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                invoice_no_crop_path = f"{vlm_crop_debug_prefix}page{i+1}_invoice_no_llm_located.png"
                save_debug_image(invoice_no_crop, invoice_no_crop_path)
                print(f"[前綴補救][LLM定位+VLM] 發票號碼裁切圖片已儲存：{invoice_no_crop_path}，"
                      f"bbox=[{bbox_result['x1']}, {bbox_result['y1']}, {bbox_result['x2']}, {bbox_result['y2']}]，"
                      f"尺寸：{invoice_no_crop.size}")
                vlm_invoice_retry = extract_invoice_number_from_image(invoice_no_crop)
                vlm_invoice_conf_map = vlm_invoice_retry.get("__field_confidence__", {}) if vlm_invoice_retry else {}
                vlm_invoice_no = clean_invoice_no_candidate(vlm_invoice_retry.get("發票號碼") if vlm_invoice_retry else None)
            else:
                print(f"⚠️  [前綴補救][LLM定位] 無法取得發票號碼區域座標")

            if not vlm_invoice_no:
                print(f"⚠️  [前綴補救][VLM] 整張圖仍無法取得發票號碼，改裁切上方字軌區重試")
                page_w, page_h = page.size
                invoice_no_crop = crop_image_region(
                    page,
                    [0, 0, page_w, int(page_h * 0.25)],
                    padding=0
                )
                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                invoice_no_crop_path = f"{vlm_crop_debug_prefix}page{i+1}_invoice_no_top.png"
                save_debug_image(invoice_no_crop, invoice_no_crop_path)
                print(f"[前綴補救][VLM] 發票號碼裁切圖片已儲存：{invoice_no_crop_path}，尺寸：{invoice_no_crop.size}")

                vlm_invoice_retry = extract_invoice_number_from_image(invoice_no_crop)
                vlm_invoice_conf_map = vlm_invoice_retry.get("__field_confidence__", {}) if vlm_invoice_retry else {}
                vlm_invoice_no = clean_invoice_no_candidate(vlm_invoice_retry.get("發票號碼") if vlm_invoice_retry else None)

            if vlm_invoice_no:
                extracted_invoice_no = vlm_invoice_no
                lookup_invoice_no    = vlm_invoice_no
                extracted_data["發票號碼"] = vlm_invoice_no
                set_field_meta("發票號碼", vlm_invoice_no, "VLM", vlm_invoice_conf_map.get("發票號碼"))
                prefix_rule = get_validation_rules_by_prefix(
                    extracted_invoice_no,
                    page_text_clean,
                    page,
                    prefix_rules=_page_prefix_rules,
                )
                print(f"[前綴補救][VLM] 發票號碼更新為 {extracted_invoice_no}，前綴: [{prefix_rule['prefix']}] → 檢核項目: {prefix_rule['rules']}")
            else:
                print(f"⚠️  [前綴補救][VLM] 仍無法取得可用發票號碼")

        if prefix_rule["prefix"] is None:
            print(f"⚠️  LLM/VLM 補救後仍無法取得發票前綴，跳過此頁")
            _multi_invoice_fields = [
                "發票號碼", "買方統編", "買方公司名稱", "賣方統編", "賣方公司名稱",
                "營業稅稅別判斷", "未稅金額", "稅額", "合計金額", "金額大寫中文",
                "明細項目", "發票日期", "備註",
            ]
            print(f"\n第 {i+1} 頁 輸出OCR+LLM結果摘要:")
            compare_result = {}
            for _field in _multi_invoice_fields:
                print(f"  {_field}: {{'OCR結果': None, '信心值': None, '來源': None}}")
                compare_result[_field] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  折讓單日期: {{'OCR結果': None, '信心值': None}}")
            compare_result["折讓單日期"] = {"OCR結果": None, "信心值": None}
            print(f"  聯式: {{'OCR結果': None, '信心值': None, '來源': None}}")
            compare_result["聯式"] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  是否有多張發票: {{'OCR結果': False, '信心值': {repr(detection.get('confidence'))}}}")
            compare_result["是否有多張發票"] = {"OCR結果": False, "信心值": detection.get("confidence")}
            
            all_pages_result.append({
                "page":   i + 1,
                "status": "processed",
                "reason": "LLM/VLM補救後仍無法取得發票號碼前綴",
                "compare_result": compare_result
            })
            continue

        # ✅ 前綴找不到對應規則，轉人工審核
        if prefix_rule.get("unknown"):
            _multi_invoice_fields = [
                "發票號碼", "買方統編", "買方公司名稱", "賣方統編", "賣方公司名稱",
                "營業稅稅別判斷", "未稅金額", "稅額", "合計金額", "金額大寫中文",
                "明細項目", "發票日期", "備註",
            ]
            print(f"\n第 {i+1} 頁 輸出OCR+LLM結果摘要:")
            compare_result = {}
            for _field in _multi_invoice_fields:
                print(f"  {_field}: {{'OCR結果': None, '信心值': None, '來源': None}}")
                compare_result[_field] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  折讓單日期: {{'OCR結果': None, '信心值': None}}")
            compare_result["折讓單日期"] = {"OCR結果": None, "信心值": None}
            print(f"  聯式: {{'OCR結果': None, '信心值': None, '來源': None}}")
            compare_result["聯式"] = {"OCR結果": None, "信心值": None, "來源": None}
            print(f"  是否有多張發票: {{'OCR結果': False, '信心值': {repr(detection.get('confidence'))}}}")
            compare_result["是否有多張發票"] = {"OCR結果": False, "信心值": detection.get("confidence")}

            all_pages_result.append({
                "page":   i + 1,
                "status": "processed",
                "reason": f"LLM/VLM補救後發票前綴 [{prefix_rule['prefix']}] 仍無對應檢核規則",
                "compare_result": compare_result
            })
            continue

        # ---------------------------------
        # ✅ 3.6 LLM 彙整欄位
        # ---------------------------------
        print(f"\n===== 第 {i+1} 頁 LLM 欄位擷取 =====")
        # ocr_text_with_position = build_ocr_text_with_position(ocr_items)
        llm_fields = extract_invoice_fields_by_llm(ocr_text_with_position)
        llm_fields_conf_map = llm_fields.get("__field_confidence__", {}) if isinstance(llm_fields, dict) else {}
        # # ✅ 發票日期：從 llm_fields 取出（LLM 已一併擷取）
        # invoice_date = llm_fields.get("發票日期")
        # # 格式驗證：支援民國年份和西元日期
        # if invoice_date and not any(re.fullmatch(p, str(invoice_date).strip()) for p in _DATE_PATTERNS):
        #     print(f"⚠️  [格式驗證] 發票日期格式不符：'{invoice_date}'，清空並標記重辨識")
        #     invoice_date = None
        #     llm_fields["發票日期"] = None

        # ✅ 發票日期已由最前面的「LLM定位 + VLM」決定
        # 後續 LLM 結果禁止覆蓋 invoice_date

        # 若後面其他程式仍會從 llm_fields["發票日期"] 讀取，
        # 就把前面的正式結果同步回去
        llm_fields["發票日期"] = invoice_date

        # ✅ 備註：從 llm_fields 取出（LLM 已一併擷取）
        remark_text = llm_fields.get("備註")
        # 只有原文確實出現「備註」關鍵字，才代表這張發票理論上該有備註值
        _REMARK_KEYWORDS = ["備註", "備注"]
        remark_expected = any(kw in page_text_clean for kw in _REMARK_KEYWORDS)

        # LLM 初次擷取欄位來源與信心值
        for llm_field in ["年度期間", "金額大寫中文", "未稅金額", "稅額", "合計金額", "備註"]:  # 發票日期 #明細項目
            set_field_meta(llm_field, llm_fields.get(llm_field), "LLM", llm_fields_conf_map.get(llm_field))
        set_field_meta(
            "發票日期",
            invoice_date,
            "VLM",
            _date_vlm_conf_map.get("發票日期")
        )
        print(f"  發票日期:    {invoice_date}")
        print(f"  年度期間:    {llm_fields.get('年度期間')}")
        print(f"  金額大寫中文: {llm_fields.get('金額大寫中文')}")
        print(f"  未稅金額:    {llm_fields.get('未稅金額')}")
        print(f"  稅額:        {llm_fields.get('稅額')}")
        print(f"  合計金額:    {llm_fields.get('合計金額')}")
        print(f"  備註:       {remark_text}")
        # # 發票號碼找不到時，改用賣方統編從 Excel 反查
        # if standard is None and seller_tax_id:
        #     for inv_no, std in standard_dict.items():
        #         if std.get("廠商統編") == seller_tax_id:
        #             standard          = std
        #             lookup_invoice_no = inv_no
        #             print(f"⚠️  發票號碼由 Excel 反查得到：{inv_no}（依賣方統編 {seller_tax_id} 比對）")
        #             break

        # ✅ 定義哪些欄位屬於 OCR+Regex、哪些屬於 LLM
        # 「營業稅稅別判斷」也納入 VLM 重試欄位
        # ✅ 賣方公司名稱不納入 LLM重試/VLM保底，由 VLM 第3次補救專責處理
        OCR_REGEX_FIELDS = ["發票號碼", "買方統編", "賣方統編", "買方公司名稱"]
        LLM_FIELDS       = ["年度期間", "金額大寫中文", "未稅金額", "稅額", "合計金額"]  # ✅ 發票日期不納入 LLM重試/VLM保底  #"明細項目"
        TAX_FIELDS       = ["營業稅稅別判斷"]   # 稅別判斷獨立處理
        ALL_RETRY_FIELDS = OCR_REGEX_FIELDS + LLM_FIELDS + TAX_FIELDS

        def get_failed_fields(compare_result: dict, target_fields: list) -> list:
            """找出比對失敗且在 target_fields 內的欄位"""
            failed = []
            for k, v in compare_result.items():
                if k not in target_fields:
                    continue
                if isinstance(v, dict) and v.get("是否一致") == False:
                    failed.append(k)
            return failed

        def expand_failed_fields_for_amount_retry(failed_fields: list, active_rules: list) -> list:
            """
            若未稅金額/稅額/合計金額任一失敗，代表明細金額可能有誤，
            自動將「明細項目」加入重試欄位（含裁切截圖流程）。
            """
            expanded = list(failed_fields)
            amount_failed_fields = [f for f in ["未稅金額"] if f in expanded]
            if amount_failed_fields and "明細項目" in active_rules and "明細項目" not in expanded:
                expanded.append("明細項目")
                print(f"[重試欄位擴展] {amount_failed_fields}失敗，追加重試欄位：明細項目（重點重抓金額）")
            return expanded

        def expand_vlm_crop_bbox_for_fields(bbox_result: dict, fields: list, image_size: tuple[int, int]) -> list:
            """針對特定欄位微調 VLM 裁切範圍，避免欄位內容被切掉。"""
            page_w, page_h = image_size

            # 明細金額若要和未稅/稅額/合計一起重試，裁切區域需同時涵蓋明細區與下方金額區。
            # 但若同時包含賣方公司名稱，賣方區可能在頁面頂部，不可用硬編碼的中領區域。
            # if "明細項目" in fields \
            #         and any(f in fields for f in ["未稅金額", "稅額", "合計金額"]) \
            #         and "賣方公司名稱" not in fields:
            #     return [
            #         int(page_w * 0.10),
            #         int(page_h * 0.22),
            #         int(page_w * 0.82),
            #         int(page_h * 0.78),
            #     ]

            # if fields == ["賣方統編"]:
            #     # 賣方統編通常在下半部（營業人章/負責人/TEL/地址附近），
            #     # 強制用下半部 ROI，避免誤裁到上方買方統編區。
            #     return [
            #         int(page_w * 0.45),
            #         int(page_h * 0.48),
            #         page_w,
            #         page_h,
            #     ]

            x1 = int(bbox_result["x1"])
            y1 = int(bbox_result["y1"])
            x2 = int(bbox_result["x2"])
            y2 = int(bbox_result["y2"])

            # if fields == ["買方統編"]:
            #     # 統編 8 格通常由標籤右側水平延伸，定位容易只抓到左半邊。
            #     x1 -= 120
            #     y1 -= 120
            #     x2 += 900
            #     y2 += 180

            return [
                max(0, x1),
                max(0, y1),
                min(page_w, x2),
                min(page_h, y2),
            ]

        # ---------------------------------
        # ✅ 3.7 營業稅稅別判斷（應稅/零稅率/免稅）
        # ---------------------------------
        print(f"\n===== 第 {i+1} 頁 營業稅稅別判斷 =====")
        VALID_TAX_TYPES = ["應稅", "零稅率", "免稅"]
        tax_type = None
        tax_found = False
        tax_result = {"稅別": None, "已勾選": False, "判斷依據": "尚未判斷"}

        # print("[VLM稅別判斷] 使用整張圖片，只判斷營業稅稅別...")
        # vlm_tax_result = extract_fields_from_image_region(page, ["營業稅稅別判斷"])
        # vlm_tax_conf_map = vlm_tax_result.get("__field_confidence__", {}) if vlm_tax_result else {}
        # vlm_tax = str(vlm_tax_result.get("營業稅稅別判斷") or "").strip() if vlm_tax_result else ""
        vlm_tax_conf_map = {}
        vlm_tax = None

        if vlm_tax in VALID_TAX_TYPES:
            tax_type = vlm_tax
            tax_found = True
            set_field_meta("營業稅稅別判斷", tax_type, "VLM", vlm_tax_conf_map.get("營業稅稅別判斷"))
            tax_result = {
                "稅別": tax_type,
                "已勾選": True,
                "判斷依據": "VLM整張圖片直接判斷營業稅稅別"
            }
            print(f"  [VLM稅別判斷] 稅別: {tax_type}，已勾選: {tax_found}")
        else:
            # print(f"⚠️  [VLM稅別判斷] 無法判斷（回傳: {vlm_tax_result}），改用 LLM OCR 文字判斷")
            # tax_result = determine_tax_type_by_llm(ocr_text_with_position)
            # tax_type = tax_result.get("稅別")
            # tax_found = tax_result.get("已勾選", False)
            # set_field_meta("營業稅稅別判斷", tax_type, "LLM", tax_result.get("信心值"))

            # ✅ LLM 也找不到時：用 LLM 定位稅別區塊 → 裁切 → VLM 聚焦辨識
            if tax_type not in VALID_TAX_TYPES:
                print(f"⚠️  [稅別補救] LLM OCR 仍無法判斷，改用 LLM 定位稅別區塊後裁切 VLM 辨識")
                tax_bbox = locate_field_region_by_llm(ocr_text_with_position, ["營業稅稅別判斷"])
                if tax_bbox and all(k in tax_bbox for k in ["x1", "y1", "x2", "y2"]):
                    tax_crop = crop_image_region(
                        page,
                        [tax_bbox["x1"], tax_bbox["y1"], tax_bbox["x2"], tax_bbox["y2"]],
                        padding=30
                    )
                    vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                    tax_crop_path = f"{vlm_crop_debug_prefix}page{i+1}_tax_type_llm_located.png"
                    save_debug_image(tax_crop, tax_crop_path)
                    print(f"[稅別補救] 裁切圖片已儲存：{tax_crop_path}，"
                          f"bbox=[{tax_bbox['x1']}, {tax_bbox['y1']}, {tax_bbox['x2']}, {tax_bbox['y2']}]，"
                          f"尺寸：{tax_crop.size}")
                    vlm_tax_focused = extract_tax_type_from_image(tax_crop)
                    vlm_tax_focused_conf_map = vlm_tax_focused.get("__field_confidence__", {})
                    vlm_tax_focused_val = str(vlm_tax_focused.get("營業稅稅別判斷") or "").strip()
                    if vlm_tax_focused_val in VALID_TAX_TYPES:
                        tax_type = vlm_tax_focused_val
                        tax_found = True
                        tax_result = {"稅別": tax_type, "已勾選": True, "判斷依據": "LLM定位+VLM裁切聚焦判斷稅別"}
                        set_field_meta("營業稅稅別判斷", tax_type, "VLM", vlm_tax_focused_conf_map.get("營業稅稅別判斷"))
                        print(f"[稅別補救][LLM定位+VLM] 稅別: {tax_type}")
                    else:
                        print(f"⚠️  [稅別補救][LLM定位+VLM] 仍無法判斷（回傳: {vlm_tax_focused_val}）")
                else:
                    print(f"⚠️  [稅別補救] LLM 無法定位稅別區塊座標")

            # 補救後仍無法判斷，輸出空字串
            if tax_type not in VALID_TAX_TYPES:
                tax_type = ""

        print(f"  稅別: {tax_type}，已勾選: {tax_found}，依據: {tax_result.get('判斷依據')}")

        # ---------------------------------
        # ✅ 3.8 計算標準答案（未稅金額/稅額/合計金額）
        # ---------------------------------
        # 未稅金額標準答案：由 LLM 彙整後的明細金額加總計算
        std_sales_amount = None
        ocr_detail_items = llm_fields.get("明細項目", [])
        if ocr_detail_items:
            total_from_detail = 0
            valid_sum = True
            for detail_item in ocr_detail_items:
                amt_str = str(detail_item.get("金額") or "").replace(",", "").strip()

                # 空白金額不可當作 0，否則會把標準答案錯算成 0
                if not amt_str:
                    valid_sum = False
                    break

                try:
                    total_from_detail += int(float(amt_str))
                except (ValueError, TypeError):
                    valid_sum = False
                    break
            if valid_sum:
                std_sales_amount = total_from_detail
                print(f"  未稅金額標準答案（明細加總）: {std_sales_amount}")
            else:
                print(f"  ⚠️  明細金額含無法解析的值，無法計算標準答案")
        else:
            print(f"  ⚠️  無明細項目，無法計算未稅金額標準答案")

        # 稅額標準答案：對應稅別判斷
        # 應稅 → 未稅金額 × 5%；零稅率 / 免稅 → 0
        std_tax_amount = None
        if std_sales_amount is not None and tax_type is not None:
            if tax_type == "應稅":
                std_tax_amount = round(std_sales_amount * 0.05)
            else:
                std_tax_amount = 0
            print(f"  稅額標準答案（{tax_type}）: {std_tax_amount}")

        # 合計金額標準答案：未稅金額 + 稅額
        std_total_amount = None
        if std_sales_amount is not None and std_tax_amount is not None:
            std_total_amount = std_sales_amount + std_tax_amount
            print(f"  合計金額標準答案：{std_total_amount}")
        # ✅ 第一次比對
        compare_result = compare_with_standard(
            buyer_tax_id, seller_tax_id,
            buyer_company_name, seller_company_name,
            amount_validation, extracted_invoice_no,
            llm_fields,
            active_rules=prefix_rule["rules"],
            active_detail_fields=prefix_rule["detail_fields"],
            tax_type=tax_type,
            tax_found=tax_found,
            std_sales_amount=std_sales_amount,
            std_tax_amount=std_tax_amount,
            std_total_amount=std_total_amount,
            invoice_date=invoice_date,
            remark_text=remark_text
        )


        # ✅ LLM 重試（OCR+Regex 和 LLM 欄位都納入）
        MAX_FIELD_RETRY = 2
        _forced_retry_fields = set()   # 跨 iteration 強制重試的欄位（如 qty×price 不符）
        for retry_attempt in range(MAX_FIELD_RETRY):

            failed_fields = get_failed_fields(compare_result, ALL_RETRY_FIELDS)
            failed_fields = expand_failed_fields_for_amount_retry(failed_fields, prefix_rule["rules"])
            # 將上一輮強制標記的欄位帶入本輮重試
            for _ff in _forced_retry_fields:
                if _ff not in failed_fields:
                    failed_fields.append(_ff)
                    print(f"[LLM重試] 強制帶入上輮標記欄位：{_ff}")
            _forced_retry_fields.clear()
            if not failed_fields:
                print(f"✅ 所有欄位比對通過，不需重試")
                break

            print(f"\n[LLM重試] 第{retry_attempt+1}次，失敗欄位：{failed_fields}")

            # ✅ 「營業稅稅別判斷」失敗：優先用 VLM 判斷，LLM 作為備援
            if "營業稅稅別判斷" in failed_fields:
                print(f"[LLM重試] 重新判斷稅別...")
                vlm_tax_retry = extract_fields_from_image_region(page, ["營業稅稅別判斷"])
                vlm_tax_retry_conf_map = vlm_tax_retry.get("__field_confidence__", {}) if vlm_tax_retry else {}
                vlm_tax_retry_val = str(vlm_tax_retry.get("營業稅稅別判斷") or "").strip() if vlm_tax_retry else ""

                if vlm_tax_retry_val in VALID_TAX_TYPES:
                    tax_type = vlm_tax_retry_val
                    tax_found = True
                    set_field_meta("營業稅稅別判斷", tax_type, "VLM", vlm_tax_retry_conf_map.get("營業稅稅別判斷"))
                else:
                    tax_retry_result = determine_tax_type_by_llm(ocr_text_with_position)
                    tax_type  = tax_retry_result.get("稅別")
                    tax_found = tax_retry_result.get("已勾選", False)
                    set_field_meta("營業稅稅別判斷", tax_type, "LLM", tax_retry_result.get("信心值"))

                print(f"[LLM重試] 稅別重試結果：{tax_type}，已勾選={tax_found}")
                # 重新計算稅額/合計的標準答案
                if std_sales_amount is not None and tax_type is not None:
                    std_tax_amount   = round(std_sales_amount * 0.05) if tax_type == "應稅" else 0
                    std_total_amount = std_sales_amount + std_tax_amount
                    print(f"[LLM重試] 重新計算標準答案: 稅額={std_tax_amount}, 合計={std_total_amount}")
                # 從失敗欄位清單中排除，其餘欄位繼續走一般 LLM 重試流程
                other_failed_fields = [f for f in failed_fields if f != "營業稅稅別判斷"]
            else:
                other_failed_fields = failed_fields

            if other_failed_fields:
                retry_result = reextract_specific_fields(ocr_text_with_position, other_failed_fields)
                retry_conf_map = retry_result.get("__field_confidence__", {}) if retry_result else {}
                if not retry_result:
                    print(f"⚠️  [LLM重試] 回傳空值，放棄重試")
                    # 更新比對後直接 break
                else:
                    # ✅ 更新對應來源的變數
                    for field in other_failed_fields:
                        val = retry_result.get(field)
                        if val is None:
                            continue
                        print(f"[LLM重試] 更新欄位 [{field}]: → {val}")

                        if field == "發票號碼":
                            extracted_invoice_no = val
                            set_field_meta("發票號碼", val, "LLM", retry_conf_map.get("發票號碼"))
                        elif field == "買方統編":
                            buyer_tax_id = val
                            set_field_meta("買方統編", val, "LLM", retry_conf_map.get("買方統編"))
                        elif field == "賣方統編":
                            seller_tax_id = val
                            set_field_meta("賣方統編", val, "LLM", retry_conf_map.get("賣方統編"))
                        elif field == "買方公司名稱":
                            buyer_company_name = val
                            set_field_meta("買方公司名稱", val, "LLM", retry_conf_map.get("買方公司名稱"))
                        elif field == "賣方公司名稱":
                            val_str = str(val or "").strip()
                            if not val_str or val_str.lower() in ["null", "none", "未找到"]:
                                seller_company_name = None
                            else:
                                verdict = verify_company_existence_by_model(val_str)
                                if verdict.get("exists") is True:
                                    seller_company_name = val_str
                                    set_field_meta("賣方公司名稱", val_str, "LLM", retry_conf_map.get("賣方公司名稱"))
                                else:
                                    print(f"⚠️  [公司存在性檢查][LLM] '{val_str}' 不存在或不確定（{verdict}），視為 None")
                                    seller_company_name = None
                        elif field == "發票日期":
                            val_date_str = str(val).strip() if val else ""
                            if val_date_str and not any(re.fullmatch(p, val_date_str) for p in _DATE_PATTERNS):
                                print(f"⚠️  [格式驗證][LLM重試] 發票日期格式不符：'{val_date_str}'，不更新")
                            else:
                                invoice_date = val
                                set_field_meta("發票日期", val, "LLM", retry_conf_map.get("發票日期"))
                                print(f"[LLM重試] 發票日期更新：{invoice_date}")
                        elif field == "備註":
                            remark_text = val
                            set_field_meta("備註", val, "LLM", retry_conf_map.get("備註"))
                            print(f"[LLM重試] 備註更新：{remark_text}")
                        else:
                            llm_fields[field] = val
                            set_field_meta(field, val, "LLM", retry_conf_map.get(field))

            # 重新計算未稅/稅額/合計金額標準答案（如果明細金額更新了）
            llm_detail_qty_price_mismatch = False
            if "明細項目" in failed_fields:
                updated_detail_items = llm_fields.get("明細項目", [])
                if updated_detail_items:
                    total_from_detail = 0
                    valid_sum = True
                    for detail_item in updated_detail_items:
                        amt_str = str(detail_item.get("金額") or "").replace(",", "").strip()

                        # 空白金額不可當作 0，避免 LLM 重試用不完整明細覆蓋標準答案
                        if not amt_str:
                            valid_sum = False
                            break

                        try:
                            total_from_detail += int(float(amt_str))
                        except (ValueError, TypeError):
                            valid_sum = False
                            break
                    if valid_sum:
                        std_sales_amount = total_from_detail
                        std_tax_amount   = round(std_sales_amount * 0.05) if tax_type == "應稅" else 0
                        std_total_amount = std_sales_amount + std_tax_amount
                        print(f"[LLM重試] 重新計算標準答案: 未稅={std_sales_amount}, 稅額={std_tax_amount}, 合計={std_total_amount}")

            # ✅ 無條件驗算當前明細的 數量*單價 是否等於 金額
            # 不論明細項目是否在 failed_fields，只要有明細就驗算
            _current_detail_items = llm_fields.get("明細項目", [])
            for detail_item in (_current_detail_items or []):
                if not isinstance(detail_item, dict):
                    continue
                qty_str   = str(detail_item.get("數量") or "").replace(",", "").strip()
                price_str = str(detail_item.get("單價") or "").replace(",", "").strip()
                amt_str2  = str(detail_item.get("金額") or "").replace(",", "").strip()
                if not qty_str or not price_str or not amt_str2:
                    continue
                qty_num_match = re.match(r'[\d.]+', qty_str)
                if not qty_num_match:
                    print(f"⚠️  [LLM重試] 明細 數量無法解析數字部分: qty='{qty_str}'，標記重抓")
                    llm_detail_qty_price_mismatch = True
                    continue
                qty_num_str = qty_num_match.group()
                try:
                    if "/" in qty_num_str:
                        parts = qty_num_str.split("/", 1)
                        qty = float(parts[0]) / float(parts[1])
                    else:
                        qty = float(qty_num_str)
                    price = float(price_str)
                    amt   = float(amt_str2)
                    if abs(qty * price - amt) > 0.1:
                        print(f"⚠️  [LLM重試] 明細 數量*單價≠金額: {qty_num_str}×{price_str}={qty * price:.2f}，金額={amt}，標記重抓")
                        llm_detail_qty_price_mismatch = True
                except (ValueError, ZeroDivisionError):
                    print(f"⚠️  [LLM重試] 明細 數量/單價/金額 解析失敗: qty='{qty_str}', price='{price_str}', amt='{amt_str2}'，標記重抓")
                    llm_detail_qty_price_mismatch = True

            # 重新比對
            compare_result = compare_with_standard(
                buyer_tax_id, seller_tax_id,
                buyer_company_name, seller_company_name,
                amount_validation, extracted_invoice_no,
                llm_fields,
                active_rules=prefix_rule["rules"],
                active_detail_fields=prefix_rule["detail_fields"],
                tax_type=tax_type,
                tax_found=tax_found,
                std_sales_amount=std_sales_amount,
                std_tax_amount=std_tax_amount,
                std_total_amount=std_total_amount,
                invoice_date=invoice_date,
                remark_text=remark_text
            )

            still_failed = get_failed_fields(compare_result, ALL_RETRY_FIELDS)
            still_failed = expand_failed_fields_for_amount_retry(still_failed, prefix_rule["rules"])
            # 強制將數量*單價不符的明細項目加入重抓清單
            if llm_detail_qty_price_mismatch and "明細項目" not in still_failed:
                still_failed.append("明細項目")
                _forced_retry_fields.add("明細項目")  # 將明細項目登錄到下一輮強制重試
                print(f"[LLM重試] 明細項目 數量*單價 不一致，強制加入重抓清單")
            if still_failed:
                print(f"[LLM重試] 第{retry_attempt+1}次後仍失敗：{still_failed}")
            else:
                print(f"[LLM重試] 第{retry_attempt+1}次後全部通過 ✅")
                break

        # ✅ VLM 保底（LLM 重試後仍失敗）
        VLM_MAX_RETRY        = 2
        vlm_still_failed     = []   # 初始化，保證後續判斷安全
        _vlm2_seller_name_candidate = None  # VLM 第2次存在性檢查失敗時保留的候選値
        final_failed_fields = get_failed_fields(compare_result, ALL_RETRY_FIELDS)
        final_failed_fields = expand_failed_fields_for_amount_retry(final_failed_fields, prefix_rule["rules"])

        if final_failed_fields:
            # 稅別獨立重試，避免在多欄位 VLM 擷取時互相干擾
            if "營業稅稅別判斷" in final_failed_fields:
                print("\n[VLM稅別專用補救] 失敗欄位包含稅別，先單獨重試稅別...")
                vlm_tax_retry = extract_fields_from_image_region(page, ["營業稅稅別判斷"])
                vlm_tax_retry_conf_map = vlm_tax_retry.get("__field_confidence__", {}) if vlm_tax_retry else {}
                vlm_tax_retry_val = str(vlm_tax_retry.get("營業稅稅別判斷") or "").strip() if vlm_tax_retry else ""

                if vlm_tax_retry_val in VALID_TAX_TYPES:
                    tax_type = vlm_tax_retry_val
                    tax_found = True
                    set_field_meta("營業稅稅別判斷", tax_type, "VLM", vlm_tax_retry_conf_map.get("營業稅稅別判斷"))
                    if std_sales_amount is not None:
                        std_tax_amount   = round(std_sales_amount * 0.05) if tax_type == "應稅" else 0
                        std_total_amount = std_sales_amount + std_tax_amount
                    print(f"[VLM稅別專用補救] 稅別更新為 {tax_type}，標準稅額={std_tax_amount}，合計={std_total_amount}")
                else:
                    print(f"[VLM稅別專用補救] 仍無法判斷有效稅別（回傳：{vlm_tax_retry}）")

                compare_result = compare_with_standard(
                    buyer_tax_id, seller_tax_id,
                    buyer_company_name, seller_company_name,
                    amount_validation, extracted_invoice_no,
                    llm_fields,
                    active_rules=prefix_rule["rules"],
                    active_detail_fields=prefix_rule["detail_fields"],
                    tax_type=tax_type,
                    tax_found=tax_found,
                    std_sales_amount=std_sales_amount,
                    std_tax_amount=std_tax_amount,
                    std_total_amount=std_total_amount,
                    invoice_date=invoice_date,
                    remark_text=remark_text
                )
                final_failed_fields = get_failed_fields(compare_result, ALL_RETRY_FIELDS)
                final_failed_fields = expand_failed_fields_for_amount_retry(final_failed_fields, prefix_rule["rules"])

            print(f"\n[VLM保底] LLM重試{MAX_FIELD_RETRY}次仍失敗：{final_failed_fields}")
            print(f"[VLM保底] 改用 VLM 圖片理解，最多重試 {VLM_MAX_RETRY} 次...")

            vlm_current_fields = [f for f in final_failed_fields if f != "營業稅稅別判斷"]
            # vlm_current_fields.extend(["明細項目"])
            if not vlm_current_fields:
                print("[VLM保底] 無需多欄位 VLM 補救（僅稅別欄位已處理）")
            else:
                for vlm_attempt in range(1, VLM_MAX_RETRY + 1):
                    print(f"\n[VLM保底] 第{vlm_attempt}次，擷取欄位：{vlm_current_fields}")

                    # ✅ 第2次起：先讓 LLM 定位失敗欄位區域，裁切後再給 VLM
                    if vlm_attempt >= 2:
                        print(f"[VLM保底] 嘗試 LLM 區域定位裁切...")
                        bbox_result = locate_field_region_by_llm(ocr_text_with_position, vlm_current_fields)

                        if bbox_result and all(k in bbox_result for k in ["x1", "y1", "x2", "y2"]):
                            print(f"[VLM保底] LLM 定位結果：{bbox_result}（reason: {bbox_result.get('reason', '')}）")

                            crop_bbox = expand_vlm_crop_bbox_for_fields(
                                bbox_result,
                                vlm_current_fields,
                                page.size
                            )

                            cropped = crop_image_region(
                                page,
                                crop_bbox,
                                padding=200
                            )
                            vlm_input_image = cropped

                            # ✅ 儲存裁切圖片供 debug 確認
                            vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                            crop_save_path = f"{vlm_crop_debug_prefix}page{i+1}_attempt{vlm_attempt}_{'_'.join(vlm_current_fields)}.png"
                            save_debug_image(cropped, crop_save_path)
                            print(f"[VLM保底] 裁切圖片已儲存：{crop_save_path}，bbox={crop_bbox}，尺寸：{cropped.size}")

                        else:
                            print(f"[VLM保底] LLM 定位失敗，改用整張圖")
                            vlm_input_image = page
                    else:
                        # 第1次：整張圖
                        vlm_input_image = page

                    vlm_result = extract_fields_from_image_region(vlm_input_image, vlm_current_fields)
                    vlm_conf_map = vlm_result.get("__field_confidence__", {}) if vlm_result else {}

                    if not vlm_result:
                        print(f"⚠️  [VLM保底] 第{vlm_attempt}次回傳空值")
                        if vlm_attempt == VLM_MAX_RETRY:
                            validation_result["需人工審核欄位"] = vlm_current_fields
                        continue

                    # ✅ 更新對應欄位變數
                    vlm_detail_updated = False
                    for field in vlm_current_fields:
                        val = vlm_result.get(field)
                        if val is None:
                            continue
                        print(f"[VLM保底] 第{vlm_attempt}次 更新欄位 [{field}]: → {val}")

                        if field == "發票號碼":
                            extracted_invoice_no = val
                            set_field_meta("發票號碼", val, "VLM", vlm_conf_map.get("發票號碼"))
                        elif field == "買方統編":
                            buyer_tax_id = val
                            set_field_meta("買方統編", val, "VLM", vlm_conf_map.get("買方統編"))
                        elif field == "賣方統編":
                            seller_tax_id = val
                            set_field_meta("賣方統編", val, "VLM", vlm_conf_map.get("賣方統編"))
                        elif field == "買方公司名稱":
                            buyer_company_name = val
                            set_field_meta("買方公司名稱", val, "VLM", vlm_conf_map.get("買方公司名稱"))
                        # elif field == "賣方公司名稱":
                        #     seller_company_name = val
                        #     set_field_meta("賣方公司名稱", val, "VLM", vlm_conf_map.get("賣方公司名稱"))
                        elif field == "賣方公司名稱":
                            val_str = str(val).strip() if val is not None else ""

                            if val_str.lower() in ["", "null", "none", "未找到"]:
                                seller_company_name = None
                            elif vlm_attempt == 1:
                                verdict = verify_company_existence_by_model(val_str)
                                if verdict.get("exists") is True:
                                    seller_company_name = val_str
                                    set_field_meta("賣方公司名稱", val_str, "VLM", vlm_conf_map.get("賣方公司名稱"))
                                else:
                                    print(f"⚠️  [公司存在性檢查][VLM第1次] '{val_str}' 不存在或不確定（{verdict}），先視為失敗以觸發後續裁切重抓")
                                    seller_company_name = None
                            else:
                                seller_company_name = val_str
                                set_field_meta("賣方公司名稱", val_str, "VLM", vlm_conf_map.get("賣方公司名稱"))
                        elif field == "營業稅稅別判斷":
                            # ✅ VLM 回傳的稅別字串（"應稅"/"零稅率"/"免稅"/null）解析為 tax_type/tax_found
                            VALID_TAX_TYPES = ["應稅", "零稅率", "免稅"]
                            vlm_tax = str(val).strip() if val else None
                            if vlm_tax in VALID_TAX_TYPES:
                                tax_type  = vlm_tax
                                tax_found = True
                                set_field_meta("營業稅稅別判斷", tax_type, "VLM", vlm_conf_map.get("營業稅稅別判斷"))
                                print(f"[VLM保底] 稅別更新：{tax_type}（已勾選）")
                                # 重新計算稅額/合計標準答案
                                if std_sales_amount is not None:
                                    std_tax_amount   = round(std_sales_amount * 0.05) if tax_type == "應稅" else 0
                                    std_total_amount = std_sales_amount + std_tax_amount
                                    print(f"[VLM保底] 重新計算標準答案: 稅額={std_tax_amount}, 合計={std_total_amount}")
                            else:
                                tax_type  = None
                                tax_found = False
                                print(f"[VLM保底] 稅別仍未找到（回傳：{val}）")
                        elif field == "發票日期":
                            val_date_str = re.sub(r'\s+', '', str(val).strip()) if val else ""
                            if val_date_str and not any(re.fullmatch(p, val_date_str) for p in _DATE_PATTERNS):
                                print(f"⚠️  [格式驗證][VLM保底] 發票日期格式不符：'{val_date_str}'，不更新")
                            else:
                                invoice_date = val
                                set_field_meta("發票日期", val, "VLM", vlm_conf_map.get("發票日期"))
                                print(f"[VLM保底] 發票日期更新：{invoice_date}")
                        elif field == "備註":
                            remark_text = val
                            set_field_meta("備註", val, "VLM", vlm_conf_map.get("備註"))
                            print(f"[VLM保底] 備註更新：{remark_text}")
                        else:
                            # 明細項目：存入前清理數量欄位的單位後綴（如 "40.0 YD" → "40.0"）
                            if field == "明細項目" and isinstance(val, list):
                                for _item in val:
                                    if not isinstance(_item, dict):
                                        continue
                                    _qty_raw = str(_item.get("數量") or "").strip()
                                    if _qty_raw:
                                        _qty_clean_m = re.match(r'[\d./]+', _qty_raw.replace(",", ""))
                                        if _qty_clean_m:
                                            _qty_clean = _qty_clean_m.group()
                                            if _qty_clean != _qty_raw:
                                                print(f"[VLM保底] 明細 數量單位後綴清除: '{_qty_raw}' → '{_qty_clean}'")
                                                _item["數量"] = _qty_clean
                            llm_fields[field] = val
                            set_field_meta(field, val, "VLM", vlm_conf_map.get(field))
                            if field == "明細項目":
                                vlm_detail_updated = True
                                _detail_reason = vlm_result.get("detail_reason")
                                if _detail_reason:
                                    print(f"[VLM保底][明細項目 reason] {_detail_reason}")

                    # VLM 補到完整明細時，立刻用明細金額重算標準答案
                    detail_qty_price_mismatch = False
                    if vlm_detail_updated:
                        updated_detail_items = llm_fields.get("明細項目", [])
                        if updated_detail_items:
                            total_from_detail = 0
                            valid_sum = True
                            for detail_item in updated_detail_items:
                                amt_str = str(detail_item.get("金額") or "").replace(",", "").strip()

                                if not amt_str:
                                    valid_sum = False
                                    break

                                try:
                                    total_from_detail += int(float(amt_str))
                                except (ValueError, TypeError):
                                    valid_sum = False
                                    break

                            if valid_sum:
                                std_sales_amount = total_from_detail
                                std_tax_amount   = round(std_sales_amount * 0.05) if tax_type == "應稅" else 0
                                std_total_amount = std_sales_amount + std_tax_amount
                                print(f"[VLM保底] 重新計算標準答案: 未稅={std_sales_amount}, 稅額={std_tax_amount}, 合計={std_total_amount}")

                            # 驗證每個明細項目的 數量*單價 是否等於 金額
                            for detail_item in updated_detail_items:
                                qty_str   = str(detail_item.get("數量") or "").replace(",", "").strip()
                                price_str = str(detail_item.get("單價") or "").replace(",", "").strip()
                                amt_str2  = str(detail_item.get("金額") or "").replace(",", "").strip()
                                if not qty_str or not price_str or not amt_str2:
                                    continue
                                # 數量可能帶有單位後綴（如 "40.0 YD"、"40條"），只取前綴數字部分
                                qty_num_match = re.match(r'[\d.]+', qty_str)
                                if not qty_num_match:
                                    print(f"⚠️  [VLM保底] 明細 數量無法解析數字部分: qty='{qty_str}'，標記重抓")
                                    detail_qty_price_mismatch = True
                                    continue
                                qty_num_str = qty_num_match.group()
                                if qty_num_str != qty_str.split()[0]:
                                    print(f"⚠️  [VLM保底] 明細 數量含單位後綴: '{qty_str}' → 取數值 '{qty_num_str}'")
                                try:
                                    if "/" in qty_num_str:
                                        parts = qty_num_str.split("/", 1)
                                        qty = float(parts[0]) / float(parts[1])
                                    else:
                                        qty = float(qty_num_str)
                                    price = float(price_str)
                                    amt   = float(amt_str2)
                                    if abs(qty * price - amt) > 0.1:
                                        print(f"⚠️  [VLM保底] 明細 數量*單價≠金額: {qty_num_str}×{price_str}={qty * price:.2f}，金額={amt}，標記重抓")
                                        detail_qty_price_mismatch = True
                                except (ValueError, ZeroDivisionError):
                                    print(f"⚠️  [VLM保底] 明細 數量/單價/金額 解析失敗: qty='{qty_str}', price='{price_str}', amt='{amt_str2}'，標記重抓")
                                    detail_qty_price_mismatch = True

                            # # 若 VLM 對明細項目的信心值未達 1.0，也標記需要重抓
                            # detail_conf = vlm_conf_map.get("明細項目")
                            # if detail_conf is not None and float(detail_conf) < 1.0:
                            #     print(f"⚠️  [VLM保底] 明細項目 VLM信心值={detail_conf} < 1.0，標記重抓")
                            #     detail_qty_price_mismatch = True

                    # ✅ 重新比對，確認內容是否正確
                    compare_result = compare_with_standard(
                        buyer_tax_id, seller_tax_id,
                        buyer_company_name, seller_company_name,
                        amount_validation, extracted_invoice_no,
                        llm_fields,
                        active_rules=prefix_rule["rules"],
                        active_detail_fields=prefix_rule["detail_fields"],
                        tax_type=tax_type,
                        tax_found=tax_found,
                        std_sales_amount=std_sales_amount,
                        std_tax_amount=std_tax_amount,
                        std_total_amount=std_total_amount,
                        invoice_date=invoice_date,
                        remark_text=remark_text
                    )

                    vlm_still_failed = get_failed_fields(compare_result, ALL_RETRY_FIELDS)
                    vlm_still_failed = expand_failed_fields_for_amount_retry(vlm_still_failed, prefix_rule["rules"])

                    # 強制將數量*單價不符的明細項目加入重抓清單
                    if detail_qty_price_mismatch and "明細項目" not in vlm_still_failed:
                        vlm_still_failed.append("明細項目")
                        print(f"[VLM保底] 明細項目 數量*單價 不一致，強制加入重抓清單")

                    if not vlm_still_failed:
                        print(f"[VLM保底] 第{vlm_attempt}次比對通過 ✅")
                        break

                    print(f"[VLM保底] 第{vlm_attempt}次比對後仍失敗：{vlm_still_failed}")

                    if vlm_attempt == VLM_MAX_RETRY:
                        print(f"[VLM保底] 已達最大重試次數（{VLM_MAX_RETRY}次），轉人工審核")
                        validation_result["需人工審核欄位"] = vlm_still_failed
                    else:
                        # ✅ 下一輪只針對仍失敗的欄位
                        vlm_current_fields = vlm_still_failed
                        print(f"[VLM保底] 下一次只針對失敗欄位重試：{vlm_current_fields}")


        # ✅ VLM 第3次：賣方公司名稱與統一編號交叉驗證
        # 當兩者都有值時，請 LLM 確認是否對應同一家公司；若確認不符則清空名稱並重抓
        # 若賣方公司名稱無值，直接裁切重抓一次
        seller_company_name = None
        if not seller_company_name:
            print(f"[VLM第3次] 賣方公司名稱無值，嘗試 LLM 裁切後重新辨識...")
            bbox_result_3 = locate_field_region_by_llm(ocr_text_with_position, ["賣方公司名稱"])
            if bbox_result_3 and all(k in bbox_result_3 for k in ["x1", "y1", "x2", "y2"]):
                crop_bbox_3 = expand_vlm_crop_bbox_for_fields(bbox_result_3, ["賣方公司名稱"], page.size)
                cropped_3 = crop_image_region(page, crop_bbox_3, padding=100)
                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                crop_save_3 = f"{vlm_crop_debug_prefix}page{i+1}_attempt3_賣方公司名稱_rescue.png"
                save_debug_image(cropped_3, crop_save_3)
                print(f"[VLM第3次] 裁切圖片已儲存：{crop_save_3}，bbox={crop_bbox_3}")
                vlm_input_rescue = cropped_3
            else:
                print("[VLM第3次] LLM 定位失敗，改用整張圖")
                vlm_input_rescue = page

            vlm_result_rescue = extract_fields_from_image_region(vlm_input_rescue, ["賣方公司名稱"])
            vlm_conf_map_rescue = vlm_result_rescue.get("__field_confidence__", {}) if vlm_result_rescue else {}
            val_rescue = (vlm_result_rescue or {}).get("賣方公司名稱")
            val_str_rescue = str(val_rescue).strip() if val_rescue else ""

            if val_str_rescue and val_str_rescue.lower() not in ["", "null", "none", "未找到"]:
                seller_company_name = val_str_rescue
                set_field_meta("賣方公司名稱", val_str_rescue, "VLM", vlm_conf_map_rescue.get("賣方公司名稱"))
                print(f"[VLM第3次] 賣方公司名稱更新：{seller_company_name}")
            else:
                print("[VLM第3次] 仍無法取得賣方公司名稱，維持 None")

            # 重新比對
            compare_result = compare_with_standard(
                buyer_tax_id, seller_tax_id,
                buyer_company_name, seller_company_name,
                amount_validation, extracted_invoice_no,
                llm_fields,
                active_rules=prefix_rule["rules"],
                active_detail_fields=prefix_rule["detail_fields"],
                tax_type=tax_type,
                tax_found=tax_found,
                std_sales_amount=std_sales_amount,
                std_tax_amount=std_tax_amount,
                std_total_amount=std_total_amount,
                invoice_date=invoice_date,
                remark_text=remark_text
            )
    
        elif seller_company_name and seller_tax_id:
            seller_verify = verify_seller_name_matches_tax_id(seller_company_name, seller_tax_id)
            if seller_verify.get("match") is False or seller_verify.get("match") is None:
                print(f"[VLM第3次] 賣方公司名稱 '{seller_company_name}' 與統編 '{seller_tax_id}' 不符（{seller_verify.get('reason')}），清空後重抓")
                seller_company_name = None
                # 重新定位並裁切賣方公司名稱區域
                bbox_result_3 = locate_field_region_by_llm(ocr_text_with_position, ["賣方公司名稱"])
                if bbox_result_3 and all(k in bbox_result_3 for k in ["x1", "y1", "x2", "y2"]):
                    crop_bbox_3 = expand_vlm_crop_bbox_for_fields(bbox_result_3, ["賣方公司名稱"], page.size)
                    cropped_3 = crop_image_region(page, crop_bbox_3, padding=100)
                    vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                    crop_save_3 = f"{vlm_crop_debug_prefix}page{i+1}_attempt3_賣方公司名稱.png"
                    save_debug_image(cropped_3, crop_save_3)
                    print(f"[VLM第3次] 裁切圖片已儲存：{crop_save_3}，bbox={crop_bbox_3}")
                    vlm_input_3 = cropped_3
                else:
                    print("[VLM第3次] LLM 定位失敗，改用整張圖")
                    vlm_input_3 = page

                vlm_result_3 = extract_fields_from_image_region(vlm_input_3, ["賣方公司名稱"])
                vlm_conf_map_3 = vlm_result_3.get("__field_confidence__", {}) if vlm_result_3 else {}
                val_3 = (vlm_result_3 or {}).get("賣方公司名稱")
                val_str_3 = str(val_3).strip() if val_3 else ""

                if val_str_3 and val_str_3.lower() not in ["", "null", "none", "未找到"]:
                    seller_company_name = val_str_3
                    set_field_meta("賣方公司名稱", val_str_3, "VLM", vlm_conf_map_3.get("賣方公司名稱"))
                    print(f"[VLM第3次] 賣方公司名稱更新：{seller_company_name}")
                else:
                    print("[VLM第3次] 仍無法取得賣方公司名稱，維持 None")

                # 第3次後重新比對
                compare_result = compare_with_standard(
                    buyer_tax_id, seller_tax_id,
                    buyer_company_name, seller_company_name,
                    amount_validation, extracted_invoice_no,
                    llm_fields,
                    active_rules=prefix_rule["rules"],
                    active_detail_fields=prefix_rule["detail_fields"],
                    tax_type=tax_type,
                    tax_found=tax_found,
                    std_sales_amount=std_sales_amount,
                    std_tax_amount=std_tax_amount,
                    std_total_amount=std_total_amount,
                    invoice_date=invoice_date,
                    remark_text=remark_text
                )
            else:
                print(f"[VLM第3次] 賣方公司名稱與統編驗證通過（match={seller_verify.get('match')}），不需重抓")

        # ✅ VLM 第3次：賣方統一編號與公司名稱交叉驗證
        # 當兩者都有值時，請 LLM 確認是否對應同一家公司；若確認不符則將賣方統編列為失敗項目並重新裁切辨識
        if seller_company_name and seller_tax_id:
            seller_verify_taxid = verify_seller_name_matches_tax_id(seller_company_name, seller_tax_id)
            if seller_verify_taxid.get("match") is False or seller_verify_taxid.get("match") is None:
                print(f"[VLM第3次][賣方統編] 賣方統編 '{seller_tax_id}' 與公司名稱 '{seller_company_name}' 不符（{seller_verify_taxid.get('reason')}），列為失敗項目，重新裁切辨識...")
                seller_tax_id = None
                def find_seller_taxid_bbox_by_regex(ocr_text_with_position: str, page_height: int, exclude_ids=None) -> dict | None:
                    """
                    直接從 OCR 文字中掃描孤立的 8 位數字，排除已知買方統編，
                    優先回傳位於頁面下半部的候選座標（賣方統編通常位於底部章戳附近）
                    """
                    exclude_ids = exclude_ids or set()
                    pattern = re.compile(r'\[(\d+),(\d+),(\d+),(\d+)\](\d{8})(?!\d)')
                    candidates = []
                    for m in pattern.finditer(ocr_text_with_position):
                        x1, y1, x2, y2, num = m.groups()
                        if num in exclude_ids:
                            continue
                        y1, y2 = int(y1), int(y2)
                        if (y1 + y2) / 2 >= page_height * 0.5:
                            candidates.append({"x1": int(x1), "y1": y1, "x2": int(x2), "y2": y2, "num": num})
                    if not candidates:
                        return None
                    # 取最靠下方的候選
                    best = max(candidates, key=lambda c: c["y2"])
                    return {"x1": best["x1"], "y1": best["y1"], "x2": best["x2"], "y2": best["y2"],
                            "reason": f"regex直接命中賣方統編候選數字 {best['num']}"}
                bbox_seller_taxid = find_seller_taxid_bbox_by_regex(
                    ocr_text_with_position, page.size[1], exclude_ids={BUYER_TAX_ID_FIXED}
                )
                if not bbox_seller_taxid:
                    print("[VLM第3次][賣方統編] regex未命中，改用LLM定位")

                    # 重新定位並裁切賣方統編區域
                    bbox_seller_taxid = locate_field_region_by_llm(ocr_text_with_position, ["賣方統編"])
                if bbox_seller_taxid and all(k in bbox_seller_taxid for k in ["x1", "y1", "x2", "y2"]):
                    crop_bbox_seller_taxid = expand_vlm_crop_bbox_for_fields(bbox_seller_taxid, ["賣方統編"], page.size)
                    cropped_seller_taxid = crop_image_region(page, crop_bbox_seller_taxid, padding=0)
                    vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                    crop_save_seller_taxid = f"{vlm_crop_debug_prefix}page{i+1}_attempt3_賣方統編.png"
                    save_debug_image(cropped_seller_taxid, crop_save_seller_taxid)
                    print(f"[VLM第3次][賣方統編] 裁切圖片已儲存：{crop_save_seller_taxid}，bbox={crop_bbox_seller_taxid}")
                    vlm_input_seller_taxid = cropped_seller_taxid
                else:
                    print("[VLM第3次][賣方統編] LLM 定位失敗，改用整張圖")
                    vlm_input_seller_taxid = page

                vlm_result_seller_taxid = extract_fields_from_image_region(vlm_input_seller_taxid, ["賣方統編"])
                vlm_conf_seller_taxid = vlm_result_seller_taxid.get("__field_confidence__", {}) if vlm_result_seller_taxid else {}
                val_seller_taxid = (vlm_result_seller_taxid or {}).get("賣方統編")
                val_str_seller_taxid = re.sub(r"[^0-9]", "", str(val_seller_taxid).strip()) if val_seller_taxid else ""

                if val_str_seller_taxid:
                    seller_tax_id = val_str_seller_taxid
                    set_field_meta("賣方統編", val_str_seller_taxid, "VLM", vlm_conf_seller_taxid.get("賣方統編"))
                    print(f"[VLM第3次][賣方統編] 更新：{seller_tax_id}")
                else:
                    seller_tax_id = None
                    field_source.pop("賣方統編", None)
                    field_confidence.pop("賣方統編", None)
                    print("[VLM第3次][賣方統編] 辨識結果為空，維持原值 None")

                # 重新比對
                compare_result = compare_with_standard(
                    buyer_tax_id, seller_tax_id,
                    buyer_company_name, seller_company_name,
                    amount_validation, extracted_invoice_no,
                    llm_fields,
                    active_rules=prefix_rule["rules"],
                    active_detail_fields=prefix_rule["detail_fields"],
                    tax_type=tax_type,
                    tax_found=tax_found,
                    std_sales_amount=std_sales_amount,
                    std_tax_amount=std_tax_amount,
                    std_total_amount=std_total_amount,
                    invoice_date=invoice_date,
                    remark_text=remark_text
                )
            else:
                print(f"[VLM第3次][賣方統編] 賣方統編與公司名稱驗證通過（match={seller_verify_taxid.get('match')}），不需重抓")
        elif seller_tax_id is None or ( seller_company_name is None and seller_tax_id is not None):
            def find_seller_taxid_bbox_by_regex(ocr_text_with_position: str, page_height: int, exclude_ids=None) -> dict | None:
                """
                直接從 OCR 文字中掃描孤立的 8 位數字，排除已知買方統編，
                優先回傳位於頁面下半部的候選座標（賣方統編通常位於底部章戳附近）
                """
                exclude_ids = exclude_ids or set()
                pattern = re.compile(r'\[(\d+),(\d+),(\d+),(\d+)\](\d{8})(?!\d)')
                candidates = []
                for m in pattern.finditer(ocr_text_with_position):
                    x1, y1, x2, y2, num = m.groups()
                    if num in exclude_ids:
                        continue
                    y1, y2 = int(y1), int(y2)
                    if (y1 + y2) / 2 >= page_height * 0.5:
                        candidates.append({"x1": int(x1), "y1": y1, "x2": int(x2), "y2": y2, "num": num})
                if not candidates:
                    return None
                # 取最靠下方的候選
                best = max(candidates, key=lambda c: c["y2"])
                return {"x1": best["x1"], "y1": best["y1"], "x2": best["x2"], "y2": best["y2"],
                        "reason": f"regex直接命中賣方統編候選數字 {best['num']}"}
            bbox_seller_taxid = find_seller_taxid_bbox_by_regex(
                ocr_text_with_position, page.size[1], exclude_ids={BUYER_TAX_ID_FIXED}
            )
            if not bbox_seller_taxid:
                print("[VLM第3次][賣方統編] regex未命中，改用LLM定位")

                # 重新定位並裁切賣方統編區域
                bbox_seller_taxid = locate_field_region_by_llm(ocr_text_with_position, ["賣方統編"])
            if bbox_seller_taxid and all(k in bbox_seller_taxid for k in ["x1", "y1", "x2", "y2"]):
                crop_bbox_seller_taxid = expand_vlm_crop_bbox_for_fields(bbox_seller_taxid, ["賣方統編"], page.size)
                cropped_seller_taxid = crop_image_region(page, crop_bbox_seller_taxid, padding=0)
                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                crop_save_seller_taxid = f"{vlm_crop_debug_prefix}page{i+1}_attempt3_賣方統編.png"
                save_debug_image(cropped_seller_taxid, crop_save_seller_taxid)
                print(f"[VLM第3次][賣方統編] 裁切圖片已儲存：{crop_save_seller_taxid}，bbox={crop_bbox_seller_taxid}")
                vlm_input_seller_taxid = cropped_seller_taxid
            else:
                print("[VLM第3次][賣方統編] LLM 定位失敗，改用整張圖")
                vlm_input_seller_taxid = page

            vlm_result_seller_taxid = extract_fields_from_image_region(vlm_input_seller_taxid, ["賣方統編"])
            vlm_conf_seller_taxid = vlm_result_seller_taxid.get("__field_confidence__", {}) if vlm_result_seller_taxid else {}
            val_seller_taxid = (vlm_result_seller_taxid or {}).get("賣方統編")
            val_str_seller_taxid = re.sub(r"[^0-9]", "", str(val_seller_taxid).strip()) if val_seller_taxid else ""

            if val_str_seller_taxid:
                seller_tax_id = val_str_seller_taxid
                set_field_meta("賣方統編", val_str_seller_taxid, "VLM", vlm_conf_seller_taxid.get("賣方統編"))
                print(f"[VLM第3次][賣方統編] 更新：{seller_tax_id}")
            else:
                seller_tax_id = None
                field_source.pop("賣方統編", None)
                field_confidence.pop("賣方統編", None)
                print("[VLM第3次][賣方統編] 辨識結果為空，維持原值 None")

            # 重新比對
            compare_result = compare_with_standard(
                buyer_tax_id, seller_tax_id,
                buyer_company_name, seller_company_name,
                amount_validation, extracted_invoice_no,
                llm_fields,
                active_rules=prefix_rule["rules"],
                active_detail_fields=prefix_rule["detail_fields"],
                tax_type=tax_type,
                tax_found=tax_found,
                std_sales_amount=std_sales_amount,
                std_tax_amount=std_tax_amount,
                std_total_amount=std_total_amount,
                invoice_date=invoice_date,
                remark_text=remark_text
            )

        # ✅ VLM 第3次：買方統一編號補救
        # 若買方統編不等於固定值，透過 LLM 定位裁切後再做 VLM 辨識
        if buyer_tax_id != BUYER_TAX_ID_FIXED:
            print(f"[VLM第3次][買方統編] 目前值='{buyer_tax_id}'，不符 {BUYER_TAX_ID_FIXED}，嘗試裁切辨識...")
            bbox_buyer = locate_field_region_by_llm(ocr_text_with_position, ["買方統編"])
            if bbox_buyer and all(k in bbox_buyer for k in ["x1", "y1", "x2", "y2"]):
                crop_bbox_buyer = expand_vlm_crop_bbox_for_fields(bbox_buyer, ["買方統編"], page.size)
                cropped_buyer = crop_image_region(page, crop_bbox_buyer, padding=0)
                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                crop_save_buyer = f"{vlm_crop_debug_prefix}page{i+1}_attempt3_買方統編_rescue.png"
                save_debug_image(cropped_buyer, crop_save_buyer)
                print(f"[VLM第3次][買方統編] 裁切圖片已儲存：{crop_save_buyer}，bbox={crop_bbox_buyer}")
                vlm_input_buyer = cropped_buyer
            else:
                print("[VLM第3次][買方統編] LLM 定位失敗，改用整張圖")
                vlm_input_buyer = page

            vlm_result_buyer = extract_fields_from_image_region(vlm_input_buyer, ["買方統編"])
            vlm_conf_buyer = vlm_result_buyer.get("__field_confidence__", {}) if vlm_result_buyer else {}
            val_buyer = (vlm_result_buyer or {}).get("買方統編")
            val_str_buyer = re.sub(r"[^0-9]", "", str(val_buyer).strip()) if val_buyer else ""

            if val_str_buyer == BUYER_TAX_ID_FIXED:
                buyer_tax_id = val_str_buyer
                set_field_meta("買方統編", val_str_buyer, "VLM", vlm_conf_buyer.get("買方統編"))
                print(f"[VLM第3次][買方統編] 更新成功：{buyer_tax_id}")
            elif val_str_buyer:
                # 不符固定值，但仍更新為最新辨識結果（輸出顯示最新值，比對仍會失敗）
                buyer_tax_id = val_str_buyer
                set_field_meta("買方統編", val_str_buyer, "VLM", vlm_conf_buyer.get("買方統編"))
                print(f"[VLM第3次][買方統編] 辨識結果 '{val_str_buyer}' 仍不符 {BUYER_TAX_ID_FIXED}，更新為最新辨識值")
            else:
                print(f"[VLM第3次][買方統編] 辨識結果為空，維持原值")

            # 重新比對
            compare_result = compare_with_standard(
                buyer_tax_id, seller_tax_id,
                buyer_company_name, seller_company_name,
                amount_validation, extracted_invoice_no,
                llm_fields,
                active_rules=prefix_rule["rules"],
                active_detail_fields=prefix_rule["detail_fields"],
                tax_type=tax_type,
                tax_found=tax_found,
                std_sales_amount=std_sales_amount,
                std_tax_amount=std_tax_amount,
                std_total_amount=std_total_amount,
                invoice_date=invoice_date,
                remark_text=remark_text
            )

        # ✅ VLM 第3次：買方公司名稱補救
        # 若買方公司名稱比對失敗，透過 LLM 定位裁切後再做 VLM 辨識
        buyer_company_failed = (
            "買方公司名稱" in prefix_rule.get("rules", [])
            and not is_buyer_company_name_match(
                buyer_company_name
            )[0]
        )
        if buyer_company_failed:
            print(f"[VLM第3次][買方公司名稱] 目前值='{buyer_company_name}'，不符，嘗試裁切辨識...")
            bbox_buyer_name = locate_field_region_by_llm(ocr_text_with_position, ["買方公司名稱"])
            if bbox_buyer_name and all(k in bbox_buyer_name for k in ["x1", "y1", "x2", "y2"]):
                crop_bbox_buyer_name = expand_vlm_crop_bbox_for_fields(bbox_buyer_name, ["買方公司名稱"], page.size)
                cropped_buyer_name = crop_image_region(page, crop_bbox_buyer_name, padding=50)
                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)
                crop_save_buyer_name = f"{vlm_crop_debug_prefix}page{i+1}_attempt3_買方公司名稱_rescue.png"
                save_debug_image(cropped_buyer_name, crop_save_buyer_name)
                print(f"[VLM第3次][買方公司名稱] 裁切圖片已儲存：{crop_save_buyer_name}，bbox={crop_bbox_buyer_name}")
                vlm_input_buyer_name = cropped_buyer_name
            else:
                print("[VLM第3次][買方公司名稱] LLM 定位失敗，改用整張圖")
                vlm_input_buyer_name = page

            vlm_result_buyer_name = extract_fields_from_image_region(vlm_input_buyer_name, ["買方公司名稱"])
            vlm_conf_buyer_name = vlm_result_buyer_name.get("__field_confidence__", {}) if vlm_result_buyer_name else {}
            val_buyer_name = (vlm_result_buyer_name or {}).get("買方公司名稱")
            val_str_buyer_name = str(val_buyer_name).strip() if val_buyer_name else ""

            if val_str_buyer_name and val_str_buyer_name.lower() not in ["", "null", "none", "未找到"]:
                buyer_company_name = val_str_buyer_name
                set_field_meta("買方公司名稱", val_str_buyer_name, "VLM", vlm_conf_buyer_name.get("買方公司名稱"))
                # ✅ 改成允許多組買方公司名稱
                match_ok, match_method = is_buyer_company_name_match(
                    val_str_buyer_name
                )
                print(f"[VLM第3次][買方公司名稱] 更新：'{buyer_company_name}'，比對={'通過' if match_ok else '仍不符'}（{match_method}）")
            else:
                buyer_company_name = None
                field_source.pop("買方公司名稱", None)
                field_confidence.pop("買方公司名稱", None)
                print("[VLM第3次][買方公司名稱] 辨識結果為空，清空買方公司名稱")

            # 重新比對
            compare_result = compare_with_standard(
                buyer_tax_id, seller_tax_id,
                buyer_company_name, seller_company_name,
                amount_validation, extracted_invoice_no,
                llm_fields,
                active_rules=prefix_rule["rules"],
                active_detail_fields=prefix_rule["detail_fields"],
                tax_type=tax_type,
                tax_found=tax_found,
                std_sales_amount=std_sales_amount,
                std_tax_amount=std_tax_amount,
                std_total_amount=std_total_amount,
                invoice_date=invoice_date,
                remark_text=remark_text
            )

        # ============================================================
        # ✅ VLM 第3次：明細項目補救
        # ============================================================
        # 若 vlm_still_failed 包含「明細項目」，
        # 透過 LLM 定位明細區塊後裁切，再交給 VLM 辨識
        vlm_still_failed.append("明細項目")
        _DETAIL_FIELD = "明細項目"

        if _DETAIL_FIELD in vlm_still_failed:
            _detail_failed_fields = [_DETAIL_FIELD]

            # print(
            #     f"\n[VLM第3次][明細項目] "
            #     f"欄位 {_detail_failed_fields} 辨識失敗，"
            #     f"嘗試 LLM 定位後重新進行 VLM 辨識..."
            # )

            # bbox_detail = locate_field_region_by_llm(
            #     ocr_text_with_position,
            #     _detail_failed_fields
            # )

            # if (
            #     bbox_detail
            #     and all(k in bbox_detail for k in ["x1", "y1", "x2", "y2"])
            # ):
            #     crop_bbox_detail = expand_vlm_crop_bbox_for_fields(
            #         bbox_detail,
            #         _detail_failed_fields,
            #         page.size
            #     )

            #     cropped_detail = crop_image_region(
            #         page,
            #         crop_bbox_detail,
            #         padding=100
            #     )

            #     vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)

            #     crop_save_detail = (
            #         f"{vlm_crop_debug_prefix}"
            #         f"page{i+1}_attempt3_明細項目_rescue.png"
            #     )

            #     cropped_detail.save(crop_save_detail)

            #     print(
            #         f"[VLM第3次][明細項目] "
            #         f"裁切圖片已儲存：{crop_save_detail}，"
            #         f"bbox={crop_bbox_detail}"
            #     )

            #     vlm_input_detail = cropped_detail

            # else:
            #     print(
            #         "[VLM第3次][明細項目] "
            #         "LLM 定位失敗，改用整張圖"
            #     )

            #     vlm_input_detail = page
            vlm_input_detail = page
            vlm_result_detail = extract_fields_from_image_region(
                vlm_input_detail,
                _detail_failed_fields
            )

            vlm_conf_detail = (
                vlm_result_detail.get("__field_confidence__", {})
                if vlm_result_detail
                else {}
            )

            if vlm_result_detail:
                detail_items = vlm_result_detail.get(_DETAIL_FIELD)

                if isinstance(detail_items, list):
                    print(
                        f"[VLM第3次][明細項目] "
                        f"更新欄位 [{_DETAIL_FIELD}]: → {detail_items}"
                    )

                    # # ------------------------------------------------
                    # # 清理明細中的數量欄位
                    # # 例如：
                    # # 10個 → 10
                    # # 2PCS → 2
                    # # 1.5公斤 → 1.5
                    # # ------------------------------------------------
                    # for item in detail_items:
                    #     if not isinstance(item, dict):
                    #         continue

                    #     quantity_raw = str(
                    #         item.get("數量") or ""
                    #     ).strip()

                    #     if not quantity_raw:
                    #         continue

                    #     quantity_match = re.match(
                    #         r"[\d./]+",
                    #         quantity_raw.replace(",", "")
                    #     )

                    #     if (
                    #         quantity_match
                    #         and quantity_match.group() != quantity_raw
                    #     ):
                    #         cleaned_quantity = quantity_match.group()

                    #         print(
                    #             f"[VLM第3次][明細項目] "
                    #             f"數量後綴清除："
                    #             f"'{quantity_raw}' → '{cleaned_quantity}'"
                    #         )

                    #         item["數量"] = cleaned_quantity

                    # 更新辨識結果
                    llm_fields[_DETAIL_FIELD] = detail_items

                    _detail_reason_3 = vlm_result_detail.get("detail_reason")
                    if _detail_reason_3:
                        print(f"[VLM第3次][明細項目 reason] {_detail_reason_3}")

                    set_field_meta(
                        _DETAIL_FIELD,
                        detail_items,
                        "VLM",
                        vlm_conf_detail.get(_DETAIL_FIELD)
                    )

                    # ------------------------------------------------
                    # 根據明細項目的「金額」重新計算標準答案
                    # ------------------------------------------------
                    detail_total = 0
                    detail_amount_valid = True

                    for detail_item in detail_items:
                        if not isinstance(detail_item, dict):
                            detail_amount_valid = False
                            break

                        amount_raw = str(
                            detail_item.get("金額") or ""
                        ).replace(",", "").strip()

                        if not amount_raw:
                            detail_amount_valid = False
                            break

                        try:
                            detail_total += int(float(amount_raw))

                        except (ValueError, TypeError):
                            detail_amount_valid = False
                            break

                    if detail_amount_valid:
                        std_sales_amount = detail_total

                        if tax_type == "應稅":
                            std_tax_amount = round(
                                std_sales_amount * 0.05
                            )
                        else:
                            std_tax_amount = 0

                        std_total_amount = (
                            std_sales_amount + std_tax_amount
                        )

                        print(
                            "[VLM第3次][明細項目] "
                            "重新計算標準答案："
                            f"未稅={std_sales_amount}, "
                            f"稅額={std_tax_amount}, "
                            f"合計={std_total_amount}"
                        )

                    else:
                        print(
                            "[VLM第3次][明細項目] "
                            "明細金額不完整或格式錯誤，"
                            "不重新計算標準答案"
                        )

                    # ------------------------------------------------
                    # 重新比對
                    # ------------------------------------------------
                    compare_result = compare_with_standard(
                        buyer_tax_id,
                        seller_tax_id,
                        buyer_company_name,
                        seller_company_name,
                        amount_validation,
                        extracted_invoice_no,
                        llm_fields,
                        active_rules=prefix_rule["rules"],
                        active_detail_fields=prefix_rule["detail_fields"],
                        tax_type=tax_type,
                        tax_found=tax_found,
                        std_sales_amount=std_sales_amount,
                        std_tax_amount=std_tax_amount,
                        std_total_amount=std_total_amount,
                        invoice_date=invoice_date,
                        remark_text=remark_text
                    )

                else:
                    print(
                        "[VLM第3次][明細項目] "
                        "VLM 未回傳有效的明細項目 list，"
                        "維持原結果"
                    )

            else:
                print(
                    "[VLM第3次][明細項目] "
                    "VLM 回傳空值，維持原結果"
                )


        # ============================================================
        # ✅ VLM 第3次：金額欄位補救
        # ============================================================
        # 若 vlm_still_failed 包含：
        # 未稅金額、稅額、合計金額、金額大寫中文
        # 則透過 LLM 定位金額區塊後裁切，再交給 VLM 辨識
        # vlm_still_failed.extend([
        #     '未稅金額',
        #     '稅額',
        #     '合計金額',
        #     '金額大寫中文'
        # ])
        _AMOUNT_FIELDS = [
            "未稅金額",
            "稅額",
            "合計金額",
            "金額大寫中文"
        ]

        _amount_failed_fields = [
            field_name
            for field_name in _AMOUNT_FIELDS
            if field_name in vlm_still_failed
        ]

        if _amount_failed_fields:
            _amount_failed_fields = _AMOUNT_FIELDS.copy()

            # 裁切金額欄位時一併涵蓋明細項目表格，讓 VLM 能看到數量/單價/金額，交叉核對金額是否合理
            # 注意：這裡只是把「明細項目」加進定位/裁切的範圍，不會要求 VLM 重新輸出明細項目本身
            _amount_locate_fields = _amount_failed_fields + ["明細項目"]

            print(
                f"\n[VLM第3次][金額欄位] "
                f"欄位 {_amount_failed_fields} 辨識失敗，"
                f"嘗試 LLM 定位後重新進行 VLM 辨識..."
            )

            # bbox_amount = locate_field_region_by_llm(
            #     ocr_text_with_position,
            #     _amount_locate_fields
            # )

            # if (
            #     bbox_amount
            #     and all(k in bbox_amount for k in ["x1", "y1", "x2", "y2"])
            # ):
            #     crop_bbox_amount = expand_vlm_crop_bbox_for_fields(
            #         bbox_amount,
            #         _amount_failed_fields,
            #         page.size
            #     )

            #     cropped_amount = crop_image_region(
            #         page,
            #         crop_bbox_amount,
            #         padding=100
            #     )

            #     vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)

            #     crop_save_amount = (
            #         f"{vlm_crop_debug_prefix}"
            #         f"page{i+1}_attempt3_金額欄位_rescue.png"
            #     )

            #     cropped_amount.save(crop_save_amount)

            #     print(
            #         f"[VLM第3次][金額欄位] "
            #         f"裁切圖片已儲存：{crop_save_amount}，"
            #         f"bbox={crop_bbox_amount}"
            #     )

            #     vlm_input_amount = cropped_amount

            # else:
            #     print(
            #         "[VLM第3次][金額欄位] "
            #         "LLM 定位失敗，改用整張圖"
            #     )

            #     vlm_input_amount = page

             # ✅ 先清除舊值
            for field_name in _amount_failed_fields:
                llm_fields[field_name] = None
                field_source.pop(field_name, None)
                field_confidence.pop(field_name, None)

            print(
                f"\n[VLM第3次][金額欄位] "
                f"欄位 {_amount_failed_fields} 辨識失敗，"
                f"改用整張圖重新進行 VLM 辨識..."
            )

            vlm_input_amount = page

            vlm_result_amount = extract_fields_from_image_region(
                vlm_input_amount,
                _amount_failed_fields
            )

            vlm_conf_amount = (
                vlm_result_amount.get("__field_confidence__", {})
                if vlm_result_amount
                else {}
            )

            if vlm_result_amount:
                amount_field_updated = False

                for field_name in _amount_failed_fields:
                    field_value = vlm_result_amount.get(field_name)

                    if field_value is None:
                        print(
                            f"[VLM第3次][金額欄位] "
                            f"欄位 [{field_name}] 未取得值，"
                            f"維持原結果"
                        )

                        continue

                    # 清除數字欄位中的逗號及前後空白
                    if isinstance(field_value, str):
                        field_value = (
                            field_value
                            .replace(",", "")
                            .strip()
                        )

                    print(
                        f"[VLM第3次][金額欄位] "
                        f"更新欄位 [{field_name}]: → {field_value}"
                    )

                    llm_fields[field_name] = field_value

                    set_field_meta(
                        field_name,
                        field_value,
                        "VLM",
                        vlm_conf_amount.get(field_name)
                    )

                    amount_field_updated = True

                # 至少成功更新一個欄位才重新比對
                if amount_field_updated:
                    compare_result = compare_with_standard(
                        buyer_tax_id,
                        seller_tax_id,
                        buyer_company_name,
                        seller_company_name,
                        amount_validation,
                        extracted_invoice_no,
                        llm_fields,
                        active_rules=prefix_rule["rules"],
                        active_detail_fields=prefix_rule["detail_fields"],
                        tax_type=tax_type,
                        tax_found=tax_found,
                        std_sales_amount=std_sales_amount,
                        std_tax_amount=std_tax_amount,
                        std_total_amount=std_total_amount,
                        invoice_date=invoice_date,
                        remark_text=remark_text
                    )

                else:
                    print(
                        "[VLM第3次][金額欄位] "
                        "沒有任何欄位成功更新，"
                        "不重新執行比對"
                    )

                    compare_result = compare_with_standard(
                        buyer_tax_id,
                        seller_tax_id,
                        buyer_company_name,
                        seller_company_name,
                        amount_validation,
                        extracted_invoice_no,
                        llm_fields,
                        active_rules=prefix_rule["rules"],
                        active_detail_fields=prefix_rule["detail_fields"],
                        tax_type=tax_type,
                        tax_found=tax_found,
                        std_sales_amount=std_sales_amount,
                        std_tax_amount=std_tax_amount,
                        std_total_amount=std_total_amount,
                        invoice_date=invoice_date,
                        remark_text=remark_text
                    )

            else:
                print(
                    "[VLM第3次][金額欄位] "
                    "VLM 回傳空值，維持原結果"
                )
        # ============================================================
        # ✅ VLM 第3次：備註補救
        # ============================================================
        # 若 vlm_still_failed 包含「備註」，
        # 透過 LLM 定位備註區塊後裁切，再交給 VLM 辨識
        _REMARK_FIELD = "備註"
        vlm_still_failed.append(_REMARK_FIELD)
        if _REMARK_FIELD in vlm_still_failed:
            _remark_failed_fields = [_REMARK_FIELD]

            print(
                f"\n[VLM第3次][備註] "
                f"欄位 {_remark_failed_fields} 辨識失敗，"
                f"嘗試 LLM 定位後重新進行 VLM 辨識..."
            )

            bbox_remark = locate_field_region_by_llm(
                ocr_text_with_position,
                _remark_failed_fields
            )

            if (
                bbox_remark
                and all(k in bbox_remark for k in ["x1", "y1", "x2", "y2"])
            ):
                crop_bbox_remark = expand_vlm_crop_bbox_for_fields(
                    bbox_remark,
                    _remark_failed_fields,
                    page.size
                )

                cropped_remark = crop_image_region(
                    page,
                    crop_bbox_remark,
                    padding=100
                )

                vlm_crop_debug_dir.mkdir(parents=True, exist_ok=True)

                crop_save_remark = (
                    f"{vlm_crop_debug_prefix}"
                    f"page{i+1}_attempt3_備註_rescue.png"
                )


                save_debug_image(cropped_remark, crop_save_remark)

                print(
                    f"[VLM第3次][備註] "
                    f"裁切圖片已儲存：{crop_save_remark}，"
                    f"bbox={crop_bbox_remark}"
                )

                vlm_input_remark = cropped_remark

            else:
                print(
                    "[VLM第3次][備註] "
                    "LLM 定位失敗，改用整張圖"
                )

                vlm_input_remark = page

            vlm_result_remark = extract_fields_from_image_region(
                vlm_input_remark,
                _remark_failed_fields
            )

            vlm_conf_remark = (
                vlm_result_remark.get("__field_confidence__", {})
                if vlm_result_remark
                else {}
            )


            remark_val = vlm_result_remark.get(_REMARK_FIELD)

            if remark_val is not None:
                remark_val = str(remark_val).strip()

                # 統一換行格式
                remark_val = (
                    remark_val
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )

                # 若模型真的回傳字面上的 "\\n"，也一併處理
                remark_val = remark_val.replace("\\n", "\n")

                # 所有換行改成 ;
                remark_val = re.sub(
                    r"[ \t]*\n+[ \t]*",
                    ";",
                    remark_val
                )

                # 避免連續 ;;
                remark_val = re.sub(
                    r";{2,}",
                    ";",
                    remark_val
                )

                remark_val = remark_val.strip(" ;")

            print(
                f"[VLM第3次][備註] "
                f"更新欄位 [{_REMARK_FIELD}]: → {remark_val}"
            )


            remark_text = remark_val
            llm_fields[_REMARK_FIELD] = remark_val

            set_field_meta(
                _REMARK_FIELD,
                remark_val,
                "VLM",
                vlm_conf_remark.get(_REMARK_FIELD)
            )

            # 重新比對
            compare_result = compare_with_standard(
                buyer_tax_id,
                seller_tax_id,
                buyer_company_name,
                seller_company_name,
                amount_validation,
                extracted_invoice_no,
                llm_fields,
                active_rules=prefix_rule["rules"],
                active_detail_fields=prefix_rule["detail_fields"],
                tax_type=tax_type,
                tax_found=tax_found,
                std_sales_amount=std_sales_amount,
                std_tax_amount=std_tax_amount,
                std_total_amount=std_total_amount,
                invoice_date=invoice_date,
                remark_text=remark_text
            )
        validation_result["與Excel比對結果"] = compare_result

        # ✅ 發票日期單獨記錄（只要有值就好，不需比對標準答案）
        validation_result["發票日期"] = invoice_date
        validation_result["備註"] = remark_text
        # ---------------------------------
        # 5. 顯示結果
        # ---------------------------------
        # print(f"第 {i+1} 頁擷取結果:")
        # for k, v in extracted_data.items():
        #     print(f"  {k}: {v}")

        # print(f"\n第 {i+1} 頁檢核結果:")
        # for k, v in validation_result.items():
        #     if k != "金額檢核":  # 金額檢核細節太多，不在此印出
        #         print(f"  {k}: {v}")

        # print(f"\n第 {i+1} 頁 與標準答案比對:")
        # for k, v in compare_result.items():
        #     if k == "明細項目" and isinstance(v, dict):
        #         print(f"  明細項目 是否一致: {v.get('是否一致')}")
        #         # ✅ 印出每個欄位摘要
        #         for field, summary in v.get("欄位摘要", {}).items():
        #             print(f"    {field}: {summary}")
        #         # ✅ 印出失敗項目
        #         for f in v.get("失敗項目摘要", []):
        #             print(f"    ❌ 第{f['第幾筆']}筆 [{f['品名']}] 失敗欄位：{f['失敗欄位']}")
        #     else:
        #         print(f"  {k}: {v}")

        print(f"\n第 {i+1} 頁 輸出OCR+LLM結果摘要:")
        for k, v in compare_result.items():
            if k == "全部比對通過":
                continue

            if k == "明細項目" and isinstance(v, dict):
                detail_ocr_summary = {
                    field: summary.get("OCR結果")
                    for field, summary in v.get("欄位摘要", {}).items()
                    if isinstance(summary, dict)
                }
                conf = format_confidence(field_confidence.get(k))
                src = field_source.get(k, "未知")
                print(f"  {k}: {{'OCR結果': {detail_ocr_summary}, '信心值': '{conf}', '來源': '{src}'}}")
                # print(f"  {k}: {{'OCR結果': {detail_ocr_summary}, '信心值': '{conf}'}}")
            elif isinstance(v, dict):
                if "OCR結果" in v:
                    conf = format_confidence(field_confidence.get(k))
                    src = field_source.get(k, "未知")
                    print(f"  {k}: {{'OCR結果': {repr(v.get('OCR結果'))}, '信心值': '{conf}', '來源': '{src}'}}")
                    # print(f"  {k}: {{'OCR結果': {repr(v.get('OCR結果'))}, '信心值': '{conf}'}}")
                elif "結果" in v:
                    conf = format_confidence(field_confidence.get(k))
                    src = field_source.get(k, "未知")
                    print(f"  {k}: {{'OCR結果': {repr(v.get('結果'))}, '信心值': '{conf}', '來源': '{src}'}}")
                    # print(f"  {k}: {{'OCR結果': {repr(v.get('結果'))}, '信心值': '{conf}'}}")
        if "備註" not in compare_result:
            # 若這個前綴群組沒把「備註」列入 active_rules，仍在此印出目前擷取值
            print(f"  備註: {{'OCR結果': {repr(remark_text)}, '信心值': 'N/A'}}")
        print(f"  折讓單日期: {{'OCR結果': None, '信心值': None}}")

        multi_inv_result = detection.get("has_multiple_invoices", False)
        multi_inv_conf   = detection.get("confidence")
        print(f"  聯式: {{'OCR結果': {repr(prefix_rule.get('聯式'))}, '信心值': '{format_confidence(field_confidence.get("發票號碼"))}', '來源': '{field_source.get("發票號碼", "未知")}'}}")
        print(f"  是否有多張發票: {{'OCR結果': {multi_inv_result}, '信心值': {repr(multi_inv_conf)}}}")

        print(f"\n第 {i+1} 頁 TSR cells 數量: {len(cells)}")
        print("-" * 50)

        # ---------------------------------
        # 6. 視覺化輸出
        # ---------------------------------
        vis_img = img.copy()
        vis_img = draw_ocr_boxes(vis_img, ocr_items)   # 畫 OCR 框（綠）
        vis_img = draw_cells(vis_img, aligned_cells)    # 畫 Cell 框（藍）

        # 左上角摘要資訊
        summary_lines = [
            f"Page: {i+1}",
            f"發票號碼: {extracted_invoice_no or '未找到'}",
            f"買方統編: {buyer_tax_id or '未找到'} ({'✓' if validation_result['買方統編固定值檢核'] else '✗'})",
            f"賣方統編: {seller_tax_id or '未找到'}",
            f"統一編號格式正確: {validation_result['統一編號格式正確']}",
            f"Excel比對全部通過: {compare_result.get('全部比對通過', 'N/A')}",
            f"OCR文字框數: {len(ocr_items)}",
            f"TSR cells數: {len(cells)}"
        ]

        final_img = draw_texts_with_pil(
            vis_img,
            ocr_items,
            summary_lines,
            aligned_cells=aligned_cells
        )

        # 儲存視覺化結果
        save_path = f"{output_prefix}page_{i+1}_{PDF_NAME}.jpg"
        save_or_upload(encode_jpg_bytes(final_img), save_path, content_type="image/jpeg")
        print(f"輸出: {save_path}")

        # 儲存水平線與垂直線遮罩（Debug 用）
        # cv2.imwrite(f"{output_prefix}page_{i+1}_{PDF_NAME}_horizontal.jpg", horizontal_img)
        # cv2.imwrite(f"{output_prefix}page_{i+1}_{PDF_NAME}_vertical.jpg",   vertical_img)
        # cv2_imwrite_unicode(f"{output_prefix}page_{i+1}_{PDF_NAME}_horizontal.jpg", horizontal_img)
        # cv2_imwrite_unicode(f"{output_prefix}page_{i+1}_{PDF_NAME}_vertical.jpg",   vertical_img)


        api_extra_fields = {
            "折讓單日期": {
                "OCR結果": None,
                "信心值": None,
            },
            "聯式": {
                "OCR結果": prefix_rule.get("聯式"),
                "信心值": format_confidence(
                    field_confidence.get("發票號碼")
                ),
                "來源": field_source.get("發票號碼", "未知"),
            },
            "是否有多張發票": {
                "OCR結果": multi_inv_result,
                "信心值": multi_inv_conf,
            },
        }

        # ---------------------------------
        # 7. JSON 輸出（單頁）
        # ---------------------------------
        page_result = {
            "page":           i + 1,
            "status":         "processed",
            "ocr_text":       page_text,
            "ocr_text_clean": page_text_clean,
            "ocr_items":      ocr_items,
            "extracted_data": extracted_data,
            "validation_result": validation_result,
            "compare_result": compare_result,
            "tsr_result":     tsr_result,
            "aligned_cells":  aligned_cells,
            "table_data":     table_data,
            "field_confidence": field_confidence,
            "field_source":     field_source,

            # 新增
            "api_extra_fields": api_extra_fields,
        }

        all_pages_result.append(page_result)

        page_json_path = f"{json_output_prefix}page_{i+1}.json"
        page_json_bytes = json.dumps(page_result, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        save_or_upload(page_json_bytes, page_json_path, content_type="application/json")


    all_pages_json_path = str(json_output_dir / "all_pages_result.json")
    all_pages_json_bytes = json.dumps(all_pages_result, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    save_or_upload(all_pages_json_bytes, all_pages_json_path, content_type="application/json")

    return all_pages_result


# =========================
# API output transformer
# =========================
def build_api_compare_result(compare_result: dict, field_confidence: dict,extra_fields: dict | None = None,) -> dict:
    """將完整流程 compare_result 轉成既有 API 的精簡輸出格式。"""
    output = {}
    for field, data in compare_result.items():
        if field == "全部比對通過" or not isinstance(data, dict):
            continue

        if field == "明細項目":
            field_summary = data.get("欄位摘要", {})
            ocr_val: Any = {}
            for detail_field, summary in field_summary.items():
                if not isinstance(summary, dict):
                    continue
                values = summary.get("OCR結果")
                ocr_val[detail_field] = values if isinstance(values, list) else str(values or "")

            if not ocr_val:
                detail_list = data.get("比對細節", [])
                check_fields = data.get("檢核欄位") or []
                rows = []
                for row in detail_list:
                    if not isinstance(row, dict):
                        continue
                    rows.append({
                        key: (row.get(key, {}) or {}).get("OCR結果", "")
                        for key in check_fields
                    })
                ocr_val = rows
        elif field == "營業稅稅別判斷":
            ocr_val = data.get("結果") or data.get("OCR結果", "")
        else:
            ocr_val = data.get("OCR結果", "")

        output[field] = {
            "OCR結果": ocr_val,
            "信心值": format_confidence(field_confidence.get(field)),
        }
    # if isinstance(extra_fields, dict):
    #     for field, data in extra_fields.items():
    #         if isinstance(data, dict):
    #             output[field] = data
    if isinstance(extra_fields, dict):
        for field, data in extra_fields.items():
            if not isinstance(data, dict):
                continue

            extra_ocr_val = data.get("OCR結果")
            raw_confidence = data.get("信心值")
            source = data.get("來源")

            # 數字信心值統一轉成小數點兩位
            # high / medium / low 等文字則保留原值
            if raw_confidence is None:
                confidence_output = None
            else:
                try:
                    confidence_output = f"{float(raw_confidence):.2f}"
                except (TypeError, ValueError):
                    confidence_output = raw_confidence

            output[field] = {
                "OCR結果": extra_ocr_val,
                "信心值": confidence_output,
            }

            # # 有來源才輸出來源
            # if source is not None:
            #     output[field]["來源"] = source
    
    return output
