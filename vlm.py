import os
import base64
import json
import re
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from io import BytesIO
from json_repair import repair_json

from pathlib import Path

from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

# =========================
# VLM 設定
# =========================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

VLLM_LLM_MODEL2 = os.environ["VLLM_LLM_MODEL2"]
VLLM_LLM_MODEL3 = os.environ["VLLM_LLM_MODEL3"]
VLLM_LLM_MODEL4 = os.environ["VLLM_LLM_MODEL4"]
VLLM_LLM_MODEL5 = os.environ["VLLM_LLM_MODEL5"]
VLLM_LLM_API_BASE2 = os.environ["VLLM_LLM_API_BASE2"]

POPPLER_PATH = os.getenv("POPPLER_PATH") or None

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

client = OpenAI(
    # api_key="sk-abc123DEF456ghi789JKL012mno345PQR678stu901VWX234yz",        # 本地部署不需要真實 key
    api_key=os.environ["VLLM_API_KEY"],
    base_url=VLLM_LLM_API_BASE2
)


def _to_confidence_01(value):
    if value is None:
        return None
    try:
        v = float(value)
        if 0.0 <= v <= 1.0:
            return round(v, 4)
    except (TypeError, ValueError):
        return None
    return None


def _normalize_field_confidence_map(raw_map, allowed_fields: list) -> dict:
    if not isinstance(raw_map, dict):
        return {}
    cleaned = {}
    for f in allowed_fields:
        if f in raw_map:
            conf = _to_confidence_01(raw_map.get(f))
            if conf is not None:
                cleaned[f] = conf
    return cleaned

# =========================
# 工具函式
# =========================
def crop_image_region(image_pil: Image.Image, bbox: list, padding: int = 30) -> Image.Image:
    """裁切圖片指定區域，加 padding 避免切太緊"""
    w, h    = image_pil.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image_pil.crop((x1, y1, x2, y2))



def image_to_base64(image_pil: Image.Image) -> str:
    """PIL Image 轉 base64 字串"""
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =========================
# 公司存在性（用模型做保守判斷）
# =========================

_company_existence_cache: dict[str, dict] = {}

def _normalize_company_name_for_check(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"\s+", "", str(name))
    s = s.replace("臺", "台")
    return s

# from urllib.parse import urlparse
# def verify_company_existence_by_model(company_name: str) -> dict:
#     """
#     用 Tavily 搜尋公司名稱是否存在。

#     判斷規則：
#     - True  : 政府 / 經濟部 / 商工登記等可信來源中，有完整相符公司名稱
#     - None  : 有搜尋到完整公司名稱，但來源不是可信政府/登記來源
#     - False : 搜尋結果中完全沒有完整相符公司名稱，或名稱明顯不像公司名

#     回傳:
#     {
#         "exists": True/False/None,
#         "reason": "...",
#         "source": "...",
#         "matched_url": "...",
#         "matched_title": "..."
#     }
#     """
#     name = (company_name or "").strip()
#     if not name:
#         return {"exists": None, "reason": "empty"}

#     key = _normalize_company_name_for_check(name)
#     cache_key = f"tavily:{key}"

#     if cache_key in _company_existence_cache:
#         return _company_existence_cache[cache_key]

#     trusted_domains = {
#         "findbiz.nat.gov.tw",
#         "data.gcis.nat.gov.tw",
#         "gcis.nat.gov.tw",
#     }

#     supporting_domains = {
#         "104.com.tw",
#         "1111.com.tw",
#         "yes123.com.tw",
#         "findcompany.com.tw",
#         "twincn.com",
#         "iyp.com.tw",
#         "info.technews.tw",
#         "mygov.tw",
#     }

#     def get_domain(url: str) -> str:
#         try:
#             netloc = urlparse(url).netloc.lower()
#             if netloc.startswith("www."):
#                 netloc = netloc[4:]
#             return netloc
#         except Exception:
#             return ""

#     def domain_in(domain: str, allow_domains: set[str]) -> bool:
#         return any(domain == d or domain.endswith("." + d) for d in allow_domains)

#     def normalize_for_match(text: str) -> str:
#         text = text or ""
#         text = re.sub(r"\s+", "", text)
#         text = text.replace("　", "")
#         return text

#     def has_exact_company_name(text: str, target_name: str) -> bool:
#         return normalize_for_match(target_name) in normalize_for_match(text)

#     try:
#         # 先做非常基本的格式檢查，避免 OCR 明顯雜訊直接打搜尋
#         normalized_name = _normalize_company_name_for_check(name)

#         if not normalized_name:
#             result = {
#                 "exists": False,
#                 "reason": "公司名稱清洗後為空",
#                 "source": "format_check",
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 明顯不像台灣公司名稱的先擋掉
#         company_suffixes = (
#             "有限公司",
#             "股份有限公司",
#             "有限合夥",
#             "商行",
#             "企業社",
#             "工作室",
#             "行",
#             "店",
#         )

#         if not any(normalized_name.endswith(suffix) for suffix in company_suffixes):
#             result = {
#                 "exists": False,
#                 "reason": "名稱不像常見台灣公司或商業登記名稱",
#                 "source": "format_check",
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         queries = [
#             f'"{normalized_name}" 統一編號 公司登記 經濟部',
#             f'"{normalized_name}" 商工登記',
#             f'"{normalized_name}" 公司登記',
#         ]

#         evidence = []
#         seen_urls = set()

#         for query in queries:
#             response = tavily_client.search(
#                 query=query,
#                 search_depth="advanced",
#                 max_results=10,
#                 include_answer=False,
#                 include_raw_content=False,
#             )

#             for item in response.get("results", []):
#                 url = item.get("url", "") or ""
#                 if not url or url in seen_urls:
#                     continue

#                 seen_urls.add(url)

#                 title = item.get("title", "") or ""
#                 content = item.get("content", "") or ""
#                 combined_text = f"{title} {content}"

#                 # 沒有完整公司名的結果先不列入命中
#                 if not has_exact_company_name(combined_text, normalized_name):
#                     continue

#                 domain = get_domain(url)

#                 hit = {
#                     "title": title,
#                     "url": url,
#                     "domain": domain,
#                     "content": content[:500],
#                     "score": item.get("score"),
#                     "query": query,
#                     "is_trusted": domain_in(domain, trusted_domains),
#                     "is_supporting": domain_in(domain, supporting_domains),
#                 }

#                 evidence.append(hit)

#         trusted_hits = [item for item in evidence if item["is_trusted"]]
#         supporting_hits = [item for item in evidence if item["is_supporting"]]
#         other_hits = [
#             item for item in evidence
#             if not item["is_trusted"] and not item["is_supporting"]
#         ]

#         # 1. 政府 / 經濟部 / 商工登記可信來源有完整公司名稱
#         if trusted_hits:
#             hit = trusted_hits[0]
#             result = {
#                 "exists": True,
#                 "reason": "可信政府或公司登記來源中出現完整相符公司名稱",
#                 "source": "tavily_trusted",
#                 "matched_url": hit["url"],
#                 "matched_title": hit["title"],
#                 "evidence": trusted_hits[:3],
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 2. 一般支援來源有完整公司名稱，但不是官方登記來源
#         if supporting_hits:
#             hit = supporting_hits[0]
#             result = {
#                 "exists": True,
#                 "reason": "搜尋結果有完整相符公司名稱，但來源不是政府或公司登記官方資料",
#                 "source": "tavily_supporting",
#                 "matched_url": hit["url"],
#                 "matched_title": hit["title"],
#                 "evidence": supporting_hits[:3],
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 3. 其他網站有完整公司名稱，但可信度不足
#         if other_hits:
#             hit = other_hits[0]
#             result = {
#                 "exists": None,
#                 "reason": "搜尋結果有完整相符公司名稱，但來源可信度不足",
#                 "source": "tavily_other",
#                 "matched_url": hit["url"],
#                 "matched_title": hit["title"],
#                 "evidence": other_hits[:3],
#             }
#             _company_existence_cache[cache_key] = result
#             return result

#         # 4. 完全沒有完整相符結果
#         result = {
#             "exists": False,
#             "reason": "Tavily 搜尋結果中沒有找到完整相符公司名稱",
#             "source": "tavily",
#             "matched_url": None,
#             "matched_title": None,
#             "evidence": [],
#         }
#         _company_existence_cache[cache_key] = result
#         return result

#     except Exception as e:
#         result = {
#             "exists": None,
#             "reason": f"exception: {e}",
#             "source": "tavily",
#         }
#         _company_existence_cache[cache_key] = result
#         return result
    
def verify_company_existence_by_model(company_name: str) -> dict:
    """
    用模型做「保守」存在性/可疑性判斷：
    - true  : 很有把握存在/合理（通常很少）
    - false : 明顯不像公司名/明顯錯
    - null  : 無法確定（不要猜）
    回傳: {"exists": True/False/None, "reason": "..."}
    """
    name = (company_name or "").strip()
    if not name:
        return {"exists": None, "reason": "empty"}

    key = _normalize_company_name_for_check(name)
    if key in _company_existence_cache:
        return _company_existence_cache[key]

    #請只根據公司名稱字面是否合理，做保守判斷。
    prompt = f"""
請判斷是否存在此公司名稱

規則：
- 若名稱包含明顯 OCR 雜訊/符號，或不像台灣公司名稱 → exists=false
- 若名稱看起來合理，但你無法確定真實存在 → exists=null（不要猜）
- 只有在你非常確定是明確存在且常見的公司時 → exists=true（通常很少）

公司名稱：{name}

請只回傳 JSON，不要加任何說明：
{{
  "exists": true 或 false 或 null,
  "reason": "一句話原因"
}}"""

    try:
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL5,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0

        )

        content = (response.choices[0].message.content or "").strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            result = {"exists": None, "reason": "non-json"}
            _company_existence_cache[key] = result
            return result

        # raw = json.loads(json_match.group())
        raw = json.loads(repair_json(json_match.group()))
        exists = raw.get("exists")

        # 兼容模型可能回 "null"/"true"/"false" 字串
        if isinstance(exists, str):
            low = exists.strip().lower()
            if low == "true":
                exists = True
            elif low == "false":
                exists = False
            else:
                exists = None
        elif exists is True:
            exists = True
        elif exists is False:
            exists = False
        else:
            exists = None

        result = {"exists": exists, "reason": str(raw.get("reason", "")).strip()}
        _company_existence_cache[key] = result
        return result

    except Exception as e:
        result = {"exists": None, "reason": f"exception: {e}"}
        _company_existence_cache[key] = result
        return result


FINANCIAL_UPPERCASE_NUMERALS = set("零壹貳參叁肆伍陸柒捌玖")
FINANCIAL_UPPERCASE_UNITS = set("拾佰仟萬億元角分整")


def _has_suspected_financial_uppercase_amount(
    image_pil: Image.Image,
) -> tuple[bool, str | None]:
    """
    獨立檢查圖片中是否「實際出現」疑似中文財務大寫金額。

    回傳：
    - True：圖片中有疑似中文財務大寫，後續沿用原本 prompt 邏輯。
    - False：圖片中沒有疑似中文財務大寫，最後強制輸出 None。
    """
    gate_prompt = """
你只負責判斷圖片中是否「實際出現疑似中文財務大寫金額」，不要擷取其他欄位。

【判斷為 true 的條件】
- 圖片中可看到一段疑似中文財務大寫金額文字。
- 即使部分模糊、缺字或無法完整辨識，也可以判斷為 true。
- 該段文字必須在同一金額區域中，至少同時疑似包含：
  1. 一個財務大寫數字：
     零、壹、貳、參、肆、伍、陸、柒、捌、玖
  2. 一個金額位數或結尾字：
     拾、佰、仟、萬、億、元、角、分、整

【判斷為 false 的條件】
- 圖片只有阿拉伯數字金額，例如 33,075.00。
- 圖片只有「合計」「總計」「金額」「新臺幣」「金額大寫」等標題，
  但沒有實際中文財務大寫金額內容。
- 圖片中完全看不到疑似中文財務大寫數字。

【嚴格禁止】
- 禁止根據未稅金額、稅額、合計金額換算或產生中文大寫。
- 禁止因為知道阿拉伯數字金額，就假設圖片中存在中文大寫。
- visible_evidence 只能抄錄圖片中實際看到的疑似文字。
- 看不清楚的字可以使用「?」表示。

只輸出以下 JSON，不要輸出其他文字：
{
  "has_suspected_financial_uppercase_amount": false,
  "visible_evidence": null
}
"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": gate_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        },
                    },
                ],
            }],
            max_tokens=128,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (
            response.choices[0].message.content or ""
        ).strip()

        json_match = re.search(r"\{.*\}", content, re.DOTALL)

        if not json_match:
            print(
                "⚠️ [金額大寫存在性 Gate] 找不到 JSON，"
                "不阻擋原本辨識"
            )

            # Gate 自己失敗時，不能影響原本流程。
            return True, None

        gate_result = json.loads(
            repair_json(json_match.group())
        )

        model_found = (
            gate_result.get(
                "has_suspected_financial_uppercase_amount"
            )
            is True
        )

        evidence = gate_result.get("visible_evidence")

        if evidence is not None:
            evidence = str(evidence).strip()

        # 除了模型回傳 true，證據文字還必須同時包含：
        # 1. 至少一個財務大寫數字
        # 2. 至少一個金額位數或結尾字
        #
        # 避免模型只看到「金額大寫」或「新臺幣」標題就回傳 true。
        has_numeral = bool(evidence) and any(
            char in FINANCIAL_UPPERCASE_NUMERALS
            for char in evidence
        )

        has_unit = bool(evidence) and any(
            char in FINANCIAL_UPPERCASE_UNITS
            for char in evidence
        )

        found = bool(
            model_found
            # and has_numeral
            and has_unit
        )

        print(
            "[金額大寫存在性 Gate] "
            f"found={found}, "
            f"model_found={model_found}, "
            f"evidence={evidence!r}"
        )

        return found, evidence

    except Exception as e:
        # Fail-open：
        # Gate 發生例外時，保留你原本的辨識行為，
        # 避免新邏輯影響既有發票。
        print(
            f"⚠️ [金額大寫存在性 Gate] 檢查失敗：{e}，"
            "不阻擋原本辨識"
        )

        return True, None


def extract_detail_items_from_image(
    image_pil: Image.Image,
) -> dict:
    """
    使用獨立且精簡的 VLM prompt 辨識明細項目。

    只處理：
    - 品名
    - 數量
    - 單價
    - 金額

    不處理公司名稱、統編、稅額、合計金額等其他欄位。
    """

    # json_template = {
    #     "明細項目": [
    #         {
    #             "品名": "<「品名 / 規格」欄位的完整內容；必須包含商品名稱、規格、型號、料號、容量、重量、包裝規格等所有屬於同一商品的文字；或null>",
    #             "價格列原文": "<只包含數量、單價、金額欄位的原始文字，不包含品名/規格欄文字>",
    #             "單價": "<純數字或null>",
    #             "數量": "<數值；若原發票有單位則保留單位，例如 40.0 LT、5 PCS；或null>",
    #             "金額": "<金額欄完整內容；若圖片中的金額有貨幣單位或貨幣代碼，必須保留，例如 USD 902.40、EUR 1,200.00、NT$ 3,500；若沒有貨幣單位則只輸出金額；或null>",
    #         }
    #     ],
    #     "欄位信心值": {
    #         "明細項目": "<0~1或null>"
    #     },
    #     "reason": "<簡短說明價格列的欄位對應>"
    # }
    json_template = {
        
        "分筆判斷": [
            {
                "筆次": 1,
                "價格列原文": "<此筆的數量、單價、金額原文>",
                "品名範圍": "<只描述此筆對應的品名/規格文字>"
            }
        ],
        "reason": "<簡短說明價格列的欄位對應>",
        "明細項目": [
            {
                "品名": "<「品名 / 規格」欄位的完整內容；必須包含商品名稱、規格、型號、料號、容量、重量、包裝規格等所有屬於同一商品的文字；或null>",
                "價格列原文": "<只包含數量、單價、金額欄位的原始文字，不包含品名/規格欄文字>",
                "單價": "<純數字或null>",
                "數量": "<數值；若原發票有單位則保留單位，例如 40.0 LT、5 PCS；或null>",
                "金額": "<金額欄完整內容；若圖片中的金額有貨幣單位或貨幣代碼，必須保留，例如 USD 902.40、EUR 1,200.00、NT$ 3,500；若沒有貨幣單位則只輸出金額；或null>"
            }
        ],
        "欄位信心值": {
            "明細項目": "<0~1或null>"
        }
        
    }

    prompt = f"""
你只負責辨識這張發票圖片中的「商品明細」，不要辨識公司名稱、稅額、總計或其他欄位。

請擷取每筆明細的：

- 品名：
  此 JSON 欄位名稱雖然叫「品名」，
  但實際輸出內容定義為「品名 / 規格欄位的完整文字」。

  如果原發票欄位標題是「品名 / 規格」，
  則該欄位內屬於同一商品的所有內容都必須輸出到「品名」，
  包含：
  - 商品名稱
  - 中文名稱
  - 英文名稱
  - 型號
  - 料號
  - 容量
  - 重量
  - 包裝規格
  - 複合規格，例如 32.00PCx25.00KG

  不可只輸出商品名稱而省略規格。

- 單價
- 數量
- 金額

【最高優先原則：先判斷表格欄位】

第一步：
請先辨識明細表格的欄位標題及各欄位的水平範圍，例如：

貨號
品名 / 規格
數量
單價
金額
備註

第二步：
先依文字在圖片中的水平位置，
判斷每段文字屬於哪一個欄位。

第三步：
只有確認文字屬於「數量」、「單價」、「金額」欄位後，
才可將該文字放入「價格列原文」。

==================================================
【空白欄位處理－最高優先】
==================================================

數量、單價、金額是三個彼此獨立的欄位。

每一個欄位都必須分別查看圖片中該商品列對應的實際儲存格。

只有在該欄位的儲存格內實際看到文字或數字時，
才可以輸出該欄位的值。

如果某個欄位的儲存格是空白：

- 必須輸出 null。
- 不可從其他欄位搬移數字過來。
- 不可從相鄰欄位補值。
- 不可從「備註」欄取數字當成數量、單價或金額。
- 不可利用數量 × 單價 = 金額反推出缺少的欄位。
- 不可因為一筆商品通常應該有數量、單價、金額，就自行補齊。
- 不可將同一個數字同時填入兩個不同欄位。

【欄位實際位置優先】

若某個數字實際位於「金額」欄，
它只能輸出到「金額」。

即使「單價」欄為空白，
也禁止把金額欄的數字複製到單價。

若某個數字實際位於「備註」欄，
不論它看起來多像數量、單價或金額，
都禁止填入數量、單價或金額。

若表格已有明確欄位標題，欄位內容不得因其他欄位空白而向左或向右移位補值。
位於「備註」欄水平範圍內的內容，必須直接排除，不得放入數量、單價、金額或價格列原文。

【空白是合法結果】

一筆明細不要求「數量、單價、金額」三個欄位一定都有值。

可能出現：

數量 = null
單價 = null
金額 = 551200

這是合法結果。

必須照圖片實際內容輸出，

【價格列原文定義】

- 「價格列原文」不是指同一水平列上的所有文字。
- 「價格列原文」只包含數量、單價、金額欄位中的原始文字。
- 位於「品名 / 規格」欄位中的文字，
  即使與數量、單價、金額位於同一水平列，
  也不可放入「價格列原文」。

【欄位位置優先】

欄位位置的優先權高於文字格式。

例如：

32.00PCx25.00KG

雖然包含數字、x、PC、KG，
但如果它位於「品名 / 規格」欄位內，
它就是商品規格，
必須完整加入品名。

不可因為它包含 x，
就套用「x數量」的判斷規則。

【單價*數量複合欄位】

- 若表格欄位標題明確為「單價*數量」、「單價×數量」、
  「單價x數量」或「單價X數量」，
  則必須依表頭順序拆解。

- 「*、×、x、X」前面的數值 = 單價。
- 「*、×、x、X」後面的數值 = 數量。

例如：

表頭：單價*數量
內容：2000*5.3

則：
單價 = 2000
數量 = 5.3

不可將整段「2000*5.3」只當成數量。

【無法依欄位標題明確判斷時】

只有在無法從欄位標題與水平位置明確判斷欄位時，
才可以使用貨幣符號、x／× 等格式作為輔助判斷。

例如同一列由左到右為：

$3,500    x5    $17,500    TX

且 x5 是獨立的數量文字，不是商品規格的一部分時，

必須判斷為：

- 單價：3500
- 數量：5
- 金額：17500
- TX：稅別標記，不輸出

- 若沒有明確「單價*數量」表頭，
  但價格 / 數量區域中出現兩個數字以「*、×、x、X」連接，
  且其中只有一個數字帶有貨幣符號「$」，
  則必須以貨幣符號判斷單價，不可單純依「*」前後順序判斷。

- 帶有「$」的數值 = 單價。
- 另一個沒有「$」的數值 = 數量。
- 不論帶「$」的數值出現在「*」前面或後面，都適用此規則。

例如：

$255*18   $4,590TX

必須判斷為：

單價 = 255
數量 = 18
金額 = $4,590TX

例如：

18*$255   $4,590TX

也必須判斷為：

單價 = 255
數量 = 18
金額 = 4590

- 即使「$255*18」整段在視覺位置上落於「數量欄」，
  該欄位位置只能表示這整段屬於數量 / 單價的價格區域，
  不代表「$255」與「18」都屬於數量。

- 只要其中一個數值明確帶有「$」，
  必須將帶「$」的數值拆為單價，
  另一個數值拆為數量。

- 若「*」格式出現在品名 / 規格欄，例如 520mm*630mm，
  則仍屬於商品規格，不可拆成單價與數量。

【價格列映射規則】

1. 「x5」、「X5」、「×5」若是獨立出現在「數量欄」或明確的價格區域中，
   其中的 5 才優先判定為數量。

2. 若 x / X / × 是完整商品規格的一部分，例如：

   32.00PCx25.00KG
   12PCx500ML
   24EAx1KG

   則不得套用「x數量」規則。

   如果整段位於「品名 / 規格」欄，
   必須將整段完整加入「品名」。

   例如：

   32.00PCx25.00KG

   必須整段保留，
   不可拆成：
   - 數量 = 32
   - 數量 = 25
   - 單價 = 25

3. 位於獨立 x數量 左邊的貨幣金額，
   優先判定為單價。

4. 位於獨立 x數量 右邊的貨幣金額，
   優先判定為該筆總金額。

5. 帶有「$」、「NT$」、「NTD」等貨幣符號的數字，
   不可判定為數量。

6. 「TX」或「Tax」是稅別標記。
   若 TX / Tax 緊接在金額數字後方，例如「$4,590TX」，
   則「金額」必須保留完整原文「$4,590TX」，
   不可移除 TX / Tax。

   若 TX / Tax 是獨立文字，則不單獨輸出為其他欄位。

7. JSON 欄位排列順序不代表圖片上的排列順序，
   必須依圖片中的欄位位置判斷。

【禁止反推】

- 必須先依圖片中的位置、貨幣符號、x／× 符號決定欄位。
- 「數量 × 單價 = 金額」只能在欄位辨識完成後進行一致性檢查。
- 禁止使用除法自行產生圖片上沒有直接看到的數量或單價。
- 禁止因為算式成立，就重新交換數量與單價。
- 例如禁止將：
  $3,500 x5 $17,500
  解讀為：
  數量=3.5、單價=5000、金額=17500。

【數量輸出規則】

- 「x5」、「×5」輸出「5」。
- 不可將金額中的千分位小數點或逗號誤認成數量小數點。
- $3,500 或 $3.500 都是金額 3500，不是數量 3.5。

- 圖片可能使用「點號」作為千分位分隔符號，同時最後再帶小數 .00。
- 例如：
  4.000.00 = 4,000.00 = 4000
  8.000.00 = 8,000.00 = 8000
  12.000.00 = 12,000.00 = 12000
  14.800.00 = 14,800.00 = 14800

- 若價格列為：
  4.000.00 1.850 7.400.00

  必須輸出：
  數量 = 4000
  單價 = 1.85
  金額 = 7400

  絕對不可輸出：
  數量 = 4.0
  單價 = 1.85
  金額 = 7.4

- 若價格列為：
  8.000.00 1.850 14.800.00

  必須輸出：
  數量 = 8000
  單價 = 1.85
  金額 = 14800

- 「4.000.00」中的第一個點是千分位格式的一部分，
  最後的「.00」才是小數部分。
  不可把 4.000.00 解讀成 4.0。


【輸出順序－最高優先】

你必須按照以下順序處理：

1. 先找出所有不同的「數量 + 單價 + 金額」資料列。
2. 每找到一組新的「數量 + 單價 + 金額」，建立一筆「分筆判斷」。
3. 根據每一組價格資料列的位置，
   判斷其對應的「商品主名稱 + 跨行規格/料號」。

   商品名稱可能：
   - 與價格列位於同一水平列
   - 位於價格列上方
   - 跨越價格列上下多行

   不可單純把兩個價格列之間的所有文字全部歸給前一筆。
4. 完成所有分筆後，才可以產生「明細項目」。

「明細項目」筆數必須與「分筆判斷」筆數完全相同。

每一筆「品名」只能包含屬於該筆商品的文字，
禁止包含其他「分筆判斷」所對應的商品文字。

如果第 1 筆與第 2 筆有不同的數量、單價、金額資料列，
禁止將兩筆商品的品名合併後同時填入兩筆。

==================================================
【明細分筆規則－最高優先】
==================================================

在擷取品名之前，必須先判斷「共有幾筆商品」。

不可先把品名欄所有文字串在一起，再嘗試切分。

請優先使用：

- 數量欄
- 單價欄
- 金額欄

來判斷每一筆商品的起始資料列。

如果某一個水平列同時出現：

- 一個新的數量
- 一個新的單價
- 一個新的金額

則該水平列視為：

「一筆新商品的主資料列」。

每出現一個新的主資料列，
就代表開始下一筆商品。

==================================================
【同一筆商品的品名範圍】
==================================================

一筆商品的品名包含：

1. 該商品主資料列中位於「品名 / 規格」欄的文字。
2. 該主資料列下方的品名跨行文字。
3. 該主資料列下方的型號、規格、料號、P/N。

但是只能收集到：

「下一筆商品主資料列出現之前」。

一旦下一個水平列再次出現新的：

數量 + 單價 + 金額

就代表：

上一筆商品已結束，
下一筆商品開始。

後面的品名文字不得再加入上一筆商品。

==================================================
【價格列之間的品名歸屬－最高優先】
==================================================

非常重要：

兩組相鄰的「數量 + 單價 + 金額」價格列之間，
可能同時存在：

1. 前一筆商品的跨行續行內容
2. 下一筆商品的新商品名稱

因此：

禁止將「目前價格列之後、下一價格列之前」的所有品名文字，
無條件全部加入前一筆商品。

必須逐行判斷該文字屬於前一筆還是下一筆。

--------------------------------------------------
【前一筆商品的續行】
--------------------------------------------------

如果價格列下方的文字明顯只是延續前一筆商品，例如：

- 品名上一行尚未完成的後半段
- 型號
- 料號
- P/N
- 規格
- 容量
- 包裝文字
- DRUM
- PLASTIC DRUM
- 單獨的補充描述

且這些文字與前一筆商品共同構成完整品名，
則可以繼續加入前一筆商品。

例如：

商品A                    640 L    1.4100    902.40
DRUM
P/N 12-9080001

則：

DRUM
P/N 12-9080001

仍屬於商品A。

--------------------------------------------------
【下一筆商品的新商品名稱】
--------------------------------------------------

如果目前價格列之後出現一行新的完整商品名稱，
而該商品名稱下方或相鄰的下一個價格列
對應新的「數量 + 單價 + 金額」，

則該商品名稱必須歸屬於「下一筆商品」，

禁止加入前一筆商品。

也就是：

價格列A
商品名稱B
價格列B

必須判斷為：

商品A
→ 價格列A

商品B
→ 價格列B

禁止判斷成：

商品A + 商品B
→ 價格列A

--------------------------------------------------
【回頭確認規則】
--------------------------------------------------

每找到下一組新的「數量 + 單價 + 金額」價格列時，

必須回頭檢查：

「上一個價格列」與「目前價格列」之間的所有品名文字。

若其中存在新的商品主名稱，
則該文字應歸屬目前這一筆，
不可留在上一筆。

只有明確屬於上一筆的跨行規格、料號、型號或續行文字，
才可以保留在上一筆。

==================================================

==================================================
【本案例】
==================================================

若圖片為：

CUPOSIT 補充液 Y-2 20L PLASTIC      640 L    1.4100    USD 902.40
DRUM
P/N 12-9080001

CIRCUPOSIT(TM) 清潔調整劑3323A 20L  160 L    6.5000    USD 1,040.00
PLASTIC DRUM
P/N 12-93323A01

因為：

640 L / 1.4100 / USD 902.40

是一組完整的新價格資料，

而：

160 L / 6.5000 / USD 1,040.00

又是另一組完整的新價格資料，

所以必須分成兩筆商品。

第一筆：

"品名":
"CUPOSIT 補充液 Y-2 20L PLASTIC DRUM P/N 12-9080001"

"數量": "640 L"
"單價": "1.4100"
"金額": "902.40"

第二筆：

"品名":
"CIRCUPOSIT(TM) 清潔調整劑3323A 20L PLASTIC DRUM P/N 12-93323A01"

"數量": "160 L"
"單價": "6.5000"
"金額": "1040.00"

禁止第一筆輸出：

"CUPOSIT 補充液 Y-2 20L PLASTIC DRUM
P/N 12-9080001
CIRCUPOSIT(TM) 清潔調整劑3323A 20L
PLASTIC DRUM
P/N 12-93323A01"

因為：

CIRCUPOSIT(TM) 清潔調整劑3323A 20L

所在水平列已經同時出現：

160 L
6.5000
USD 1,040.00

這代表它是「下一筆商品」，
不是第一筆商品的跨行品名。  


【數量欄不可加入品名】

若文字位於明確的「數量」欄，例如：

640 L
160 L
40 LT
5 PCS

即使包含 L、LT、PCS、KG 等單位，
也必須輸出到「數量」。

不可因為它包含單位，
就把它當成商品規格加入品名。

「數字 + 單位」是否屬於品名，
必須先看它位於哪一個表格欄位。

例如：

品名                     數量

CUPOSIT ...              640 L

則：

640 L = 數量

不是品名規格。


【品名 / 規格欄位，最高優先】

- 請先確認明細表格上方的欄位標題。
- 若欄位標題為：
  「品名 / 規格」
  「品名/規格」
  「品名規格」
  或其他表示「品名與規格共用同一欄」的標題，

  則該欄位儲存格內所有屬於同一商品的：
  - 商品名稱
  - 型號
  - 料號
  - 容量
  - 重量
  - 包裝規格
  - 商品規格文字

  都必須合併輸出到「品名」。

【容量規格特別規則】

- 250ml、500ml、1000ml、1L、500g、1kg 等
  「數字 + 容量/重量單位」，
  若位於「品名 / 規格」欄位內，
  屬於商品規格，必須保留在「品名」。

- 不可因為 250ml、500ml 等文字位於品名欄的最右側，
  就將它忽略。

- 不可因為 250ml、500ml 是「數字 + 單位」，
  就直接判定為「數量」。

- 必須先依圖片中的表格欄線、欄位標題及水平位置判斷它屬於哪一個欄位。

- 若 250ml、500ml 位於「品名 / 規格」欄內，
  而真正的數量 1.0、1.000 等位於另一個數量欄，
  則：
  250ml / 500ml 必須加入品名，
  1.0 / 1.000 才是數量。

 【複合包裝規格規則－最高優先】

「品名 / 規格」欄位中可能出現複合包裝規格，例如：

32.00PCx25.00KG
4.00PCx25.00KG
8.00PCx25.00KG
16.00PCx25.00KG
12PCx500ML
24EAx1KG
6BOXx10PCS

上述格式屬於：

商品規格 / 包裝規格

不是：
- 數量
- 單價
- 金額

只要整段文字位於「品名 / 規格」欄位內，
就必須完整保留並合併到「品名」。

【x 符號的重要區別】

「x5」、「X5」、「×5」這類獨立出現在數量或價格區域中的文字，
才可以依前面的規則判定為數量。

但是：

32.00PCx25.00KG

是一個完整的包裝規格字串。

其中：

32.00PC
x
25.00KG

共同構成一個完整商品規格。

不可因為其中包含 x、X、×，
就套用「x5 為數量」的規則。

不可將：

32.00PCx25.00KG

拆解為：

數量 = 32

或：

數量 = 25

也不可把它放入「價格列原文」。

【本案例】

若圖片為：

品名 / 規格                         數量      單價       金額

REDUCTION SOLUTION CU-14
                    32.00PCx25.00KG   800.00   99.6989   79,759
化銅還原劑 Cu-14

則必須輸出：

"品名": "REDUCTION SOLUTION CU-14 32.00PCx25.00KG 化銅還原劑 Cu-14"

"數量": "800.00"

"單價": "99.6989"

"金額": "79759"

而：

"價格列原文": "800.00 99.6989 79,759"

禁止輸出：

"價格列原文": "32.00PCx25.00KG 800.00 99.6989 79,759"

【跨行品名與右側規格】

- 同一筆商品的品名可能跨越兩行以上。
- 第二行可能同時存在：
  左側的料號或型號，
  以及右側的容量或規格。
- 即使兩段文字距離很遠，只要仍位於同一個「品名 / 規格」儲存格，
  都屬於同一筆品名。

例如圖片呈現：

Conductivity, standard of1413 uS/cm 25°C PR-RS
#394659                                      250ml

必須輸出：

"品名": "Conductivity, standard of1413 uS/cm 25°C PR-RS #394659 250ml"

不可輸出：

"品名": "Conductivity, standard of1413 uS/cm 25°C PR-RS #394659"

另一個例子：

Conductivity, standard of 84 uS/cm25°C
(HANNA-HI-7033L)                            500ml

必須輸出：

"品名": "Conductivity, standard of 84 uS/cm25°C (HANNA-HI-7033L) 500ml"

不可漏掉：

500ml

【同一儲存格優先於文字距離】

- 判斷品名是否結束時，不能只依文字之間的距離判斷。
- 「品名 / 規格」欄本身可能很寬。
- 左側商品名稱與最右側的 250ml、500ml 即使相隔很遠，
  只要中間沒有跨越表格欄線，且都位於同一「品名 / 規格」欄，
  就必須視為同一筆商品內容。
  
【品名範圍】

- 品名的範圍必須優先由「品名 / 規格」欄位的位置決定，
  不可只依文字位於價格列上方或下方判斷。

- 品名可能跨越多行。

- 品名 / 規格欄中的文字，
  可能與數量、單價、金額位於相同水平列。

- 即使位於同一水平列，
  只要文字仍在「品名 / 規格」欄位範圍內，
  就必須視為品名的一部分。

例如：

品名 / 規格                         數量      單價       金額

REDUCTION SOLUTION CU-14   32.00PCx25.00KG   800.00   99.6989   79,759
化銅還原劑 Cu-14

其中：

REDUCTION SOLUTION CU-14
32.00PCx25.00KG
化銅還原劑 Cu-14

全部都屬於同一筆「品名 / 規格」。

因此必須輸出：

"品名": "REDUCTION SOLUTION CU-14 32.00PCx25.00KG 化銅還原劑 Cu-14"

【品名結束條件】

不可使用：

「找到目前商品的價格列後，品名立即結束」

作為判斷方式。

因為目前商品的品名可能在價格列下方繼續跨行。

但是如果下一個水平列出現一組新的：

- 數量
- 單價
- 金額

則該列視為「下一筆商品主資料列」。

此時：

上一筆商品的品名必須立即結束。

品名是否結束依以下優先順序判斷：

1. 下一組新的「數量 + 單價 + 金額」資料列出現。
   → 上一筆商品立即結束。

2. 已進入下一筆商品的品名 / 規格區域。

3. 已進入備註、稅額、小計、合計、總計等其他區域。

4. 文字已明確離開目前商品的品名 / 規格欄位。

【最高優先】

下一筆商品的數量 / 單價 / 金額，
不可被視為上一筆商品品名的延續依據。

不可因為同一水平列已經出現數量、單價或金額，
就忽略同一列中仍位於「品名 / 規格」欄位內的商品規格。

【排除內容】

- 明確位於備註欄中的文字，不加入品名。
- 明確位於數量、單價、金額欄中的文字，不加入品名。
- 「應稅銷售額」、「稅額」、「合計」、「總計」不是商品明細。

【項次與料號】

- 單獨成行的「0001」、「0002」等四位流水號，優先視為明細項次，不加入品名。
- 較長的商品編號若與商品描述位於同一商品區域，可保留在品名中。
- 不可因為 JSON 沒有料號欄位，就把明細項次加入品名。

請只輸出以下 JSON，不要輸出其他說明：

{json.dumps(json_template, ensure_ascii=False, indent=2)}
"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    f"{b64_image}"
                                )
                            },
                        },
                    ],
                }
            ],
            max_tokens=8192,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (
            response.choices[0].message.content or ""
        ).strip()

        print(
            f"[VLM 明細專用] 回應：\n{content}\n"
        )

        json_match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if not json_match:
            print(
                "⚠️ [VLM 明細專用] 找不到 JSON"
            )
            return {}

        result = json.loads(
            repair_json(json_match.group())
        )

        items = result.get("明細項目")

        if not isinstance(items, list):
            items = []

        cleaned_items = []

        for item in items:
            if not isinstance(item, dict):
                continue

            quantity = item.get("數量")

            if quantity is not None:
                quantity = str(quantity).strip()

                # 移除數量前方的 x、X、×。
                quantity = re.sub(
                    r"^[xX×＊*]\s*",
                    "",
                    quantity
                ).strip()

            unit_price = item.get("單價")

            if unit_price is not None:
                unit_price = (
                    str(unit_price)
                    .replace(",", "")
                    .strip()
                )

            amount = item.get("金額")

            if amount is not None:
                amount = (
                    str(amount)
                    .replace(",", "")
                    .strip()
                )

            # 「價格列原文」只用於迫使模型先看圖，
            # 不放入最終明細格式。
            cleaned_items.append({
                "品名": item.get("品名"),
                "數量": quantity,
                "單價": unit_price,
                "金額": amount,
            })

        confidence_map = (
            _normalize_field_confidence_map(
                result.get("欄位信心值"),
                ["明細項目"]
            )
        )

        _detail_reason = result.get("reason")
        if _detail_reason:
            print(f"[VLM 明細專用][reason] {_detail_reason}")

        return {
            "明細項目": cleaned_items,
            "__field_confidence__": confidence_map,
            "detail_reason": _detail_reason,
        }

    except json.JSONDecodeError as exc:
        print(
            "⚠️ [VLM 明細專用] "
            f"JSON 解析失敗：{exc}"
        )
        return {}

    except Exception as exc:
        print(
            "⚠️ [VLM 明細專用] "
            f"呼叫失敗：{exc}"
        )
        return {}

def _has_explicit_remark_field(
    image_pil: Image.Image
) -> bool:

    prompt = """
你只負責判斷圖片中是否明確存在「備註」欄位標記。

【只有以下情況才回答 true】

1. 表格中明確看到「備註」兩個字作為欄位標題。
2. 圖片中明確看到「備註：」。
3. 圖片中明確看到「備註:」。

【必須回答 false 的情況】

- 圖片中完全沒有看到「備註」兩個字。
- 只有手寫數字、編號、文字，但附近沒有「備註」欄位標題。
- 某段文字只是位於交易明細下方或旁邊。
- 看到類似 670-213043、6700203988 等編號，
  但圖片中沒有明確的「備註」標記。
- 只有「總備註：」，不算一般備註欄位。

【禁止推論】

不得根據：
- 位置
- 內容
- 編號格式
- 手寫文字
- 發票版型
- 上下文

推論它是備註。

只看圖片中是否真的存在「備註」文字標記。

只輸出：

{
  "found": true
}

或

{
  "found": false
}
"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    f"{b64_image}"
                                )
                            }
                        }
                    ]
                }
            ],
            max_tokens=128,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (
            response.choices[0]
            .message.content
            or ""
        ).strip()

        print(
            f"[備註存在性 Gate] 回應：\n"
            f"{content}"
        )

        json_match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if not json_match:
            print(
                "⚠️ [備註存在性 Gate] "
                "找不到 JSON"
            )

            # Gate 自己失敗時先維持原邏輯
            return True

        result = json.loads(
            repair_json(json_match.group())
        )

        found = result.get("found") is True

        print(
            "[備註存在性 Gate] "
            f"found={found}"
        )

        return found

    except Exception as e:
        print(
            f"⚠️ [備註存在性 Gate] "
            f"檢查失敗：{e}"
        )

        # Fail-open，不影響原流程
        return True

def extract_fields_from_image_region(
    image_pil: Image.Image,
    failed_fields: list
) -> dict:
    """
    單次 VLM 辨識，回傳擷取結果
    重試邏輯由 app.py 的比對結果決定
    """
    # VLLM_MODEL = VLLM_LLM_MODEL4
    # if len(failed_fields) == 1 and failed_fields[0] == "賣方公司名稱":
    #     VLLM_MODEL = VLLM_LLM_MODEL4
    # elif len(failed_fields) == 1 and failed_fields[0] == "買方公司名稱":
    #     VLLM_MODEL = VLLM_LLM_MODEL4
    # else:
    #     VLLM_MODEL = VLLM_LLM_MODEL2
    if len(failed_fields) == 1 and failed_fields[0] == "營業稅稅別判斷":
        VLLM_MODEL = VLLM_LLM_MODEL3
    else:
        VLLM_MODEL = VLLM_LLM_MODEL4

     # 保留最初要求的欄位清單。
    requested_fields = list(
        dict.fromkeys(failed_fields)
    )

    detail_vlm_result = {}

    # 明細項目使用獨立、精簡的圖片 prompt。
    if "明細項目" in requested_fields:
        detail_vlm_result = (
            extract_detail_items_from_image(
                image_pil
            )
        )

        detail_items = detail_vlm_result.get(
            "明細項目"
        )

        # 明細專用辨識成功後，
        # 不再把明細塞進原本的大 prompt。
        if (
            isinstance(detail_items, list)
            and len(detail_items) > 0
        ):
            failed_fields = [
                field
                for field in requested_fields
                if field != "明細項目"
            ]
        else:
            # 專用辨識失敗時採 fail-open，
            # 保留原本的大 prompt 邏輯。
            failed_fields = requested_fields
    if (
        not failed_fields
        and detail_vlm_result
    ):
        return detail_vlm_result



    #     "發票日期": """發票上的開立日期
    #    - 可能是民國格式（如「115年03月17日」）或西元格式（如「2026/03/17」或「2026-03-17」）
    #    - 直接輸出原始格式，不需轉換
    #    - 找不到填 null
    #    - 請逐字確認格式正確，不要輸出到錯誤內容
    #    - 其中內容需包含年,月,日

    remark_gate_found = True

    if "備註" in failed_fields:
        remark_gate_found = _has_explicit_remark_field(image_pil)

    # 獨立存在性 Gate：
    # 不修改原 prompt，只決定最後是否允許保留「金額大寫中文」。
    amount_uppercase_gate_found = True

    if "金額大寫中文" in failed_fields:
        amount_uppercase_gate_found, _ = (
            _has_suspected_financial_uppercase_amount(
                image_pil
            )
        )


    field_instructions = {
        "發票日期": """
            - 目標是直接從圖片中辨識「實際發票日期文字」。
            - 不需要辨識「發票日期」標題本身，也不要把「統一編號」、
            「買受人」、「地址」等其他欄位內容當成發票日期。

            
            【日期輸出格式－最高優先】

            - 發票日期必須保留圖片中日期本身的原始格式。
            - 只允許移除日期後方的時間 HH:MM 或 HH:MM:SS。
            - 除了移除時間以外，禁止對日期做任何格式轉換。
            - 禁止將民國年轉換成西元年。
            - 禁止將西元年轉換成民國年。
            - 禁止改變 /、-、年/月/日 等原始分隔格式。
            - 禁止自行補零或去除前導零。

            例如圖片顯示：

            115/07/09 14:06:41

            必須輸出：

            115/07/09

            禁止輸出：

            2026-7-9
            2026/07/09
            115/7/9

            
            【最高優先原則】
            - 優先尋找圖片中具有「年、月、日」完整日期意義的文字。
            

            - 可接受的日期格式包含但不限於：
            中華民國115年7月15日
            115年7月15日
            115年07月15日
            115/07/15
            115-07-15
            0115.07.15
            2026/07/15
            2026-07-15
            2026-7-13
            2026-7-13 10:34:04
            2026/7/13 10:34:04

            【民國年前導零判斷】

            - 若日期為「4位數年份 + 分隔符號 + 月 + 分隔符號 + 日」，
            且4位數年份的第一位為 0，
            則必須將第一位 0 視為民國年份的前導零。

            - 例如年份部分為 0115 時，
            必須視為民國 115 年，
            不可因為年份是 4 位數或開頭為 0，
            就將整段判斷為一般數字格式。

            - 此類文字只要後方同時具有有效的月份與日期，
            就必須視為完整的年月日日期候選，
            不需要附近另外出現「日期」、「發票日期」或「開立日期」標題。
            

            - 若日期後方同時帶有時間，例如：
            2026-7-13 10:34:04

            則仍應將其中的：
            2026-7-13

            視為有效的發票日期候選。

            - 若輸出欄位只需要日期，
            則應去除後方時間，只輸出日期部分。

            例如圖片顯示：
            2026-7-13 10:34:04

            應輸出：
            2026-7-13

            【完整日期優先於年度期間】
            - 若圖片中同時存在：

            115年07-08月

            以及：

            2026-7-13 10:34:04

            則：
            「115年07-08月」只是發票年度期間，
            「2026-7-13」才是具有年、月、日的完整日期。

            因此必須優先輸出：
            2026-7-13

            - 以下格式屬於「年度期間」，不是發票日期：
            115年07-08月
            115年01-02月
            115年03-04月
            115年05-06月
            115年09-10月
            115年11-12月

            看到這類格式時不可輸出為發票日期。

            【多個日期的判斷】
            - 若圖片中有多個具有「年、月、日」的完整日期，
            才需要進一步判斷哪一個最可能是發票開立日期。

            - 若某日期位於「日期」、「發票日期」、「開立日期」附近，
            可以提高其優先度。

            - 但「附近沒有日期標題」不能作為排除完整日期的理由。

            - 若圖片中只有一個可信的完整年月日日期，
            即使旁邊沒有「發票日期」標題，
            仍應優先將該日期視為發票日期候選。

            【時間字串規則】
            - 禁止因為日期後方帶有 HH:MM 或 HH:MM:SS，
            就自行判定為「系統時間」而排除。

            - 只有當圖片中明確出現：
            「系統時間」
            「列印時間」
            「列印日期」
            「產生時間」
            「建立時間」
            或其他可明確證明不是發票日期的標籤時，
            才可以排除該日期時間。

            - 不可只根據版面位置或常識，
            自行把一個完整日期時間判斷為系統時間。

            【圖片內容優先】
            - 必須以圖片實際可見內容為準。
            - 不可根據統一編號、發票號碼、年度期間、
            金額或其他欄位推算日期。
            - 不可自行補上圖片中不存在的年月日。

            【禁止事項】
            - 禁止把「115年07-08月」等雙月期間當成發票日期。
            - 禁止因為日期後面有時間，就直接排除該日期。
            - 禁止在沒有明確標籤證據的情況下，
            自行宣稱某個日期是「系統時間」。
            - 禁止把統一編號中的數字誤認為日期。
            - 禁止把發票號碼誤認為日期。
            - 禁止根據年度期間推算發票日期。
            - 禁止輸出圖片中實際不存在的日期。

            - 若完全找不到任何具有「年、月、日」意義的可信日期，
            才輸出 null。
        """,
        "備註": """
            ==================================================
            【最高優先原則：先判斷備註型態】
            ==================================================

            請先判斷目前圖片中的「備註」屬於以下哪一種類型：

            【類型 A：表格備註欄】

            - 「備註」與「品名」、「數量」、「單價」、「金額」等欄位標題
            位於同一列或明顯屬於表格表頭。
            - 此時「備註」是一個表格欄位。
            - 必須依下方【表格備註欄】規則處理。
            - 備註欄內實際存在的內容都必須照實輸出，
            不可因為內容看起來像金額、稅額、總計或其他欄位就刪除。


            【類型 B：非表格備註】

            - 如果「備註」、「備註：」或「備註:」
            是一般文字標籤，而不是表格欄位標題，
            則只擷取「備註」標籤後面的內容。

            - 不要輸出「備註」兩個字本身。

            - 如果「備註」後方同一行有內容，
            只輸出該行位於「備註」後方的內容。

            - 不可把「備註」上方的其他欄位內容一起輸出。

            - 即使裁切圖片中同時看到：
            稅額、總計、合計、銷售額、日期、金額或其他欄位，
            只要這些內容位於「備註」標籤之前或其他行，
            都不可當成備註輸出。

            - 非表格備註不得套用
            「備註欄內所有文字都必須保留」
            的表格備註規則。

            - 若「備註」後方同一行沒有內容，
            才可以查看緊接在「備註」下一行的文字，
            並只擷取明確屬於該備註的內容。


            ==================================================
            【共同原則：照實抄錄】
            ==================================================

            - 只擷取實際屬於備註範圍內的原文。
            - 不可自行摘要、改寫、補字或推論不存在的內容。

            【表格備註欄】

            若圖片中「備註」是表格的獨立欄位：

            - 不要輸出欄位標題本身的「備註」兩個字。
            - 但是「備註」欄位標題下方的所有實際內容都必須保留。
            - 包含：
            - 中文文字
            - 英文文字
            - 數字
            - 編號
            - 型號
            - 標籤名稱
            - 冒號
            - 括號
            - 斜線
            - 連字號
            - 其他實際可見符號

            例如圖片中的備註欄實際顯示：

            客戶編號：
            B0293

            銷貨單號：
            115070700

            客戶訂單：
            6700212033

            則必須輸出完整內容：

            "客戶編號：B0293；銷貨單號：115070700；客戶訂單：6700212033"

            禁止輸出：

            "B0293；115070700；6700212033"

            因為這樣會遺漏圖片中實際存在的：

            客戶編號：
            銷貨單號：
            客戶訂單：


            ==================================================
            【無標籤備註內容也必須保留－最高優先】
            ==================================================

            備註欄內的文字不一定具有「標籤 + 值」格式。

            只要文字實際位於「備註」欄位標題下方，
            且仍位於此次提供圖片中的備註欄區域內，
            不論文字內容的語意或格式為何，
            都必須原文輸出。

            例如：

            備註

            1.4100/1 L

            6.5000/1 L

            其中：

            1.4100/1 L
            6.5000/1 L

            雖然沒有：

            客戶編號：
            銷貨單號：
            PO：
            規格：

            等文字標籤，

            但它們實際位於「備註」欄中，
            因此仍然是有效備註內容。

            必須輸出：

            "1.4100/1 L；6.5000/1 L"

            禁止因為：

            - 沒有中文標籤
            - 內容主要由數字組成
            - 看起來像單價
            - 看起來像數量
            - 包含「/1 L」
            - 可以被理解成價格資訊

            就將其省略。

            【欄位位置優先於文字語意－最高優先】

            - 判斷某段文字是否屬於備註時，只看它是否實際位於「備註」欄內。
            - 文字的語意、名稱、格式、數值用途，不得作為排除依據。
            - 即使文字看起來像其他發票欄位，只要它實際印在「備註」欄內，就仍然必須輸出。

            因此，下列內容如果實際位於「備註」欄內，也必須原文保留：

            - 銷售額
            - 銷售額合計
            - 稅額
            - 營業稅
            - 總計
            - 合計
            - USD 金額
            - 幣別
            - Rate
            - 匯率
            - PO#
            - Billing
            - Order
            - Customer

            禁止因為某段文字「看起來像銷售額、稅額、總計或其他金額欄位」
            就將它從備註內容中排除。

            例如圖片中的備註欄實際顯示：

            銷售額 USD 1942.40
            稅 額 USD 97.1200
            總 計 USD 2039.5200
            PO#:6700212719
            Rate:32.16500

            則以上全部都是此次備註原文擷取的內容，
            必須全部輸出。

            不可只輸出：

            PO#:6700212719；Rate:32.16500

            因為這樣會遺漏實際位於備註欄中的
            「銷售額」、「稅額」、「總計」內容。

            【標籤與值的關係】

            - 如果一段文字由「標籤 + 值」組成，標籤和值都必須保留。

            例如：

            客戶編號：
            B0293

            必須輸出：

            客戶編號：B0293

            不可只輸出：

            B0293

            例如：

            銷貨單號：
            115070700

            必須輸出：

            銷貨單號：115070700

            不可只輸出：

            115070700

            例如：

            客戶訂單：
            6700212033

            必須輸出：

            客戶訂單：6700212033

            不可只輸出：

            6700212033

            【閱讀順序】

            - 請依圖片由上到下、由左到右的實際閱讀順序擷取。
            - 同一組「標籤 + 值」如果因換行而分成兩行，輸出時可以合併為同一段。
            - 不同內容區塊之間使用「；」分隔。
            - 不可改變原始文字順序。

            【禁止語意整理】

            - 不可判斷哪些文字「比較重要」而只留下那些內容。
            - 不可認為中文標籤只是說明文字而刪除。
            - 不可只留下數字或編號。
            - 不可把內容重新整理成自己理解的格式。
            - 不可根據內容推論其用途。
            - 只負責忠實讀取圖片。

            【數字完整性規則】

            - 同一個完整編號中的連續數字必須保留為一個完整字串。
            - 即使數字之間有空隙、筆畫斷裂、書寫傾斜或部分重疊，也不可自行拆成多段。
            - 不可在同一個完整編號中間加入「；」、空格、逗號或其他不存在的分隔符號。
            - 請依照圖片實際內容逐字辨識，不可自行補數字、刪除數字或重新排列。



            【輸出規則】

            - 只輸出圖片中備註欄實際存在的內容。
            - 有「標籤 + 值」時，標籤與值都必須保留。
            - 沒有標籤時，也必須保留備註欄內實際存在的原始文字。
            - 不輸出欄位表頭本身的「備註」。
            - 不要輸出解釋或摘要。
            - 找不到有效備註內容時輸出 null。

            【正確範例】

            圖片：

            客戶編號：
            B0293

            銷貨單號：
            115070700

            客戶訂單：
            6700212033

            正確：

            "客戶編號：B0293；銷貨單號：115070700；客戶訂單：6700212033"

            錯誤：

            "B0293；115070700；6700212033"

            錯誤原因：
            遺漏了圖片實際存在的文字標籤。
            """,

        "金額大寫中文": """
            - 金額大寫中文：只輸出繁體中文財務大寫金額本身，不要包含「新臺幣」前綴。

            - 注意: 「一」不是有效的金額大寫中文，請不要自行轉換成「壹」。

            - 此欄位所表示的數值，必須與「合計金額」完全相同。
            - 金額大寫中文是「合計金額」的中文財務大寫表示，
            不是未稅金額，也不是稅額。

            - 請先逐字辨識圖片上實際印出的金額大寫中文。
            - 如果圖片中的金額大寫中文清楚，而且與合計金額一致，請使用圖片上的文字。
            - 如果金額大寫中文模糊、缺字、形近字辨識錯誤，或與合計金額不一致，
            但合計金額清楚可信，請將合計金額轉換為繁體中文財務大寫後輸出。

            - 例如：
            合計金額 35910
            必須輸出：參萬伍仟玖佰壹拾元整

            - 禁止將未稅金額 34200 轉換成金額大寫中文。
            - 禁止混合未稅金額、稅額及合計金額的數字。
            - 若合計金額也無法可靠辨識，才輸出 null。

            - 使用以下繁體中文財務數字：
            零、壹、貳、參、肆、伍、陸、柒、捌、玖、拾、佰、仟、萬、億、元、整。

            - 表格框線、底線、刪除線、水平長線及欄位分隔線都不是文字，必須完全忽略。
            """,

        "未稅金額": """
        - 未稅金額：擷取代表「稅前銷售金額」的主要結算金額。
        - 常見標示例如「銷售額」、「銷售額合計」、「銷售額(應稅)」，但不限於這些名稱。
        - 不可將原幣、外幣或匯率換算區的金額誤認為未稅金額。
        - 輸出純數字並去除逗號。
        """,

        "稅額": """
        - 稅額：擷取代表本張發票「營業稅／稅金」的主要結算金額。
        - 常見標示例如「營業稅」、「稅額」，但不限於這些名稱。
        - 不可將原幣、外幣或匯率換算區的稅額誤認為本次稅額。
        - 輸出純數字並去除逗號。
        """,

        "合計金額": """
        - 合計金額：擷取代表本張發票最終含稅應付總額的主要結算金額。
        - 常見標示例如「總計」、「合計」、「應付金額」，但不限於這些名稱。
        - 不可將原幣、外幣或匯率換算區的含稅金額誤認為本次合計金額。
        - 輸出純數字並去除逗號。
        """,

        
        #"未稅金額":    "- 未稅金額：未含稅的銷售金額（純數字，去除逗號）",
        #"稅額":       "- 稅額：營業稅金額（純數字，去除逗號）",
        #"合計金額":    "- 合計金額：含稅總計金額（純數字，去除逗號）",
        "年度期間":    "- 年度期間：格式為「民國年份年MM-MM月」，例如「115年03-04月」",
          "發票號碼":    "- 發票號碼：2個英文字母+8個數字，例如「AY83205584」",
        "買方統編":    """- 買方統編：買方的8位數字統一編號
                            - 注意：統一編號只會有八碼
                            - 注意：請逐個數字確認
                            - 注意：如果有找到「05637971」這串數字，這個就是買方統編""",
        "賣方統編": """
            - 賣方統編：8 位數字統一編號。

            【固定排除規則】
            - 05637971 是固定買方統編，任何情況下都不可輸出為賣方統編。
            - 判斷圖片中的 8 位數字候選時，必須先排除 05637971，再套用後續規則。


            【最高優先原則：單一連續8位數字直接取值】

            - 第一優先先檢查圖片中所有連續的阿拉伯數字。

            - 「連續8位數字」是指 8 個阿拉伯數字本身連續排列，
            前面或後面出現非數字符號，不會破壞這 8 位數字的連續性。


            - 如果圖中出現非數字符號，例如 #、NO.、No.、-、/、等，禁止把非數字符號兩側的數字拼接成一組 8 位數字。

            - 例如圖片中出現「日期時間# + 8位數字」時，
            # 只是前方的分隔符號，
            # 後面的 8 位數字本身仍然必須視為一組完整的「連續8位數字」。

            - 同理，「NO. + 8位數字」、「No. + 8位數字」，
            後方的 8 位數字本身也屬於完整的連續8位數字。

            - 不要求 8 位數字前方一定是空白。
            - 不要求 8 位數字後方一定是空白。
            - 不可因為 8 位數字前方緊接 #、NO.、No. 或其他非數字符號，
            就忽略該 8 位數字。

            - 排除固定買方統編 05637971 後，
            如果目前圖片中只剩下一組可清楚辨識的「連續 8 位阿拉伯數字」，
            必須直接將這組 8 位數字輸出為「賣方統編」。

            - 此規則只判斷「是否為連續8位數字」，
            不需要先判斷這組數字是否能確認為統一編號。

            - 不得因為附近沒有「統一編號」、「統編」、「賣方」等文字而輸出 null。

            - 不得因為「無法確認此8位數字是否為賣方統編」而輸出 null。


            【多組統編時】
            - 排除 05637971 後，
            如果圖片中仍存在兩組以上不同的完整連續8位數字，
            才需要依「買方」、「賣方」、「買受人」等標籤與位置判斷哪一組屬於賣方。


            【其他常見版型】
            - 若圖片中實際清楚看到「# + 8位數字」，可擷取 # 後面的 8 位數字。
            - 若圖片中實際清楚看到「NO. + 8位數字」或「No. + 8位數字」，
            可擷取後方的 8 位數字。
            - 輸出時不要包含 #、NO.、No.。


            【禁止事項】
            - 只能輸出圖片中實際清楚看到的完整 8 位數字。
            - 不可補字、猜測或產生圖片中不存在的數字。
            - 05637971 絕對不可輸出為賣方統編。
            - 排除 05637971 後，若完全看不到完整連續8位數字，才輸出 null。
        """,
        "買方公司名稱": """
            - 買方公司名稱：請輸出圖片中實際顯示的買方名稱文字。

            【最高優先原則】
            - 如果辨識出的買方公司名稱，文字中未含有「燿華」、「燿」、「耀」、「華」，買方公司名稱應修正輸出成 null。

            【優先原則：只能輸出圖片原文】
            - 必須依照圖片中實際看到的文字逐字輸出。
            - 不可依公司常見名稱、正式登記名稱、統編、既有知識或上下文自行補全公司名稱。
            - 不可把簡稱、廠別名稱、分公司名稱或單位名稱改寫成正式公司名稱。
            - 不可因為知道某文字可能隸屬於某家公司，就將其轉換成該公司的完整法定名稱。

            【例如】
            - 圖片實際顯示：「燿華-宜蘭廠」
            → 必須輸出：「燿華-宜蘭廠」

            - 圖片實際顯示：「燿華電子(股)公司」
            → 必須輸出：「燿華電子(股)公司」

            - 圖片實際顯示：「燿華電子股份有限公司」
            → 必須輸出：「燿華電子股份有限公司」

            【禁止範例】
            - 圖片只顯示「燿華-宜蘭廠」，
            禁止自行改成「燿華電子股份有限公司」。

            - 圖片只顯示「燿華-宜蘭廠」，
            禁止自行改成「燿華電子(股)公司」。

            【形近字規則】
            - 特別注意「燿／耀」。
            - 只能根據圖片實際筆畫判斷是哪一個字。
            - 不可以因為已知公司名稱，而強制把「耀」改成「燿」或把「燿」改成「耀」。

            【找不到時】
            - 如果圖片中沒有清楚可辨識的買方名稱文字，輸出 null。
            - 不可自行產生圖片中不存在的公司名稱。

            """,
        "賣方公司名稱": """
            - 賣方公司名稱：完整公司名稱。

            【先找公司名稱文字候選】
            買方公司名稱與賣方公司名稱的判斷，必須先找到圖片中實際出現的公司名稱文字候選。

            - 股份有限公司
            - 有限公司
            - 公司
            - 企業有限公司
            - 企業社
            - 商行
            - 商號
            - 行
            - 店

            【最高優先原則：單一公司名稱直接取值】

            - 如果目前輸入圖片中只出現一個清楚、完整的公司名稱，
            請直接將該公司名稱輸出為「賣方公司名稱」。
            - 此情況不需要再判斷該公司實際屬於買方或賣方。
            - 不可因為圖片中沒有「賣方」標籤就輸出 null。
            - 不可因為無法判斷買方／賣方角色就輸出 null。

            例如目前圖片中只有：

            明揚特殊氣體有限公司

            則直接輸出：

            賣方公司名稱 = 明揚特殊氣體有限公司

            【多個公司名稱時】

            - 只有當目前輸入圖片中同時出現兩個以上公司名稱時，
            才需要依「買方」、「買受人」、「賣方」、
            統編、位置等資訊判斷哪一個是賣方公司名稱。
            - 多個公司名稱時，不可將明確標示為買方的公司輸出為賣方。

            【文字辨識】

            - 必須依照圖片中實際看到的公司名稱逐字輸出。
            - 不可自行補全、改寫或推測圖片中不存在的公司名稱。

            - 特別注意以下容易混淆的形近字：
            「佳／全／住／佺」
            「鈦／鉐」

            - 遇到上述字形時，請依圖片實際筆畫判斷。
        """,
        #"賣方公司名稱": """- 賣方公司名稱：完整賣方公司名稱。
        #- 請逐字的確認，如果有出現類似「晁」「鼎」字，請確認到底是哪一個字""",
        "營業稅稅別判斷": """- 營業稅稅別判斷：請判斷發票上勾選的是「應稅」、「零稅率」或「免稅」。

   【非常重要：禁止推論】
  - 不可以根據「稅額是否大於 0」判斷為應稅。
  - 不可以根據「有營業稅金額」判斷為應稅。
  - 不可以根據「發票類型」、「金額欄位」、「稅法常識」、「一般商業邏輯」推論稅別。
  - 不可以因為看到「營業稅：xxx 元」就輸出「應稅」。
  - 不可以自行補判斷沒有明確顯示的稅別。
  - 只有在圖片中清楚看到稅別勾選框，且能確認哪一個稅別被勾選時，才可以輸出應稅、零稅率或免稅。
  - 如果圖片中沒有顯示稅別勾選框、稅別列被裁切、模糊、遮蔽、看不到勾選位置，或只能看到稅額但看不到勾選框，一律輸出 null。 

  這張台灣統一發票的營業稅稅別列，版型通常由左到右排列如下：

  營業稅 | 應稅 | 應稅的勾選框 | 零稅率 | 零稅率的勾選框 | 免稅 | 免稅的勾選框

  非常重要：
  - 勾選框是在該選項文字的右邊。
  - 勾選框屬於它左邊最近的稅別文字。
  - 不要因為勾選符號出現在某個文字左邊，就判斷成右邊那個文字。
  - 如果看到「應稅  √  零稅率」，代表勾選的是「應稅」，不是「零稅率」。
  - 如果看到「零稅率  √  免稅」，才代表勾選的是「零稅率」。
  - 如果看到「免稅  √」，才代表勾選的是「免稅」。

  判斷規則：
  1. √ / ✓ / 勾選符號在「應稅」右邊，且在「零稅率」左邊，輸出「應稅」。
  2. √ / ✓ / 勾選符號在「零稅率」右邊，且在「免稅」左邊，輸出「零稅率」。
  3. √ / ✓ / 勾選符號在「免稅」右邊，輸出「免稅」。
  4. 如果完全找不到勾選符號，輸出 null。

  只輸出：應稅 或 零稅率 或 免稅 或 null""",
        "明細項目":    """- 明細項目：每筆包含品名、數量、單價、金額，輸出為 JSON 陣列
  注意：
  - 對於容易混淆的中文字，請特別逐字確認，例如：「腦」不要誤判為「磁」、「單」不要誤判為「軍」、「劑」不要誤判為「齊」。
  - 品名請完整輸出，包含前面的料號數字
  - 品名如果換行請合併成完整品名
  - 數量請保留單位（如 40.0 LT）
  - 單價必須保留圖片中數值本身的小數點與小數位。
  - 不要包含合計列、稅額列
  - 數量的表格內可能除了數字外還會有單位，只需辨識出數字即可
  - 數量、單價、金額三者間存在「數量 × 單價 = 金額」的關係，可用來檢查辨識結果，但不能取代逐字辨識：
    1. 請先各自逐字辨識數量、單價、金額三個數字，不要一開始就用計算代替辨識。
    2. 辨識完後，用「數量 × 單價 = 金額」檢查三者是否吻合。
    3. 「金額」一律以圖片上實際辨識到的文字為準，絕對不能用數量、單價反推或修改金額。
    4. 只有「數量」或「單價」其中一個模糊、筆畫不清、或形近字造成不確定時，
       才可以用「金額」與另一個清楚、有把握的數字（單價或數量）反推、修正那一個不確定的數字。
    5. 如果數量、單價都清楚可信、只是與金額對不上（例如有折扣、單位換算等情況），
       請照實輸出圖片上的文字，不要自行修改任何一個。
"""
    }

    json_template = {}

    if failed_fields == ["賣方公司名稱"]:
        json_template = {
            "reason": (
                "<先確認目前圖片中有幾個公司名稱；"
                "若只有一個公司名稱，直接說明該公司名稱；"
                "若有兩個以上，才判斷哪一個是賣方>"
            ),
            "賣方公司名稱": (
                "<若圖片中只有一個公司名稱，直接輸出該名稱；"
                "若有多個公司名稱，輸出判斷出的賣方名稱；"
                "完全找不到公司名稱才輸出null>"
            ),
            "欄位信心值": {
                "賣方公司名稱": "<0~1或null>"
            }
        }
    # elif failed_fields == ["買方公司名稱"]:
    #     json_template = {
    #         "reason": (
    #             "<先確認目前圖片中有幾個公司名稱；"
    #             "若只有一個公司名稱，直接將該名稱視為買方公司名稱；"
    #             "若有兩個以上公司名稱，再根據買方、買受人、統編、位置等資訊"
    #             "判斷哪一個是買方公司名稱；"
    #             "完全找不到公司名稱才判斷為null>"
    #         ),
    #         "買方公司名稱": (
    #             "<若圖片中只有一個公司名稱，直接輸出該公司名稱；"
    #             "若有兩個以上公司名稱，輸出reason中判斷出的買方公司名稱；"
    #             "若完全找不到公司名稱則輸出null；"
    #             "此欄位必須與前面的reason判斷結果一致>"
    #         ),
    #         "欄位信心值": {
    #             "買方公司名稱": "<0~1或null>"
    #         }
    #     }
    elif failed_fields == ["買方公司名稱"]:
        json_template = {
            "reason": (
                "<先辨識圖片中實際可見的買方公司名稱；"
                "若候選名稱未含有燿、耀或華，必須判斷為null>"
            ),
            "買方公司名稱": (
                "<圖片實際可見且文字中含有燿、耀或華的買方公司名稱；"
                "否則輸出null>"
            ),
            "欄位信心值": {
                "買方公司名稱": "<0~1或null>"
            }
        }

    else:
        
        json_template["reason"] = (
            "<簡短說明擷取依據或找不到的原因>"
        ) 
        for field in failed_fields:
            if field == "明細項目":
                json_template[field] = [
                    {
                        "品名": "<原始文字>",
                        "數量": "<原始文字>",
                        "單價": "<數值字串；保留小數點與小數位；或null>",
                        "金額": "<純數字>"
                    }
                ]

            elif field == "營業稅稅別判斷":
                json_template[field] = (
                    "<應稅 或 零稅率 或 免稅 或 null>"
                )

            else:
                json_template[field] = "<值或null>"

        json_template["欄位信心值"] = {
            field: "<0~1或null>"
            for field in failed_fields
        }

        

        

    instructions = "\n".join([
        field_instructions[f] for f in failed_fields if f in field_instructions
    ])

    tax_layout_hint = ""

    if "營業稅稅別判斷" in failed_fields:
        tax_layout_hint = """
    【營業稅稅別判斷特別規則】

    請特別注意台灣發票的稅別勾選框位置。

    本版型不是「勾選框在文字左邊」。
    本版型是「文字在左，勾選框在右」。

    也就是：

    應稅 | 應稅勾選框 | 零稅率 | 零稅率勾選框 | 免稅 | 免稅勾選框

    因此：
    - 如果勾選符號位於「應稅」和「零稅率」之間，答案是「應稅」。
    - 如果勾選符號位於「零稅率」和「免稅」之間，答案是「零稅率」。
    - 如果勾選符號位於「免稅」右邊，答案是「免稅」。

    不要把「√ 零稅率」誤判成零稅率，因為該 √ 可能是左側「應稅」的勾選框。
    """
        
    amount_relation_hint = ""

    amount_fields = {"未稅金額", "稅額", "合計金額","金額大寫中文"}

    if amount_fields.intersection(failed_fields):
        amount_relation_hint = """
    【未稅金額、稅額、合計金額交叉判斷規則】

    未稅金額、稅額與合計金額彼此具有關聯，請不要將三個欄位完全獨立判斷。

    請依照以下順序處理：

    1. 先分別觀察圖片中的：
    - 未稅金額
    - 稅額
    - 合計金額

    2. 判斷哪一個欄位在圖片中最清楚、最可信，將該欄位作為主要依據。

    3. 金額應符合以下基本關係：

    合計金額 = 未稅金額 + 稅額

    【未稅金額與合計金額相同時的重新確認】

    若圖片辨識結果出現：

    未稅金額 = 合計金額
    且
    稅額 > 0

    則此結果視為欄位誤讀；當稅額 > 0 時，最終 JSON 禁止輸出「未稅金額 = 合計金額」。
    必須重新查看「銷售額 / 銷售額合計 / 銷售額(應稅)」直接對應的金額，不可把「總計」的數值填入未稅金額。

    必須重新分別查看：
    - 「銷售額合計 / 銷售額 / 銷售額(應稅)」旁的數字
    - 「總計」旁的數字

    兩個欄位必須獨立重新辨識，
    不可把「總計」的數字複製成未稅金額。

    例如：

    銷售額合計 = 3060
    營業稅 = 153
    總計 = 3213

    則必須輸出：

    未稅金額 = 3060
    稅額 = 153
    合計金額 = 3213

    不可因為「總計 3213」較清楚，
    就把未稅金額也輸出為 3213。

    若重新查看後仍無法確認未稅金額，
    則未稅金額輸出 null，
    不可直接沿用合計金額。

    【重要】
    重新查看只能執行一次。

    重新查看完成後，必須立即做出最終結果，不可再次觸發本規則，
    也不可在 reason 中反覆重新討論同一組數字。

    若重新查看後仍然得到：
    未稅金額 = 合計金額
    且
    稅額 > 0

    則：
    - 不可繼續重複重新確認。
    - 若未稅金額仍無法從圖片可靠確認，未稅金額輸出 null。
    - 稅額與合計金額若圖片欄位清楚，可保留圖片直接辨識值。
    - 完成後立即輸出 JSON。

    4. 如果可以確認為一般 5% 應稅發票，只能使用以下方式驗證：

    預期稅額 = 未稅金額 × 0.05

    若稅額以整數顯示，需將計算結果四捨五入至整數後，
    再與圖片直接辨識到的稅額比較。

    5. 禁止使用「稅額 ÷ 0.05」反推未稅金額。

    若圖片中的未稅金額無法可靠辨識，
    不可僅根據稅額自行產生未稅金額。

    6. 例如：

    圖片未稅金額 = 4590
    圖片稅額 = 230

    4590 × 0.05 = 229.5
    四捨五入 = 230

    因此 5% 稅率關係成立，
    保留圖片直接辨識的未稅金額 4590 與稅額 230。

    6-1.【強制檢查，即使加總關係成立也一定要做】
    「合計金額 = 未稅金額 + 稅額」這個加總關係只能證明三個數字彼此「內部一致」，
    不能證明這三個數字本身是正確的（三個數字可能同時抄錯、位數錯位或多一位數，
    只要抄錯得剛好還是能加總起來）。

    因此，只要稅額有值（大於 0）且稅額本身辨識信心足夠高，
    無論未稅金額與合計金額的加總關係是否已經成立，都必須「另外」執行以下第二重檢查：

    預期稅額 = 未稅金額 × 0.05

    若發票稅額以整數顯示，必須考慮四捨五入：

    - 將「未稅金額 × 0.05」四捨五入至整數後，
    與圖片直接辨識到的稅額比較。

    - 若四捨五入後與圖片稅額相同，
    則視為 5% 稅率關係成立，
    必須保留圖片直接辨識的未稅金額、稅額與合計金額，
    不可再使用「稅額 ÷ 0.05」反推未稅金額。

    - 只有當四捨五入後仍與圖片稅額不符時，
    才視為 5% 稅率關係不成立，再考慮其他推算方式。

        合計金額 = 推算未稅金額 + 稅額

    

    6-2.【禁止反推未稅金額】

    禁止使用：

    未稅金額 = 稅額 ÷ 0.05

    來產生或覆寫未稅金額。

    5% 稅率只能用於驗證圖片直接辨識到的未稅金額：

    預期稅額 = 未稅金額 × 0.05

    若四捨五入後與圖片稅額一致：
    → 視為 5% 稅率驗證通過。

    若不一致：
    → 只代表目前辨識結果存在疑問。
    → 應重新查看圖片中的未稅金額欄位。
    → 不可直接使用稅額反推新的未稅金額。

    7. 如果圖片直接辨識出的某個金額，與另外兩個欄位的數學關係不一致，請比較：

    - 哪個欄位字體最清楚
    - 哪個欄位信心值最高
    - 是否符合 5% 稅率（見第 6-1、6-2 點的強制檢查，優先權高於單純的加總關係）
    - 是否符合未稅金額加稅額等於合計金額

    8. 若其中一個直接辨識值疑似筆誤、模糊或多辨識一個數字，可以採用整體數學關係較合理的推算值；
       當加總關係與 5% 稅率關係互相衝突時，以 5% 稅率關係（第 6-1、6-2 點）為準。

    9. 不可在沒有任何可靠金額依據時自行猜測。

    10. 如果無法確認本張發票適用 5% 稅率，則不可只依 5% 稅率推算；此時只能使用：

        合計金額 = 未稅金額 + 稅額

    11. 欄位信心值請依來源設定：

        - 圖片清楚直接辨識：0.90～1.00
        - 根據兩個清楚金額交叉推算：0.75～0.89
        - 只根據稅額與 5% 稅率推算：0.65～0.80
        - 無法可靠判斷：輸出 null

    12. reason 必須清楚說明：
        - 哪個金額是圖片直接辨識
        - 哪個金額是利用公式推算
        - 使用了哪一條公式
        - 第 6-1、6-2 點的強制 5% 檢查結果（相符或不符，若不符是否已改用推算值）
     13. 金額大寫中文必須與合計金額表示完全相同的數值。

        金額大寫中文 = 合計金額轉換成繁體中文財務大寫

    14. 金額大寫中文只能參考「合計金額」進行轉換，
        不可以參考未稅金額或稅額直接產生。

    15. 例如：

        未稅金額 = 34200
        稅額 = 1710
        合計金額 = 35910

        則：

        金額大寫中文 = 參萬伍仟玖佰壹拾元整

        不可以輸出：
        參萬肆仟貳佰元整

        因為參萬肆仟貳佰元整代表的是未稅金額，不是合計金額。

    16. 如果圖片直接辨識出的金額大寫中文與合計金額不一致：

        - 合計金額清楚且可信時，使用合計金額轉換後的中文財務大寫。
        - 合計金額不清楚時，不可任意推算，輸出 null。

    17. reason 必須說明金額大寫中文是：

        - 圖片直接辨識且與合計金額一致；或
        - 根據合計金額轉換產生。
    """

    amount_fields = {
        "未稅金額",
        "稅額",
        "合計金額",
        "金額大寫中文",
    }

    amount_priority_instruction = ""

    if amount_fields.intersection(failed_fields):
        amount_priority_instruction = """
    【金額欄位來源優先規則－最高優先】

    本系統對金額欄位的定義固定為：

    - 未稅金額 → 「銷售額合計」、「銷售額」、「銷售額（應稅）」、「銷售額(應稅)」
    - 稅額 → 「營業稅」
    - 合計金額 → 「總計」

    如果圖片同時存在：

    - 原幣金額
    - 原幣營業稅
    - 原幣金額(含稅)
    - 幣別
    - 匯率

    這些屬於「原幣 / 外幣 / 匯率換算資訊」，
    不是本次要擷取的未稅金額、稅額、合計金額。

    禁止使用：

    原幣金額 → 未稅金額
    原幣營業稅 → 稅額
    原幣金額(含稅) → 合計金額

    即使這三個原幣數值之間：

    原幣金額 + 原幣營業稅 = 原幣金額(含稅)

    完全成立，也禁止選用。

    【判斷優先順序】

    1. 欄位標題與語意
    2. 欄位所在的主要結算區域
    3. 圖片直接看到的數值
    4. 數學關係驗證

    數值應依目標欄位標題或可辨識的欄位語意與版面關係判斷，不要求欄位標題每個字都完全清晰。
    數學關係只能驗證，不可用來決定數值屬於哪個欄位。
    """

    amount_common_instruction = """
【金額欄位辨識共通規則】

- 不可假設發票具有固定版型、固定位置或固定欄位名稱。
- 必須依圖片中的文字語意、相對位置與數值對應關係，判斷每個金額屬於哪個欄位。
- 欄位名稱可能因發票格式而有不同寫法、縮寫、空格、標點或字距，只要語意相同即可視為同一欄位。
- 常見欄位名稱只能作為提示，不是唯一允許的名稱。
- 必須先從圖片判斷欄位歸屬，再讀取該欄位對應的數值。
- 數學關係只能用於驗證辨識結果，不可用數學關係決定某個數字屬於哪個欄位。
- 若圖片同時存在本幣結算資訊與原幣／外幣／匯率換算資訊，優先使用發票主要結算使用的本幣金額。
- 找不到可靠的文字語意與數值對應關係時才輸出 null。
"""

    vlm_source_guard = """
    【圖片資料來源限制－最高優先】

    - 最終輸出的所有欄位值，只能來自本次提供的圖片中實際可見的內容。
    - Prompt 中的規則、說明、範例、JSON 範本、placeholder，都不是圖片資料。
    - 禁止把 Prompt 範例中的公司名稱、統編、日期、金額、品名、數量、單價或其他文字直接當成辨識結果。
    - 若某個值只存在於 Prompt，而圖片中實際看不到，禁止輸出該值。
    - 不可根據 Prompt 範例補值、猜值或產生圖片中不存在的內容。
    - 無法從圖片本身確認時，必須輸出 null。
    """

    if failed_fields == ["發票日期"]:
        prompt = f"""
        {vlm_source_guard}

        請只辨識這張圖片中實際可見的發票日期。

        【規則】
        - 尋找具有「年、月、日」完整意義的日期文字。
        - 日期可能使用民國年或西元年。
        - 若為4位數年份且第一位為0，仍可能是民國年前導零，不可忽略。
        - 分隔符號可能是 /、- 或 .。
        - 日期後方若有其他文字或發票號碼，不影響日期本身的判斷。
        - 輸出時保留圖片中的原始日期格式。
        - 找不到才輸出 null。
        - 不可使用 Prompt 中的範例內容作為答案。

        只輸出：
        {{
        "reason": "<簡短原因>",
        "發票日期": "<圖片中實際看到的日期或null>",
        "欄位信心值": {{
            "發票日期": "<0~1或null>"
        }}
        }}
        """
    else:
        prompt = f"""這是一張台灣統一發票圖片，請仔細閱讀圖片內容，擷取以下欄位。

        {vlm_source_guard}

        {amount_common_instruction}

        {tax_layout_hint}



        {instructions}

        請用以下 JSON 格式回答，找不到的欄位填 null，不要加任何多餘說明：
        {json.dumps(json_template, ensure_ascii=False, indent=2)}"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=2048,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM保底] 回應:\n{content}\n")
        print("VLLM_MODEL :",VLLM_MODEL)

        if not content:
            print(f"⚠️  [VLM保底] 回應為空")
            return {}

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print(f"⚠️  [VLM保底] 找不到 JSON")
            return {}

        # result = json.loads(json_match.group())
        result = json.loads(repair_json(json_match.group()))

        # 數字欄位清理
        for key in ["未稅金額", "稅額", "合計金額"]:
            if result.get(key):
                result[key] = str(result[key]).replace(",", "").strip()

        # ============================================================
        # 明細項目清理
        # 只有本次真的要求辨識「明細項目」才處理，
        # 避免其他欄位的 result 被額外塞入 "明細項目": []
        # ============================================================

        if "明細項目" in requested_fields:

            items = result.get(
                "明細項目"
            )

            if isinstance(items, list):

                for item in items:

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    for key in [
                        "單價",
                        "金額"
                    ]:
                        if item.get(key):
                            item[key] = (
                                str(item[key])
                                .replace(",", "")
                                .strip()
                            )

                result["明細項目"] = items

        result["__field_confidence__"] = _normalize_field_confidence_map(
            result.get("欄位信心值"),
            failed_fields
        )
        result.pop("欄位信心值", None)
        # reason 保留在 result 中供呼叫端使用
        if "reason" not in result:
            result["reason"] = None

        # 圖片完全沒有疑似中文財務大寫時，
        # 只覆寫此欄位，不改動其他欄位與原 prompt 邏輯。
        if (
            "金額大寫中文" in failed_fields
            and amount_uppercase_gate_found is False
        ):
            result["金額大寫中文"] = None
            result["__field_confidence__"]["金額大寫中文"] = None

        
         # 合併獨立 VLM 辨識出的明細項目。
        detail_items = detail_vlm_result.get(
            "明細項目"
        )

        if (
            isinstance(detail_items, list)
            and len(detail_items) > 0
        ):
            result["明細項目"] = detail_items

            result.setdefault(
                "__field_confidence__",
                {}
            )

            detail_confidence = (
                detail_vlm_result
                .get(
                    "__field_confidence__",
                    {}
                )
                .get("明細項目")
            )

            result[
                "__field_confidence__"
            ]["明細項目"] = detail_confidence

            if detail_vlm_result.get(
                "detail_reason"
            ):
                result["detail_reason"] = (
                    detail_vlm_result[
                        "detail_reason"
                    ]
                )

        if (
            "備註" in failed_fields
            and remark_gate_found is False
        ):
            result["備註"] = None

            result.setdefault(
                "__field_confidence__",
                {}
            )

            result[
                "__field_confidence__"
            ]["備註"] = None

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  [VLM保底] JSON 解析失敗：{e}")
        return {}
    except Exception as e:
        print(f"⚠️  [VLM保底] 呼叫失敗：{e}")
        return {}


def extract_invoice_number_from_image(image_pil: Image.Image, known_prefix: str = None) -> dict:
    """專用 VLM：只辨識發票號碼，避免通用欄位提示分散注意力。"""
    prefix_hint = ""
    if known_prefix:
        prefix_hint = f"\n已知字軌英文字母可能是「{known_prefix}」。如果你只看到 8 位數字，請和 {known_prefix} 合併成完整發票號碼。"

    prompt = f"""請只判斷這張圖片中的發票號碼。

發票號碼格式：
- 2 個英文字母 + 8 個數字

請特別看圖片左上方或上方偏左的發票字軌區。
如果看到英文字母（例如 ZX），並且右側或附近斜線底紋區有 8 位數字，請合併成完整發票號碼。{prefix_hint}

不要輸出：
- 買方統編
- 賣方統編
- 日期
- 電話
- 金額
- 地址

請只回傳 JSON，不要加任何說明：
{{
  "發票號碼": "<2個英文字母+8個數字或null>",
    "reason": "<簡短原因>",
    "信心值": "<0~1或null>"
}}"""

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=256,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM發票號碼專用] 回應:\n{content}\n")
        

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print("⚠️  [VLM發票號碼專用] 找不到 JSON")
            return {"發票號碼": None, "reason": "VLM回應非JSON"}

        # result = json.loads(json_match.group())
        result = json.loads(repair_json(json_match.group()))
        
        return {
            "發票號碼": result.get("發票號碼"),
            "reason": result.get("reason", ""),
            "__field_confidence__": {"發票號碼": _to_confidence_01(result.get("信心值"))} if _to_confidence_01(result.get("信心值")) is not None else {}
        }

    except Exception as e:
        print(f"⚠️  [VLM發票號碼專用] 呼叫失敗：{e}")
        return {"發票號碼": None, "reason": f"VLM呼叫失敗：{e}"}


def extract_tax_type_from_image(image_pil: Image.Image) -> dict:
    """專用 VLM：對裁切後的稅別區塊圖片做營業稅稅別判斷。"""
    prompt = """
    請判斷這張圖片中的營業稅稅別勾選結果。

    【最高優先原則：只能根據圖片中的實際勾選記號判斷】

    - 不可以根據「稅額是否大於 0」判斷為應稅。
    - 不可以根據「有營業稅金額」判斷為應稅。
    - 不可以根據發票類型、金額、稅法常識或一般商業邏輯推論稅別。
    - 不可以因為看到「營業稅：xxx 元」就輸出「應稅」。
    - 只能根據圖片中實際可見的勾選記號，例如：
    V
    v
    √
    ✓
    X
    ×
    ●
    ■
    或其他明顯填入的選取記號。

    ==================================================
    【非常重要：不要求一定存在方形勾選框】
    ==================================================

    台灣發票的稅別版型不一定會印出一個獨立的小方形 checkbox。

    以下兩種版型都可能存在：

    【版型 A：勾選符號在文字右側】

    應稅   V   零稅率       免稅

    此時：
    V 位於「應稅」右側、「零稅率」左側
    → 判斷為「應稅」。

    【版型 B：表格上下排列】

    應稅      零稅率      免稅
    V

    此時：
    V 位於「應稅」欄位正下方的儲存格
    → 判斷為「應稅」。

    因此：

    「沒有看到方形勾選框」
    不等於
    「沒有勾選」。

    只要可以清楚看到 V、√、✓、X、× 等選取記號，
    並且可以依其位置確認它屬於哪一個稅別欄位，
    就必須判斷該稅別。

    ==================================================
    【表格欄位判斷－最高優先】
    ==================================================

    如果圖片中版面類似：

    應稅       零稅率       免稅
    V

    請將：

    「應稅」、「零稅率」、「免稅」

    視為三個水平欄位的表頭。

    判斷勾選符號時，
    必須同時比較：

    1. 勾選符號的水平 X 位置。
    2. 三個稅別表頭的水平 X 位置。
    3. 勾選符號是否位於某個表頭正下方的儲存格。

    若勾選符號 V 的水平中心位置
    落在「應稅」欄位範圍內，
    即使 V 位於「應稅」文字的下方，
    仍必須判斷為：

    "應稅"

    同理：

    V 位於「零稅率」欄位下方
    → "零稅率"

    V 位於「免稅」欄位下方
    → "免稅"

    ==================================================
    【不要把表格線誤認成沒有勾選】
    ==================================================

    圖片中可能只有：

    - 稅別文字
    - 表格框線
    - V / √ / ✓ / X 等符號

    而沒有獨立的小方框。

    這是正常的版型。

    只要選取記號清楚存在，
    不可因為「沒有獨立方框」而輸出 null。

    ==================================================
    【什麼情況才輸出 null】
    ==================================================

    只有以下情況才輸出 null：

    1. 圖片中完全找不到任何 V、√、✓、X、× 或其他選取記號。
    2. 有看到疑似記號，但無法確認屬於應稅、零稅率或免稅哪一欄。
    3. 稅別區域被裁切，導致無法知道記號對應哪個選項。
    4. 圖片過度模糊或遮蔽，無法可靠辨識選取位置。

    禁止因為：

    「沒有看到方形 checkbox」

    單獨作為輸出 null 的理由。

    ==================================================
    【本案例類型】
    ==================================================

    如果圖片顯示：

    應稅       零稅率       免稅
    V

    其中 V 明確位於「應稅」欄位正下方，

    則必須輸出：

    {
    "營業稅稅別判斷": "應稅",
    "reason": "可見 V 記號位於應稅欄位正下方，因此判斷應稅",
    "信心値": 0.95
    }

    禁止輸出 null。

    ==================================================
    【禁止推論】
    ==================================================

    即使圖片另外顯示：

    營業稅金額
    3124

    也不可利用該金額判斷應稅。

    判斷依據只能是：

    「實際可見的勾選記號 + 它相對於稅別欄位的位置」。

    請只回傳 JSON，不要加任何說明：

    {
    "營業稅稅別判斷": "應稅 或 零稅率 或 免稅 或 null",
    "勾選符號": "實際看到的 V、v、√、✓、X、×、●、■，若沒有則 null",
    "reason": "簡短說明實際看到的勾選記號及其位置",
    "信心値": "<0~1>"
    }
    """

    try:
        b64_image = image_to_base64(image_pil)

        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=2048,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM稅別專用] 回應:\n{content}\n")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print("⚠️  [VLM稅別專用] 找不到 JSON")
            return {"營業稅稅別判斷": None, "reason": "VLM回應非JSON"}

        # result = json.loads(json_match.group())
        result = json.loads(repair_json(json_match.group()))

        tax_value = result.get("營業稅稅別判斷")
        selection_mark = result.get("勾選符號")

        # ✅ 只有真的回傳合法勾選符號，才允許判斷稅別
        VALID_SELECTION_MARKS = {
            "V", "v",
            "√", "✓",
            "X", "x", "×",
            "●", "■"
        }

        if selection_mark not in VALID_SELECTION_MARKS:
            print(
                "[VLM稅別專用] 未偵測到有效勾選符號，"
                f"勾選符號={selection_mark!r}，"
                "強制稅別設為 None"
            )

            tax_value = None

            return {
                "營業稅稅別判斷": None,
                "reason": "未看到實際勾選符號",
                "__field_confidence__": {}
            }

        return {
            "營業稅稅別判斷": tax_value,
            "reason": result.get("reason", ""),
            "__field_confidence__": {
                "營業稅稅別判斷": _to_confidence_01(
                    result.get("信心値")
                )
            } if _to_confidence_01(
                result.get("信心値")
            ) is not None else {}
        }

    except Exception as e:
        print(f"⚠️  [VLM稅別專用] 呼叫失敗：{e}")
        return {"營業稅稅別判斷": None, "reason": f"VLM呼叫失敗：{e}"}


def detect_total_ntd_text(image_pil: Image.Image) -> dict:
    """用 VLM 判斷發票是否出現「總計新臺幣/總計新台幣」字樣。"""
    prompt = """
請判斷這張台灣統一發票圖片中，
是否出現「總計新臺幣」或「總計新台幣」字樣。

【判斷結果】

present：
- 可以清楚辨識出「總計新臺幣」或「總計新台幣」。

absent：
- 可以明確確認只有「總計」，附近沒有「新臺幣／新台幣」文字。
- 且沒有看到疑似屬於「總計新臺幣」區域的金額數值或殘缺文字。

uncertain：
- 「總計」附近疑似還有「新臺幣／新台幣」文字，
  但因印刷模糊、斷字、缺墨、低解析度或部分文字看不清楚，
  無法可靠確認完整字樣。
- 若可以看到「總計」相關的金額數值，
  但無法清楚辨識該金額附近是否具有「新臺幣／新台幣」字樣，
  必須判定為 uncertain，不可判定為 absent。

【重要】
- 看不清楚不等於不存在。
- 只有能明確確認不存在時，才可以輸出 absent。
- 若疑似存在但無法完整辨識，必須輸出 uncertain。
- 不可根據 Prompt 中的文字當成圖片內容。
- 「文字沒有辨識清楚」不代表文字不存在。
- 若圖片中存在疑似對應「總計新臺幣」的金額數值，
  但文字標示模糊或無法確認，必須輸出 uncertain。

請只回傳 JSON：

{
  "status": "present 或 absent 或 uncertain",
  "matched_text": "實際看到或疑似看到的文字，否則null",
  "reason": "簡短原因"
}
"""

    try:
        b64_image = image_to_base64(image_pil)
        response = client.chat.completions.create(
            model=VLLM_LLM_MODEL2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=256,
            temperature=0.0,
            extra_body={
                "max_soft_tokens": 1120
            }
        )

        content = (response.choices[0].message.content or "").strip()
        print(f"[VLM總計新臺幣判斷] 回應:\n{content}\n")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return {
                "status": "uncertain",
                "matched_text": None,
                "reason": "VLM回應非JSON"
            }

        # result = json.loads(json_match.group())
        result = json.loads(repair_json(json_match.group()))

        status = str(result.get("status") or "").strip().lower()

        if status not in {"present", "absent", "uncertain"}:
            status = "uncertain"

        reason = result.get("reason", "")

        # ============================================================
        # status=uncertain 時，再用「金額大寫中文存在性」輔助判斷
        # ============================================================
        if status == "absent":
            amount_uppercase_exists, amount_uppercase_evidence = (
                _has_suspected_financial_uppercase_amount(
                    image_pil
                )
            )

            print(
                "[VLM總計新臺幣判斷][金額大寫輔助] "
                f"exists={amount_uppercase_exists}, "
                f"evidence={amount_uppercase_evidence!r}"
            )

            if amount_uppercase_exists is False:
                # 總計新臺幣字樣無法確認，
                # 且圖片中也沒有金額大寫中文
                status = "absent"

                reason = (
                    f"{reason}；"
                    "總計新臺幣字樣無法確認，且未偵測到金額大寫中文，"
                    "因此判定為 absent"
                )

            else:
                # 有金額大寫中文存在，
                # 不可因為總計新臺幣字樣模糊就直接判 absent
                status = "uncertain"

                reason = (
                    f"{reason}；"
                    "偵測到金額大寫中文，因此維持 uncertain"
                )

        return {
            "status": status,
            "matched_text": result.get("matched_text"),
            "reason": reason
        }

    except Exception as e:
        print(f"⚠️  [VLM總計新臺幣判斷] 呼叫失敗：{e}")
        return {
            "status": "uncertain",
            "matched_text": None,
            "reason": f"VLM呼叫失敗：{e}"
        }

# =========================
# VLM 判斷是否有多張發票
# =========================
def detect_multi_invoice(image_input, save_debug_path: str = None) -> dict:
    """
    使用 VLM 判斷圖片中是否包含多張發票
    
    Args:
        image_input:      圖片路徑（str）或 PIL Image
        save_debug_path:  若不為 None，將圖片存到此路徑供 debug 用
    Returns:
        dict: {
            "has_multiple_invoices": bool,
            "invoice_count":         int,
            "confidence":            float,  # 0.0 ~ 1.0
            "reason":                str,
            "raw_response":          str
        }
    """
    # 準備圖片
    if isinstance(image_input, str):
        b64 = image_to_base64(image_input)
    elif isinstance(image_input, Image.Image):
        b64 = image_to_base64(image_input)
        if save_debug_path:
            image_input.save(save_debug_path)
            print(f"[DEBUG] 已儲存圖片：{save_debug_path}")
    else:
        raise ValueError("image_input 必須是圖片路徑或 PIL Image")

    # Prompt
    prompt = """請仔細分析這張圖片，判斷圖片中包含幾張發票（電子發票或紙本發票）。

判斷依據：
- 每張發票通常有獨立的發票號碼（如 YW12345678）
- 每張發票有獨立的買方/賣方資訊
- 每張發票有獨立的金額合計

請用以下 JSON 格式回答，不要加任何多餘的說明：
{
  "invoice_count": <數字>,
  "has_multiple_invoices": <true 或 false>,
  "confidence": <0.0 到 1.0 之間的數字，代表判斷信心度>,
  "reason": "<簡短說明判斷原因>"
}"""

    # 呼叫 VLM
    response = client.chat.completions.create(
        model=VLLM_LLM_MODEL2,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=256,
        temperature=0.0,   # 判斷性任務用 0，減少隨機性
        extra_body={
            "max_soft_tokens": 1120
        }
    )

    raw_text = response.choices[0].message.content.strip()
    print(f"[VLM 回應] {raw_text}")

    # 解析 JSON 回應
    try:
        # 有時 VLM 會在 JSON 外面包 markdown ```json ... ```
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            # result = json.loads(json_match.group())
            result = json.loads(repair_json(json_match.group()))
        else:
            # result = json.loads(raw_text)
            result = json.loads(repair_json(raw_text))

        # 確保 confidence 是 float
        if "confidence" in result:
            c = result["confidence"]
            if isinstance(c, str):
                result["confidence"] = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(c.lower(), 0.5)
            else:
                result["confidence"] = float(c)
        result["raw_response"] = raw_text
        return result

    except json.JSONDecodeError:
        # 解析失敗時，嘗試從文字判斷
        has_multiple = any(kw in raw_text for kw in ["多張", "兩張", "2張", "multiple", "more than one"])
        return {
            "has_multiple_invoices": has_multiple,
            "invoice_count":         -1,       # 無法確定
            "confidence":            0.3,
            "reason":                "JSON解析失敗，依關鍵字推斷",
            "raw_response":          raw_text
        }

# =========================
# 批次處理資料夾內所有圖片/PDF
# =========================
def process_folder(input_dir: str, output_json: str = "multi_invoice_detection.json"):
    """
    批次處理資料夾內所有 PDF 和圖片，輸出偵測結果
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    results = []

    files = sorted(os.listdir(input_dir))
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        pages = []

        if ext == ".pdf":
            print(f"\n處理 PDF：{filename}")
            pdf_pages = convert_from_path(filepath, dpi=300, poppler_path=POPPLER_PATH)
            pages = [(f"{filename}_page{i+1}", p) for i, p in enumerate(pdf_pages)]

        elif ext in IMAGE_EXTENSIONS:
            print(f"\n處理圖片：{filename}")
            img = Image.open(filepath).convert("RGB")
            pages = [(filename, img)]

        else:
            continue

        for page_name, page_img in pages:
            print(f"  偵測：{page_name}")
            detection = detect_multi_invoice(page_img)
            results.append({
                "file":    page_name,
                "result":  detection
            })

            status = "⚠️  多張發票" if detection.get("has_multiple_invoices") else "✅  單張發票"
            print(f"  → {status}（共 {detection.get('invoice_count')} 張，信心度: {detection.get('confidence')}）")
            print(f"     原因: {detection.get('reason')}")

    # 輸出 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果已儲存：{output_json}")
    return results

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    # 單張圖片測試
    test_image = "file/png_output/page_5_page1.png"
    ext = os.path.splitext(test_image)[1].lower()

    if ext == ".pdf":
        pages = convert_from_path(test_image, dpi=700, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages):
            print(f"\n===== 第 {i+1} 頁 =====")
            result = detect_multi_invoice(page, save_debug_path=f"jpg_pages/detect_page{i+1}.png")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = detect_multi_invoice(test_image)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 批次處理整個資料夾（取消下方註解）
    # process_folder("file/split_output/scan-00003")