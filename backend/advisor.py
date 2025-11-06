import os, re, csv, json, requests
from datetime import datetime
from typing import Dict, List, Any

LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise SystemExit("Missing LLM_API_KEY in env.")
LLM_URL = "https://api.mistral.ai/v1/chat/completions"
LLM_MODEL = "mistral-tiny-latest"   # adjust if needed

KB_CSV = os.getenv("KB_CSV", "data/knowledge_base.csv")

# ---------- utilities ----------

def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence = re.match(r"```(?:json)?\s*(.*)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from model output.")

def _parse_year_range(r: str):
    if not r: return None, None
    m = re.match(r"^\s*(\d{4})\s*[-–]\s*(\d{4})\s*$", r)
    if m:
        return int(m.group(1)), int(m.group(2))
    # single year or malformed → treat as broad
    try:
        y = int(r.strip())
        return y, y
    except:
        return None, None

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def load_kb_for_listing(listing: Dict[str, Any]) -> List[str]:
    """Load KB rows whose brand/model match, and year is in range (if possible). Return list of text chunks."""
    brand = _normalize(listing.get("brand"))
    model = _normalize(listing.get("model"))
    year = listing.get("year")
    chunks = []
    with open(KB_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rb, rm = _normalize(row.get("brand")), _normalize(row.get("model"))
            if brand and rb and brand != rb: 
                continue
            if model and rm and model not in rm:  # allows "golf 7" match
                continue
            y1, y2 = _parse_year_range(row.get("year_range", ""))
            if isinstance(year, int) and y1 and y2 and not (y1 <= year <= y2):
                continue
            txt = row.get("text", "").strip()
            topic = row.get("topic", "").strip()
            if txt:
                chunks.append(f"[{row.get('brand','')}/{row.get('model','')}/{row.get('year_range','')} • {topic}] {txt}")
    # fall back to whole KB if nothing matched (still better than empty)
    if not chunks:
        with open(KB_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt = row.get("text", "").strip()
                topic = row.get("topic", "").strip()
                if txt:
                    chunks.append(f"[{row.get('brand','')}/{row.get('model','')}/{row.get('year_range','')} • {topic}] {txt}")
    return chunks[:8]  # keep prompt short

# ---------- prompting ----------

SYSTEM_PROMPT = """You are AutoAdvisor, a cautious, transparent used-car assistant.
You will produce a single JSON object with this schema:
{
  "pros": [], "cons": [], 
  "price_assessment": "Under|At|Over|Unknown",
  "mechanical_risk": "Low|Medium|High",
  "info_completeness": "Low|Medium|High",
  "questions_to_ask": [],
  "summary": "",
  "citations": []  // direct quotes from the ad text that justify key points
}
Rules:
- Output ONLY valid JSON (no extra text).
- Ground PROS/CONS in the ad text; include short direct quotes in `citations`.
- Use the KB only for general typical issues and inspection tips (don’t invent facts about this specific car).
- If you lack info for price, set price_assessment to "Unknown".
- Keep `summary` ≤ 180 words, friendly and actionable, with clear caveats.
"""

def build_user_message(listing: Dict[str, Any], kb_chunks: List[str]) -> str:
    return (
        "Listing facts (JSON):\n"
        f"{json.dumps({k: listing.get(k) for k in ['brand','model','year','mileage_km','price_eur','fuel','transmission','trim','options','service_history','known_issues','seller_notes']}, ensure_ascii=False)}\n\n"
        "Ad text:\n"
        f"<<<{listing.get('text','').strip()}>>>\n\n"
        "Retrieved knowledge (general reliability & inspection notes):\n"
        + "\n- " + "\n- ".join(kb_chunks)
        + "\n\nTasks:\n"
        "1) Extract PROS and CONS grounded in the ad text.\n"
        "2) Price assessment (Under/At/Over/Unknown) + short rationale.\n"
        "3) Mechanical risk (Low/Medium/High) based on model-year typical issues and ad evidence.\n"
        "4) Info completeness (Low/Medium/High) considering missing key details.\n"
        "5) Top 5 questions to ask the seller.\n"
        "6) Provide concise `summary` (≤180 words).\n"
    )

def advise(listing: Dict[str, Any]) -> Dict[str, Any]:
    kb_chunks = load_kb_for_listing(listing)
    user_content = build_user_message(listing, kb_chunks)

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    resp = requests.post(LLM_URL, headers=headers, json=data, timeout=45)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]
    out = _extract_json(content)

    # Minimal normalization / defaults
    out.setdefault("pros", [])
    out.setdefault("cons", [])
    out.setdefault("questions_to_ask", [])
    out.setdefault("citations", [])
    out.setdefault("price_assessment", "Unknown")
    out.setdefault("mechanical_risk", "Medium")
    out.setdefault("info_completeness", "Medium")
    out.setdefault("summary", "")

    return out

# ---------- CLI: read listing JSON file or stdin ----------

if __name__ == "__main__":
    import sys
    if sys.stdin.isatty():
        path = input("Path to extracted listing JSON (from Step 5) or paste JSON:\n").strip()
        if path.endswith(".json") and os.path.exists(path):
            listing = json.load(open(path, "r", encoding="utf-8"))
        else:
            listing = json.loads(path)
    else:
        listing = json.load(sys.stdin)

    result = advise(listing)
    print(json.dumps(result, indent=2, ensure_ascii=False))
