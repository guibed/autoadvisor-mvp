import os, re, json, requests
from datetime import datetime

LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise SystemExit("Missing LLM_API_KEY in environment.")

# Adjust if you use a different provider
LLM_URL = "https://api.mistral.ai/v1/chat/completions"
LLM_MODEL = "mistral-small-latest"

EXTRACTION_SYSTEM = """You are an automotive data extractor.
Return ONLY a SINGLE JSON object with this schema:
{
  "brand": "", "model": "", "year": 0, "mileage_km": 0, "price_eur": 0,
  "fuel": "", "transmission": "", "trim": "", "options": [],
  "service_history": "", "known_issues": [], "seller_notes": "", "text": ""
}
Rules:
- Output ONLY valid JSON (no prose, no markdown fences).
- If unknown, use null (not empty strings).
- mileage_km and price_eur must be integers (e.g., 98000, 11500).
- Normalize EU formats: "98 000 km" → 98000, "11 500€" → 11500.
- Preserve the exact ad text in the field "text".
- Do not invent facts that aren’t in the ad.
"""

WCS_URL = os.getenv("WCS_URL")
WCS_API_KEY = os.getenv("WCS_API_KEY")

def _coerce_number(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x)
    # remove currency, km, thousands sep, commas
    s = re.sub(r"[€\s,kmKM]", "", s)
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s.isdigit() else None

def _extract_json(text):
    """
    Try to parse JSON; if the model wrapped it in code fences or added text,
    extract the first {...} block.
    """
    text = text.strip()
    # Remove ```json fences if present
    fence = re.match(r"```(?:json)?\s*(.*)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    # Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: get first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from model output.")

def extract_listing(ad_text):
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": ad_text}
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    resp = requests.post(LLM_URL, headers=headers, json=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response format: {resp.text}") from e

    data = _extract_json(content)

    # Normalize & coerce fields (as before) …
    out = {
        "brand": (str(data.get("brand")).title().strip() if data.get("brand") else None),
        "model": (str(data.get("model")).strip() if data.get("model") else None),
        "year": _coerce_number(data.get("year")),
        "mileage_km": _coerce_number(data.get("mileage_km")),
        "price_eur": _coerce_number(data.get("price_eur")),
        "fuel": data.get("fuel"),
        "transmission": data.get("transmission"),
        "trim": data.get("trim"),
        "options": data.get("options") or [],
        "service_history": data.get("service_history"),
        "known_issues": data.get("known_issues") or [],
        "seller_notes": data.get("seller_notes"),
        "text": data.get("text") or ad_text,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    return out

def insert_listing_to_weaviate(listing: dict):
    if not WCS_URL or not WCS_API_KEY:
        raise SystemExit("Missing WCS_URL or WCS_API_KEY in environment.")
    payload = {"class": "Listing", "properties": listing}
    r = requests.post(
        f"{WCS_URL}/v1/objects",
        headers={"Authorization": f"Bearer {WCS_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if r.status_code not in (200, 202):
        raise RuntimeError(f"Weaviate insert failed {r.status_code}: {r.text}")
    return r.json()

if __name__ == "__main__":
    print("Paste a car listing text (end with Enter):")
    ad = input().strip()
    info = extract_listing(ad)
    print("\nExtracted JSON:")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    res = insert_listing_to_weaviate(info)
    print("\nWeaviate insert OK:", res.get("id", "no-id"))
