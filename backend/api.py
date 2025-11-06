import os, json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# local modules
from backend.extract_listing import extract_listing, insert_listing_to_weaviate
from backend.advisor import advise

# --- env checks ---
REQUIRED_ENV = ["LLM_API_KEY", "WCS_URL", "WCS_API_KEY"]
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    raise SystemExit(f"Missing env vars: {', '.join(missing)}")

app = FastAPI(title="AutoAdvisor API", version="0.1.0")

# Allow Bubble to call this API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Bubble domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    ad_text: str = Field(..., description="Raw listing text pasted by user")
    source_url: Optional[str] = Field(None, description="Optional source URL")

class AnalyzeResponse(BaseModel):
    listing: Dict[str, Any]
    advisor: Dict[str, Any]
    weaviate_id: Optional[str] = None

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest = Body(...)):
    try:
        # 1) Extract
        listing = extract_listing(req.ad_text)
        if req.source_url:
            listing["source_url"] = req.source_url

        # 2) Store in Weaviate
        inserted = insert_listing_to_weaviate(listing)
        weaviate_id = inserted.get("id")

        # 3) Advise (RAG over local KB CSV)
        advisor = advise(listing)

        return AnalyzeResponse(listing=listing, advisor=advisor, weaviate_id=weaviate_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
