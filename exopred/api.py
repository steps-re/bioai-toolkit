"""ExoPred API server.

Run: uvicorn exopred.api:app --host 0.0.0.0 --port 8000
Or:  python3 -m exopred.api
"""

import os
import time
from collections import defaultdict
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise SystemExit(
        "FastAPI not installed. Install: pip install fastapi uvicorn pydantic"
    )

from exopred.predict import ExoPredPredictor, ENZYME_REGISTRY, VALID_AA

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ExoPred API",
    description=(
        "Exopeptidase degradation prediction for therapeutic peptides. "
        "Predicts susceptibility to APN, LAP, DPP-IV, CPA, CPB, and NEP."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
_predictor = ExoPredPredictor()

# ---------------------------------------------------------------------------
# Auth + rate limiting
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("EXOPRED_API_KEY")

# In-memory rate limiter: {api_key_or_ip: [(timestamp, ...)]
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 100  # requests per minute
_RATE_WINDOW = 60.0  # seconds


def _get_client_id(request: Request) -> str:
    """Get client identifier from API key header or IP."""
    key = request.headers.get("X-API-Key", "")
    if key:
        return f"key:{key}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def _check_rate_limit(client_id: str) -> None:
    """Raise 429 if client exceeds rate limit."""
    now = time.time()
    cutoff = now - _RATE_WINDOW
    # Prune old entries
    _rate_store[client_id] = [t for t in _rate_store[client_id] if t > cutoff]
    if len(_rate_store[client_id]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({_RATE_LIMIT} requests/minute). Try again shortly.",
        )
    _rate_store[client_id].append(now)


async def auth_and_rate_limit(request: Request) -> str:
    """Dependency: authenticate and rate-limit the request."""
    api_key = request.headers.get("X-API-Key", "")

    # If EXOPRED_API_KEY is set, require a matching key
    if _API_KEY is not None:
        if not api_key or api_key != _API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Set X-API-Key header.",
            )

    client_id = _get_client_id(request)
    _check_rate_limit(client_id)
    return client_id


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=2, max_length=100, description="Peptide sequence (1-letter AA codes)")
    n_terminal: str = Field("none", description="N-terminal modification: none, acetyl, fmoc, peg, daa")
    c_terminal: str = Field("none", description="C-terminal modification: none, amide, peg, daa")
    enzymes: list[str] = Field(default=["all"], description="Enzyme names or ['all']")
    output: str = Field("full", description="Output detail: 'full' or 'summary'")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        invalid = set(v) - VALID_AA
        if invalid:
            raise ValueError(f"Invalid amino acids: {', '.join(sorted(invalid))}")
        return v

    @field_validator("n_terminal")
    @classmethod
    def validate_n_terminal(cls, v: str) -> str:
        allowed = {"none", "acetyl", "ac", "fmoc", "peg", "daa", "nh2"}
        if v.lower().strip() not in allowed:
            raise ValueError(f"Unknown N-terminal mod: {v}. Allowed: {sorted(allowed)}")
        return v.lower().strip()

    @field_validator("c_terminal")
    @classmethod
    def validate_c_terminal(cls, v: str) -> str:
        allowed = {"none", "amide", "nh2", "peg", "daa"}
        if v.lower().strip() not in allowed:
            raise ValueError(f"Unknown C-terminal mod: {v}. Allowed: {sorted(allowed)}")
        return v.lower().strip()


class BatchSequenceItem(BaseModel):
    sequence: str = Field(..., min_length=2, max_length=100)
    n_terminal: str = Field("none")
    c_terminal: str = Field("none")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        invalid = set(v) - VALID_AA
        if invalid:
            raise ValueError(f"Invalid amino acids: {', '.join(sorted(invalid))}")
        return v


class BatchPredictRequest(BaseModel):
    sequences: list[BatchSequenceItem] = Field(..., max_length=100, description="Up to 100 sequences")
    enzymes: list[str] = Field(default=["all"], description="Enzyme names or ['all']")
    output: str = Field("full", description="Output detail: 'full' or 'summary'")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    """Health check -- returns model version and status."""
    info = _predictor.model_info
    return {
        "status": "ok",
        "service": "ExoPred API",
        "version": info["version"],
        "mode": info["mode"],
        "supported_enzymes": info["supported_enzymes"],
    }


@app.post("/api/v1/predict")
async def predict_single(
    body: PredictRequest,
    _client: str = Depends(auth_and_rate_limit),
):
    """Predict exopeptidase degradation for a single peptide."""
    enzyme_arg = "all"
    if body.enzymes and body.enzymes != ["all"]:
        # If multiple specific enzymes requested, run each
        results = {}
        for enz in body.enzymes:
            pred = _predictor.predict(
                body.sequence,
                enzyme=enz,
                n_mod=body.n_terminal,
                c_mod=body.c_terminal,
            )
            results.update(pred["predictions"])
        # Re-run full for overall score and recommendation
        full = _predictor.predict(
            body.sequence,
            enzyme="all",
            n_mod=body.n_terminal,
            c_mod=body.c_terminal,
        )
        # Filter predictions to requested enzymes only
        filtered_preds = {k: v for k, v in full["predictions"].items() if k in body.enzymes}
        full["predictions"] = filtered_preds
        return full
    else:
        return _predictor.predict(
            body.sequence,
            enzyme=enzyme_arg,
            n_mod=body.n_terminal,
            c_mod=body.c_terminal,
        )


@app.post("/api/v1/predict/batch")
async def predict_batch(
    body: BatchPredictRequest,
    _client: str = Depends(auth_and_rate_limit),
):
    """Batch prediction for up to 100 peptide sequences."""
    results = []
    enzyme_str = "all" if "all" in body.enzymes else None

    for item in body.sequences:
        if enzyme_str:
            pred = _predictor.predict(
                item.sequence,
                enzyme="all",
                n_mod=item.n_terminal,
                c_mod=item.c_terminal,
            )
        else:
            pred = _predictor.predict(
                item.sequence,
                enzyme="all",
                n_mod=item.n_terminal,
                c_mod=item.c_terminal,
            )
            # Filter to requested enzymes
            pred["predictions"] = {
                k: v for k, v in pred["predictions"].items()
                if k in body.enzymes
            }
        results.append(pred)

    return {"count": len(results), "results": results}


@app.get("/api/v1/enzymes")
async def list_enzymes(
    _client: str = Depends(auth_and_rate_limit),
):
    """List all supported enzymes with metadata."""
    return {
        "count": len(ENZYME_REGISTRY),
        "enzymes": ENZYME_REGISTRY,
    }


@app.get("/api/v1/model/info")
async def model_info(
    _client: str = Depends(auth_and_rate_limit),
):
    """Model version, training data stats, and benchmark scores."""
    return _predictor.model_info


# ---------------------------------------------------------------------------
# Run with python3 -m exopred.api
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("EXOPRED_PORT", "8000"))
    print(f"Starting ExoPred API on port {port}...")
    uvicorn.run(
        "exopred.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
