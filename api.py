# api.py

import logging
from pathlib import Path
from typing import Literal, Union

import pydantic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from identifier.core import LanguageIdentifier
from identifier.advanced import HybridIdentifier

# App Setup 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Language Identifier API",
    description="An API to identify the language of a text using a simple n-gram model or an advanced hybrid model.",
    version="1.0.0"
)

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Model Loading (Singleton Pattern), we load the models once on startup. This is crucial for performance, as it avoids the expensive loading process on every single API request.
IdentifierModel = Union[LanguageIdentifier, HybridIdentifier, None]
simple_model: IdentifierModel = None
advanced_model: IdentifierModel = None

try:
    PROFILES_PATH = Path(__file__).resolve().parent / "profiles"
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Profiles directory not found at {PROFILES_PATH}")
    
    logging.info("Loading models...")
    simple_model = LanguageIdentifier(profile_dir=PROFILES_PATH)
    advanced_model = HybridIdentifier(profile_dir=PROFILES_PATH)
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    logging.critical(f"FATAL ERROR: Could not load language profiles. Please run 'build_profiles.py'. Details: {e}")


# API Data Models (using Pydantic) 
# These define the expected structure and types for API requests and responses.
# FastAPI uses them for validation and automatic documentation.

class IdentifyRequest(pydantic.BaseModel):
    text: str
    model: Literal['simple', 'advanced'] = 'advanced'
    alpha: float = 0.5

    @pydantic.validator('alpha')
    def alpha_must_be_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('alpha must be between 0.0 and 1.0')
        return v

class IdentifyResponse(pydantic.BaseModel):
    prediction: str | None = None
    distribution: list | None = None
    top_features: list | None = None
    error: str | None = None


# API Endpoints

@app.get("/", summary="Health Check")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "Language Identifier API is running."}


@app.post("/identify", response_model=IdentifyResponse, summary="Identify Language")
def identify_language(request: IdentifyRequest):
    """
    Identifies the language of the input text using the specified model.
    """
    model_instance = None
    
    if request.model == "advanced":
        model_instance = advanced_model
        # We need to re-initialize the model if a custom alpha is provided
        if request.alpha != 0.5 and advanced_model:
            model_instance = HybridIdentifier(profile_dir=PROFILES_PATH, alpha=request.alpha)
    else: # request.model == 'simple'
        model_instance = simple_model

    if not model_instance:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail=f"Model '{request.model}' is not available. It may have failed to load on startup."
        )

    return model_instance.identify(request.text)