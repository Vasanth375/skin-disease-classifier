"""
FastAPI application for Skin Disease Classification
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import prediction
from app.core.config import settings

app = FastAPI(
    title="Skin Disease Classifier API",
    description="Real-time skin disease classification using CNN",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Skin Disease Classifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

