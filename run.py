"""
Script to run the FastAPI application
"""
import uvicorn
import os
from app.core.config import settings

if __name__ == "__main__":
    # Use Render's PORT if available, otherwise use settings
    port = int(os.getenv("PORT", settings.API_PORT))
    host = os.getenv("HOST", "0.0.0.0")  # Default to 0.0.0.0 for production
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=settings.API_RELOAD
    )

