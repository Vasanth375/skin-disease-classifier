"""
Script to run the FastAPI application
"""
import uvicorn
import os
from app.main import app

# Vercel serverless handler
handler = app

if __name__ == "__main__":
    # Use Render's PORT if available, otherwise use settings
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")  # Default to 0.0.0.0 for production
    
    uvicorn.run(
        "run:handler",
        host=host,
        port=port,
        reload=False
    )

