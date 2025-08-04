#!/usr/bin/env python3
import uvicorn

if __name__ == "__main__":
    print("Starting GynAI FastAPI Server...")
    # Use import string format for reload to work properly
    uvicorn.run(
        "app.main:app",  # Import string instead of app object
        host="0.0.0.0",  # Allow network access
        port=8000,
        reload=True
    )
