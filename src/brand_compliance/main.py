import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "brand_compliance.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
    )
