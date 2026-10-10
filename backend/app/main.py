from fastapi import FastAPI

app = FastAPI(
    title="Dyslexia Reading Assistance System",
    description="Backend API for the AI-Based Adaptive Reading Assistance System for Children with Dyslexia",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Dyslexia Reading Assistance API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }