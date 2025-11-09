"""FastAPI application for Grammar Scoring Engine."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline.grammar_scorer import GrammarScoringPipeline
from src.api.models import GrammarScoreResponse, TextScoreRequest, HealthResponse
from src.utils.config import API_HOST, API_PORT, MAX_FILE_SIZE

# Initialize FastAPI app
app = FastAPI(
    title="Grammar Scoring Engine API",
    description="Score grammar quality from voice samples or text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialized on startup)
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    print("\n🚀 Starting Grammar Scoring Engine API...")
    try:
        pipeline = GrammarScoringPipeline()
        print("✅ API ready to accept requests!\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return {
        "status": "running",
        "version": "1.0.0",
        "models_loaded": pipeline is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "version": "1.0.0",
        "models_loaded": pipeline is not None
    }


@app.post("/score/audio", response_model=GrammarScoreResponse)
async def score_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, etc.)"),
    preprocess: bool = Form(default=True, description="Apply audio preprocessing"),
    return_detailed: bool = Form(default=True, description="Return detailed analysis")
):
    """
    Score grammar from audio file.
    
    Args:
        file: Audio file to process
        preprocess: Whether to apply audio preprocessing
        return_detailed: Whether to return detailed analysis
        
    Returns:
        Grammar score and analysis
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to start
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process audio
        result = pipeline.score_audio(
            tmp_path,
            preprocess=preprocess,
            return_detailed=return_detailed
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/score/text", response_model=GrammarScoreResponse)
async def score_text(request: TextScoreRequest):
    """
    Score grammar from text.
    
    Args:
        request: Text scoring request
        
    Returns:
        Grammar score and analysis
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = pipeline.score_text(
            request.text,
            return_detailed=request.return_detailed
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints."""
    return {
        "api_name": "Grammar Scoring Engine",
        "version": "1.0.0",
        "endpoints": {
            "/": "API root and health info",
            "/health": "Health check",
            "/score/audio": "Score grammar from audio file (POST)",
            "/score/text": "Score grammar from text (POST)",
            "/api/info": "This endpoint",
            "/docs": "Interactive API documentation"
        },
        "supported_audio_formats": ["wav", "mp3", "m4a", "flac", "ogg"],
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║       Grammar Scoring Engine API Server                 ║
║                                                          ║
║  Starting server at http://{API_HOST}:{API_PORT}              ║
║  Documentation: http://{API_HOST}:{API_PORT}/docs            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
