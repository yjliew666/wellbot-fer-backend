import os
import logging
import datetime
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fer_model import predict_emotion
from supabase import create_client, Client
from dotenv import load_dotenv
from fer_status_tracker import log_request, log_result, read_recent_requests, read_recent_results

load_dotenv()
logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="Well-Bot Facial Emotion Recognition API",
    description="Facial emotion recognition service",
    version="1.0.0"
)

# Create router for FER endpoints
fer_router = APIRouter(prefix="/fer", tags=["FER"])

# --- Connection Setup ---
#testing
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_validated_uuid(request_id: str) -> str:
    """Helper to handle the hierarchy: Request Form -> Env Var -> Error."""
    # 1. Check Request ID
    if request_id:
        try:
            return str(uuid.UUID(request_id))
        except ValueError:
            logging.warning(f"Invalid UUID in request: {request_id}")

    # 2. Check Environment Fallback
    env_id = os.environ.get("DEV_USER_ID")
    if env_id:
        try:
            return str(uuid.UUID(env_id))
        except ValueError:
            logging.error(f"Invalid UUID in DEV_USER_ID env: {env_id}")

    return None

@fer_router.post("/emotion")
async def detect_emotion(
    file: UploadFile = File(...), 
    user_id: str = Form(None) 
):
    # 1. Identity Validation
    validated_id = get_validated_uuid(user_id)
    if not validated_id:
        raise HTTPException(status_code=400, detail="Valid UUID required (from form or env)")

    # Get current timestamp for tracking
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now.isoformat()
    
    # Log request
    log_request(
        user_id=validated_id,
        timestamp=timestamp,
        filename=file.filename
    )

    try:
        # 2. Image Processing
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # 3. Model Inference
        result = predict_emotion(image)
        logging.info(f"User {validated_id} result: {result['emotion']}")

        # 4. Database Persistence
        db_write_success = False
        if result["emotion"].lower() != "none" and supabase:
            db_record = {
                "user_id": validated_id,
                "timestamp": timestamp,
                "predicted_emotion": result["emotion"],
                "emotion_confidence": float(result["confidence"]),
                "date": now.strftime("%Y-%m-%d")
            }

            try:
                supabase.table("face_emotion").insert(db_record).execute()
                db_write_success = True
            except Exception as db_err:
                logging.error(f"Supabase error: {db_err}")
        
        # Log result
        log_result(
            user_id=validated_id,
            timestamp=timestamp,
            emotion=result["emotion"],
            confidence=float(result["confidence"]),
            db_write_success=db_write_success,
            filename=file.filename
        )
        
        return result

    except HTTPException:
        raise # Re-raise FastAPI-specific errors
    except Exception as e:
        logging.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail=str(e))


@fer_router.get("/status")
async def get_fer_service_status():
    """
    Get detailed FER service status for cloud dashboard monitoring.

    Returns real-time information about:
    - Recent requests received
    - Processing results and database write status
    """
    try:
        now = datetime.datetime.now(datetime.timezone.utc)

        # Get recent requests (last 10 minutes)
        recent_requests = read_recent_requests(limit=20)
        ten_minutes_ago = now - datetime.timedelta(minutes=10)
        
        filtered_requests = []
        for req in recent_requests:
            try:
                req_time = datetime.datetime.fromisoformat(req["timestamp"].replace('Z', '+00:00'))
                if req_time >= ten_minutes_ago:
                    filtered_requests.append({
                        "user_id": req.get("user_id"),
                        "timestamp": req.get("timestamp"),
                        "filename": req.get("filename"),
                        "status": "received"
                    })
            except Exception:
                continue

        # Sort by timestamp (newest first)
        filtered_requests.sort(key=lambda x: x["timestamp"], reverse=True)
        filtered_requests = filtered_requests[:10]  # Keep only 10 most recent

        # Get recent results
        recent_results = read_recent_results(limit=20)
        
        # Format results for response
        formatted_results = []
        for result in recent_results:
            formatted_results.append({
                "user_id": result.get("user_id"),
                "timestamp": result.get("timestamp"),
                "filename": result.get("filename", "image.jpg"),
                "emotion": result.get("emotion"),
                "emotion_confidence": result.get("emotion_confidence"),
                "db_write_success": result.get("db_write_success", False),
                "status": "completed"
            })

        # Get the most recent result for "last successful result"
        last_successful_result = None
        if formatted_results:
            last_successful_result = formatted_results[0]

        return {
            "service": "fer",
            "timestamp": now.isoformat(),
            "status": "healthy",
            "recent_requests": filtered_requests,
            "recent_results": formatted_results[:10],  # Last 10 results (most recent)
            "last_successful_result": last_successful_result,
            "uptime": "unknown"  # Could be enhanced with actual uptime tracking
        }

    except Exception as e:
        logging.error(f"Error getting FER service status: {e}", exc_info=True)
        return {
            "service": "fer",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "error",
            "error": str(e),
            "recent_requests": [],
            "recent_results": [],
            "last_successful_result": None
        }


@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run."""
    try:
        return {
            "status": "healthy",
            "service": "fer"
        }
    except Exception as e:
        logging.error(f"Health check error: {e}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Well-Bot FER API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


# Include FER router
app.include_router(fer_router)