import os
import logging
import datetime
import uuid  # Imported to generate valid UUIDs
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fer_model import predict_emotion
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ---------------------------------------------------------
# 1. SETUP SUPABASE CONNECTION
# ---------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logging.warning("Supabase keys not found! Database save will fail.")
    supabase = None

# ---------------------------------------------------------
# 2. CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/emotion")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")
        contents = await file.read()

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(content={"error": "Could not decode image"}, status_code=400)

        # ---------------------------------------------------------
        # 3. RUN FER MODEL
        # ---------------------------------------------------------
        try:
            result = predict_emotion(image)
        except Exception as pe:
            logging.exception("Error inside predict_emotion()")
            return JSONResponse(content={"error": f"Prediction failed: {str(pe)}"}, status_code=500)

        logging.info(f"Prediction result: {result}")

        # ---------------------------------------------------------
        # 4. SAVE TO SUPABASE (Updated for your Schema)
        # ---------------------------------------------------------
        if result["emotion"].lower() != "none" and supabase:
            
            # Get current time
            now = datetime.datetime.utcnow()
            
            # Map data to your exact table columns
            db_record = {
                # Generates a random UUID. Replace this if you have a real user ID.
                "user_id": os.environ.get("DEV_USER_ID"),
                
                # 'timestamp without time zone' expects ISO format
                "timestamp": now.isoformat(),
                
                # 'character varying'
                "predicted_emotion": result["emotion"],
                
                # 'double precision'
                "emotion_confidence": float(result["confidence"]),
                
                # 'date' column expects YYYY-MM-DD
                "date": now.strftime("%Y-%m-%d")
            }

            try:
                # Ensure the table name is correct. Based on previous context, 
                # you called it 'emotion_logs' or 'face_emotion'. Update below if needed.
                response = supabase.table("face_emotion").insert(db_record).execute()
                logging.info("Logged prediction to Supabase.")
            except Exception as db_err:
                logging.error(f"Failed to save to Supabase: {db_err}")
                # We log the error but return the prediction result so the Pi doesn't crash
        
        return result

    except Exception as e:
        logging.exception("Exception in /emotion")
        return JSONResponse(content={"error": str(e)}, status_code=500)