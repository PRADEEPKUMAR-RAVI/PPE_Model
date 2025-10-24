from fastapi import APIRouter, File, UploadFile
from app.services.inference_service import InferenceService

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        inference_service = InferenceService()
        result = await inference_service.process_file(file)
        return {"message": "File processed successfully", "result": result}
    except Exception as e:
        return {"error": str(e)}