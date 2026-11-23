from fastapi import FastAPI, File, UploadFile
from app.validator import validate_image_upload
from app.predictor import run_inference

app = FastAPI(
    title="Production-Ready Image Inference Service",
    description="Modular REST API wrapping image preprocessing, strict edge-case validation, and ML inference.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "image-inference-api"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    validated_image = await validate_image_upload(file)
    result = run_inference(validated_image)
    return {
        "filename": file.filename,
        "prediction": result,
    }
