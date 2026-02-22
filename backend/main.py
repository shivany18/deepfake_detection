from fastapi import FastAPI, UploadFile, File
from detection.image_detect import detect_image
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="DeepFake Image Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"status": "Backend running successfully"}

@app.post("/detect/image")
async def image_detection(file: UploadFile = File(...)):
    result = detect_image(file)
    return result
