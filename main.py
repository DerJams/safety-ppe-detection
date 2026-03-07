from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# This points to your model in the new folder we created
model = YOLO("models/best.pt")

@app.get("/")
def home():
    return {"status": "PPE Detection API is Online"}

@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...)):
    # This reads the image you "send" to the API
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))

    # This runs your YOLO model on that image
    results = model.predict(img)
    
    # This cleans up the AI's data into a list for the user
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "item": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 2)
            })

    return {"detections": detections}