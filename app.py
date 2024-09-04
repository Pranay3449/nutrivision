import io
import base64
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO  # Adjust this import according to your YOLO implementation
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = YOLO("best.pt")  # Load your model

def process_image(image):
    detections = model.predict(image)
    data = []  # List to store class name, confidence, and area data

    # Convert model plot result to image
    img_final = Image.fromarray(detections[0].plot()[:,:,::-1])
    img_bytes = io.BytesIO()
    img_final.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    processed_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    for r in detections:
        for c in r:
            label = c.names[c.boxes.cls.tolist().pop()]
            confidence = c.boxes.conf.tolist().pop()

            b_mask = np.zeros(image.shape[:2], np.uint8)

            # Create contour mask
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # Calculate area
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            class_name = label

            data.append([class_name, confidence, area])

    return data, processed_image

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLO object detection API. Use /detect/ endpoint to upload images for detection."}

@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv8
    data, processed_image = process_image(image)

    return JSONResponse(content={"detections": data, "image": processed_image})

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)