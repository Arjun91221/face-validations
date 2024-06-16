import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from io import BytesIO
from keras.models import load_model
import h5py
from typing import List

app = FastAPI()

class ImageBase64(BaseModel):
    encoded_image: str

class ImageValidationResponse(BaseModel):
    message: str
    class_name: str
    confidence_score: float
    is_valid: bool

def load_opencv_model():
    model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

def process_image(net, image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_locations = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_locations.append((startY, endX, endY, startX))

    return face_locations

def classify_face(model, class_names, face_image):
    face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
    face_image = np.asarray(face_image, dtype=np.float32).reshape(1, 224, 224, 3)
    face_image = (face_image / 127.5) - 1

    prediction = model.predict(face_image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def remove_groups_from_config(filepath):
    with h5py.File(filepath, 'r+') as f:
        model_config = f.attrs.get('model_config')
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        model_config = model_config.replace('"groups": 1,', '')
        f.attrs['model_config'] = model_config.encode('utf-8') if isinstance(model_config, str) else model_config

@app.post("/validate_image", response_model=ImageValidationResponse)
async def validate_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        net = load_opencv_model()
        remove_groups_from_config("models/keras_model.h5")
        keras_model = load_model("models/keras_model.h5", compile=False)
        class_names = open("models/labels.txt", "r").readlines()

        frame, face_locations = image, process_image(net, image)
        class_name, confidence_score = classify_face(keras_model, class_names, frame)
        
        is_valid, message = False, "Human Face not Found"
        if "human" in class_name.lower():
            if face_locations:
                valid_faces = []
                for (top, right, bottom, left) in face_locations:
                    face_image = frame[top:bottom, left:right]
                    valid_faces.append((top, right, bottom, left))
                if len(valid_faces) == 0:
                    is_valid, message = False, "Human Face not Found"
                elif len(valid_faces) > 1:
                    is_valid, message = False, "Multiple Faces Detected"
                else:
                    is_valid, message = True, "Single Human Face Detected"
            else:
                is_valid, message = False, "Human Face not Found"

        return ImageValidationResponse(
            message=message,
            class_name=class_name,
            confidence_score=confidence_score,
            is_valid=is_valid
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_image_base64", response_model=ImageValidationResponse)
async def validate_image_base64(encoded_image: ImageBase64):
    try:
        image_data = base64.b64decode(encoded_image.encoded_image)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        net = load_opencv_model()
        remove_groups_from_config("models/keras_model.h5")
        keras_model = load_model("models/keras_model.h5", compile=False)
        class_names = open("models/labels.txt", "r").readlines()

        frame, face_locations = image, process_image(net, image)
        class_name, confidence_score = classify_face(keras_model, class_names, frame)
        
        is_valid, message = False, "Human Face not Found"
        if "human" in class_name.lower():
            if face_locations:
                valid_faces = []
                for (top, right, bottom, left) in face_locations:
                    face_image = frame[top:bottom, left:right]
                    valid_faces.append((top, right, bottom, left))
                if len(valid_faces) == 0:
                    is_valid, message = False, "Human Face not Found"
                elif len(valid_faces) > 1:
                    is_valid, message = False, "Multiple Faces Detected"
                else:
                    is_valid, message = True, "Single Human Face Detected"
            else:
                is_valid, message = False, "Human Face not Found"

        return ImageValidationResponse(
            message=message,
            class_name=class_name,
            confidence_score=confidence_score,
            is_valid=is_valid
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    