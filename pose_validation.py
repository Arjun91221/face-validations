import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
from typing import Union, List, Tuple
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
import io
from typing import List
import uuid
import aiohttp
from io import BytesIO
from pydantic import BaseModel
from io import BytesIO
from PIL import Image


app = FastAPI()

current_directory = os.path.dirname(__file__)

UPLOAD_DIR = f"{current_directory}/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Base64Image(BaseModel):
    base64_string: str
    
class PoseClassifier:
    def __init__(self, cnn_model_path: str, labels_path: str):
        """
        Initialize the PoseClassifier with necessary models and configurations.
        
        Args:
            cnn_model_path (str): Path to the CNN model file (.h5)
            labels_path (str): Path to the labels file
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Load CNN model and labels
        self.cnn_model = load_model(cnn_model_path, compile=False)
        with open(labels_path, "r") as f:
            self.class_names = f.readlines()
    
    def process_image_for_cnn(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image for input into the CNN model.
        
        Args:
            image (np.ndarray): OpenCV image (BGR format)
        
        Returns:
            np.ndarray: Processed image as a numpy array
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        image_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize the image
        normalized_image_array = (image_resized.astype(np.float32) / 127.5) - 1
        
        # Expand dimensions to match model input shape
        return np.expand_dims(normalized_image_array, axis=0)
    
    def classify_body_type(self, image: np.ndarray) -> str:
        """
        Classify whether the image contains a full body or half body.
        
        Args:
            image (np.ndarray): OpenCV image (BGR format)
        
        Returns:
            str: 'full_body', 'half_body', or 'no_person'
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return "no_person"
        
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        if nose.visibility > 0.5 and left_ankle.visibility > 0.3 and right_ankle.visibility > 0.3:
            return "full_body"
        else:
            return "half_body"
    
    def classify_pose(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify the pose (sitting or standing) using the CNN model.
        
        Args:
            image (np.ndarray): OpenCV image (BGR format)
        
        Returns:
            Tuple[str, float]: Class name and confidence score
        """
        data = self.process_image_for_cnn(image)
        prediction = self.cnn_model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index][2:].strip()
        confidence_score = prediction[0][index]
        return class_name, confidence_score
    
    def infer(self, image: Union[str, np.ndarray]) -> str:
        """
        Perform inference on a single image.
        
        Args:
            image (Union[str, np.ndarray]): Image file path or numpy array
        
        Returns:
            str: Classification result
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Invalid image format. Provide either a file path or a numpy array.")
        
        body_type = self.classify_body_type(image)
        pose, confidence = self.classify_pose(image)

        if body_type == "no_person":
            return False, "No person detected"
        elif body_type == "half_body" and pose == "Sitting":
            return False, "Invalid pose. Please, Upload a full standing body pose model"
        elif body_type == "half_body" and pose == "Standing":
            return False, "Half body image detected! Please choose a full body image"
        elif body_type == "full_body" and pose == "Sitting":
            return True, "Sitting pose"
        elif body_type == "full_body" and pose == "Standing":
            return True, "Standing pose"
    
    def infer_batch(self, images: Union[List[Union[str, np.ndarray]], str]) -> List[str]:
        """
        Perform inference on a batch of images or a directory of images.
        
        Args:
            images (Union[List[Union[str, np.ndarray]], str]): List of image file paths or numpy arrays, or a directory path
        
        Returns:
            List[str]: List of classification results
        """
        results = []
        
        if isinstance(images, str) and os.path.isdir(images):
            image_files = [os.path.join(images, f) for f in os.listdir(images) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                results.append(self.infer(image_file))
        elif isinstance(images, list):
            for image in images:
                results.append(self.infer(image))
        else:
            raise ValueError("Invalid input. Provide either a list of images or a directory path.")
        
        return results
    
def check_pose(image):
    classifier = PoseClassifier(f"{current_directory}/models/keras_model_v2.h5", "labels.txt")
    is_valid, message = classifier.infer(image)
    return is_valid, message


async def fetch_image_from_url(url: str) -> io.BytesIO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid URL")
            return BytesIO(await response.read())
        
def decode_base64_image(base64_string: str) -> BytesIO:
    image_data = base64.b64decode(base64_string)
    return BytesIO(image_data)
        
@app.post("/image-to-base64/")
async def image_to_base64(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    else:
        image_io = await fetch_image_from_url(url)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    os.remove(image_path)
    
    return encoded_string

@app.post("/base64-to-image/")
async def base64_to_image(base64_image: Base64Image):
    try:
        base64_string = base64_image.base64_string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        image.save(image_path)
        
        return StreamingResponse(open(image_path, "rb"), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")


@app.post("/detect-pose/")
async def pose_detection_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

        is_valid, message = check_pose(image_path)

        os.remove(image_path)

        return is_valid, message
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
