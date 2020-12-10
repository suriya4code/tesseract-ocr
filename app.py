import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import pytesseract
import re
from pydantic import BaseModel

print("ocr start")


def read_image(img):
    pytesseract.pytesseract.tesseract_cmd =     '/app/ .apt/usr/bin/tesseract'
    content = pytesseract.image_to_string(img)
    return content

app = FastAPI()

class ImageType (BaseModel):
    url: str

    app.post("/predict")
    def prediction(req : Request, file : bytes = File(...)):
        if req.method == "POST":
            image_stream = io.BytesIO(file)
            image_stream.seek(0)
            filebytes = np.asarray(bytearray(image_stream.read()),dtype=np.uint8)
            frame = cv2.imdecode(filebytes, cv2.IMREAD_ANYCOLOR)
            label = read_image(frame)
            return label
        return "No POST req found"



