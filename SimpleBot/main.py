from ultralytics import YOLO
import cv2
import numpy as np
#from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, Form, Request  #, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse  #, JSONResponse
#from fastapi.staticfiles import StaticFiles
#from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
#import json
# from typing import Optional
import base64
from pydantic import BaseModel
import glob
import os
import time
#import re
import json
#import torch
#torch.cuda.is_available = lambda : False  # принудительно использовать только CPU для PyTorch
from bboxes import draw_boxes

# Путь к папке для сохранения
save_path = './images'
default_mdl_path = './models/'  # Путь по умолчанию, где хранятся модели

app = FastAPI(title="Counters System API for DataSet", version="0.1.2", debug=True)  # Инициализация FastAPI

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResponseModel(BaseModel):
    image: str
    results: dict

def get_latest_model(path):  # Функция для выбора самой последней модели
    list_of_files = glob.glob(f'{path}*.pt')
    if not list_of_files:
        return None  # Ни одного файла модели не найдено
    latest_model = max(list_of_files, key=os.path.getctime)  # Выбираем самый свежий файл
    print(f'♻️  Latest Model: {latest_model}')
    return latest_model


# Добавление статических файлов
#app.mount('/static', StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")

def generate_file_name(model_name, file_ext, images_dir='./images'):
    # Получение текущего времени
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # Выделение первых двух цифр из названия модели
    #model_digits = re.findall(r'\d+', model_name)[:2]
    #model_prefix = ''.join(model_digits)[:2]  # Берем только первые две цифры

    model_prefix = model_name[:2]

    # Определение порядкового номера файла
    existing_files = glob.glob(os.path.join(images_dir, '*.', file_ext))  # предполагаем, что изображения в формате .jpg
    next_file_number = len(existing_files) + 1

    # Генерация имени файла
    file_name = f"{current_time}_{str(next_file_number).zfill(5)}_M{model_prefix}.{file_ext}"

    return file_name


# Get client IP address 
@app.get("/")
async def read_root(request: Request):
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_host = forwarded_for.split(",")[0]
    else:
        client_host = request.client.host
    return {"📡 Client IP: ": client_host}


@app.get('/info')
def read_root():
    return {'Project 2023': '📟 Counters FastAPI Server - TgeBot\n[г. Москва, 2023 г.]'}


@app.get('/models')
def list_models():
    models_dir = "models"
    models = []
    sorted_files = sorted(os.listdir(models_dir))

    for filename in sorted_files:
        if filename.endswith(".h5") or filename.endswith(".pt"):  # Расширения файлов моделей
            models.append(filename)
    #return JSONResponse(content={"Models": models})
    return {"Models": models}


@app.post('/predict')
# async def predict(file: UploadFile = File(...), mdl_name: Optional[str] = Form(None)):
async def predict(file: UploadFile = File(...), mdl_name: str = Form('./models/09_cds2_s-seg_1280_100e.pt')):
    if not file:
        return {"‼️ error": "🚷 No file uploaded"}
    
    # Получение расширения файла
    file_ext = file.filename.split('.')[-1].lower()

    # Поддерживаемые форматы
    supported_formats = ["jpg", "jpeg", "png", "bmp", "tiff"]
    if file_ext not in supported_formats:
        return {"error": f"Unsupported file format: {file_ext}"}
    
    
    # print("... Received model name:", mdl_name) # для отладки !!!
    image_stream = io.BytesIO(await file.read())
    image_stream.seek(0)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
    image_stream.close()

    if image is None:
        return {"error": "Invalid image file"}

    # Если имя модели предоставлено, создаем полный путь к модели
    if mdl_name:
        selected_model = os.path.join(default_mdl_path, mdl_name)
        # Проверяем, существует ли файл модели
        if not os.path.exists(selected_model):
            selected_model = get_latest_model(default_mdl_path)
        print(f'⚖️  Selected Model: {selected_model}')
    else:
        selected_model = get_latest_model(default_mdl_path)

    if selected_model is None:
        return {"error": "No model files found"}

    # Загружаем модель в память
    model = YOLO(selected_model) # if selected_model else None
    # results = model.predict(source=image, imgsz=640, conf=0.25, name="00", save_txt=True, save_conf=True)
    results = model.predict(source=image, imgsz=640, conf=0.25)

    # Save the image +++++++++++++++++++++++++++++++
    model_name = os.path.basename(selected_model)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # file_name = generate_file_name(model_name, file_ext)
    file_name = generate_file_name(model_name, "jpg")
    file_path = f"{save_path}/{file_name}"
    cv2.imwrite(file_path, image)
    # ===============================================

    image, counter, number, speed, wh_check = draw_boxes(image, results)

    # Save the image to BytesIO object and send it as a response with Content-Disposition header
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return {"error": "Failed to save the image"}

    image_data = {
        "model_name": model_name,
        "object_count": number,
        "inference": speed,
        "current_time": current_time,
        "counter": counter,
        "wh_check": wh_check,
        "file_name": file_name
    }
    # print(f'\n{image_data}\n')

    # Замена расширения файла на .json
    json_file_name = file_name.rsplit('.', 1)[0] + '.json'
    # Сохранение данных в JSON файл
    with open(f"{save_path}/{json_file_name}", 'w') as json_file:
        json.dump(image_data, json_file, indent=4)


    # ++++++++++++++++++++ Image Data ++++++++++++++++++++
    # image_data = {"model_name": selected_model, "mode": mkey}
    img_str = base64.b64encode(buffer).decode()  # Encode the image as base64 string 
    response_data = {
        "image": img_str,
        "results": image_data  # Extract additional data from the model
    }
    # return JSONResponse(content=json.dumps(response_data), media_type="application/json")
    response = ResponseModel(image=img_str, results=image_data)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
