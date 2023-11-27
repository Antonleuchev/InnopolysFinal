from flask import Flask, request, redirect, render_template
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from ultralytics import YOLO
from utils import *
import cv2
import io
import base64
import uuid
from PIL import Image
from matplotlib import cm
from db.db import *
import datetime
import traceback
from models import *

import argparse
from PIL import Image
import datetime


app = Flask(__name__)

result_dir = 'result//'
original_suf = '_original'
processed_suf = '_processed'
jpg_ext = '.jpg'
image_for_process_name = 'to_process.jpg'

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('best.pt')


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def process():
    stringParameter = request.form['stringParameter']
    intParameter = request.form['intParameter']
    
    print('request.files', request.files)
    imgParameter = request.files['imgParameter']
    print('request.files[imgParameter]', imgParameter)
    
    try:
        pil_image = Image.open(imgParameter)
        print(pil_image.size)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        errorName = e
        stackTrace = traceback.format_exc()
        return render_template('error.html', errorName = errorName, stackTrace = stackTrace)
    
    imgOriginal = img2byte(pil_image)
    #imgOriginal.save("imgOriginal.jpg")
    
    input_np = np.array(pil_image, dtype=np.float32)    
    input_np_cvt_color = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
    #print(input_np_cvt_color)
    cv2.imwrite(image_for_process_name, input_np_cvt_color)
    
    results = model.predict(image_for_process_name, save=True, imgsz=320, conf=0.5)
    
    #print(results)
    #print('-------------')
    data = []
    for r in results:
        #print(r.boxes)
        #print('-------------')
        for i in range(r.boxes.cls.shape[0]):
            #print(r.boxes.xywh[i])
            x1 = int(r.boxes.xywh[i][0] - (r.boxes.xywh[i][2] / 2))
            y1 = int(r.boxes.xywh[i][1] - (r.boxes.xywh[i][3] / 2))
            x2 = int(r.boxes.xywh[i][0] + (r.boxes.xywh[i][2] / 2))
            y2 = int(r.boxes.xywh[i][1] + (r.boxes.xywh[i][3] / 2))
            cv2.rectangle(input_np, (x1,y1), (x2,y2), (0,255,0), 3)
            #print(r.boxes.cls[i])
            #print(r.boxes.conf[i])
            #print(results.names[r.boxes.cls[i].item()])
            #break
        #break
    

    print(input_np.shape)
    #input_np = input_np.reshape(input_np.shape[-1], input_np.shape[0], input_np.shape[1])
    #print(input_np.shape) 
    input_np_pil = Image.fromarray(input_np.astype('uint8'), 'RGB')
    #print('test.size', input_np_pil.size)
    imgProcessed = img2byte(input_np_pil)
    
    db_manager = DbManager('db\\db.db')
    x = datetime.datetime.now()
    date, time = str(x).split(' ')
    db_manager.insert_history(date, time)
    
    
    return render_template('process.html', stringParameter = stringParameter, intParameter = intParameter,
    imgOriginal = imgOriginal.decode('utf-8'), imgProcessed = imgProcessed.decode('utf-8'))

@app.route("/result")
def result():
    db_manager = DbManager('db\\db.db')
    all_result_with_usage = db_manager.get_all_result_with_usage()
    data = []
    for res in all_result_with_usage:
        original_img = cv2.imread(res.original_path, cv2.IMREAD_GRAYSCALE)
        original_img_reshaped = original_img.reshape(original_img.shape[0], original_img.shape[1])
        original_img_reshaped_pil = Image.fromarray(original_img_reshaped, mode="L")
        original_img_byte = img2byte(original_img_reshaped_pil).decode('utf-8')
        
        processed_img = cv2.imread(res.processed_path, cv2.IMREAD_GRAYSCALE)
        processed_img_reshaped = processed_img.reshape(processed_img.shape[0], processed_img.shape[1])
        processed_img_reshaped_pil = Image.fromarray(processed_img_reshaped, mode="L")
        processed_img_byte = img2byte(processed_img_reshaped_pil).decode('utf-8')
        
        data.append(ResultWeb(original_img_byte, processed_img_byte, res.date, res.time))
    
    return render_template('result.html', data = data)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
