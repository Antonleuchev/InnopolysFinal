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

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO ('yolov8n.pt')
model = torch.hub.load('best.pt', pretrained=True)
model.eval()
app.run(host="0.0.0.0", port=args.port)


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
    print(input_np)
    input_np_norm = input_np / 255
    
    numpy_array_reshaped = input_np_norm.reshape(1, 1, input_np_norm.shape[0], input_np_norm.shape[1])
    print('numpy_array_reshaped.shape', numpy_array_reshaped.shape)
    
    tensor_val = torch.tensor(numpy_array_reshaped, dtype=torch.float32)
    print('tensor_val.size()', tensor_val.size())
    
    with torch.no_grad():
        output = ac_loaded(tensor_val)
        print('output.shape', output.shape)
        
    file_name = str(uuid.uuid4())    
    
    output_np = output.numpy()
    output_np = output_np.reshape(output_np.shape[-1], output_np.shape[-2])
    output_np = output_np * 255
    output_np = np.array(output_np, dtype=np.uint8)
    print('output_np.shape', output_np.shape)
    print('output_np', output_np)
    
    original_file_path = result_dir+file_name+original_suf+jpg_ext
    cv2.imwrite(original_file_path, input_np)
    processed_file_path = result_dir+file_name+processed_suf+jpg_ext
    cv2.imwrite(result_dir+file_name+processed_suf+jpg_ext, output_np)
    
    test = Image.fromarray(output_np, mode="L")
    print('test.size', test.size)
    imgProcessed = img2byte(test)
    
    db_manager = DbManager('db\\db.db')
    x = datetime.datetime.now()
    date, time = str(x).split(' ')
    db_manager.insert_history(date, time)
    usage_id = db_manager.get_last_inserted_row_id()
    
    db_manager.insert_result(original_file_path, processed_file_path, usage_id)
    
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
