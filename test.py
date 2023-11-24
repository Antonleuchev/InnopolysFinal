'''

через обычную консоль
python -m pip install --upgrade pip 

через анаконду 
pip install --upgrade numpy
pip install --upgrade torch
pip install --upgrade torchaudio
pip install --upgrade torchvision

'''


from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('1700542173_402556126_669686641986512_3109925329954414562_n.jpg', save=True, imgsz=320, conf=0.5)

print(results)
print('-------------')
data = []
for r in results:
    print(r.boxes)
    print('-------------')
    for i in range(r.boxes.cls.shape[0]):
        print(r.boxes.xywh[i])
        print(r.boxes.cls[i])
        print(r.boxes.conf[i])
        print(results.names[r.boxes.cls[i].item()])
        break
    break