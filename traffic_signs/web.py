# 首先导入需要的库。
from flask import Flask, render_template, request
import numpy as np
import cv2
# from tensorflow.keras import MobileNetV2, preprocess_input, decode_predictions
from skimage import exposure
from tensorflow.keras import models

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 对服务和模型进行初始化，并导入预训练模型。
app = Flask(__name__)
# model = MobileNetV2(weights='imagenet')
saved_model_path = "/Users/wangfeng/ML/ml_in_action/docker_prc/traffic_signs/save_model/1588597005"
model = models.load_model(saved_model_path)


# 实现图像识别的功能，当获取到的是 POST 传过来的数据，则开始识别，不然就返回 upload.html 界面。 在这里我们使用 render_template 模板，其功能是先引入 HTML 文件，然后根据后面传入的参数，对 HTML 进行修改渲染。
@app.route('/', methods=['POST', 'GET'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        # 将图片存在 static 文件夹中
        file.save('static/'+file.filename)
        # 读取图片
        image = cv2.imread('static/'+file.filename)
        # OpenCV 读取图片是 BGR 格式，需要转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 放缩图片到 224 * 224
        print(image.shape)
        image = cv2.resize(image, (32, 32))
        x = np.expand_dims(image, 0)
        x = process_image_data(x)
        print(x.shape)
        # 图片预处理
        # x = preprocess_input(x)
        # 进行预测
        output = model.predict(x[:,:,:,0:1])
        # 取 top1 的预测结果
        preds = np.argmax(output)
        # preds = decode_predictions(output, top=1)
        # predict = np.squeeze(preds)
        predict = preds
        # 返回数据
        return render_template('result.html', filename=file.filename, predict=predict)
    # GET 方法返回 upload.html
    return render_template('upload.html')

def process_image_data(X):
    #Y = 0.299 R + 0.587 G + 0.114 B
    X[:,:,:,0] = 0.299 * X[:,:,:,0] + 0.587*X[:,:,:,1] + 0.114*X[:,:,:,2]
    
    #0-1 scale
    X = X /255.0
    
    #直方图均衡化（图像增强）
    for i in range(X.shape[0]):
        X[i] = exposure.equalize_adapthist(X[i])

    return X

# 最后在地址 0.0.0.0:8080 启动服务。同样，必须指定 host= ，否则服务默认会在 127.0.0.1 启动，这样的话是无法访问 Web 服务的。
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)