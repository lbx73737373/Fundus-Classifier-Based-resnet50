# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import io
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # 解决跨域问题

weights_path = './model3.pt'
result_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(result_json_path), "result json path does not exist..."

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device)
# create model
model = torchvision.models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 9)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)

target_layers = [model.layer4[-1]]
cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)

class_indict = {
    '0': '正常',
    '1': '非增殖型糖尿病性视网膜病',
    '2': '黄斑萎缩',
    '3': '老年性黄斑病变',
    '4': '高血压视网膜病变',
    '5': '近视',
    '6': '其他黄斑病变',
    '7': '视盘病变',
    '8': '其他眼部疾病',
    # '9': '贫血',
    # '10': '高血压',
    # '11': '2型糖尿病',
    # '12': '肾功能衰竭'
}

model.eval()


my_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.477603, 0.295945, 0.175173],
                                        [0.299370, 0.201916, 0.1374784])])


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    grayscale_cam = cam(input_tensor=tensor)
    grayscale_cam = grayscale_cam[0, :]
    outputs = torch.sigmoid(model.forward(tensor).squeeze())
    prediction = outputs.detach().cpu().numpy()
    # print(prediction)
    # new_possibility = 1/2 *np.random.random(4)
    # prediction = np.append(prediction, new_possibility)
    index_pre = []
    for i in range(len(prediction)):
        s = [class_indict[str(i)] + ' ' * (13 - len(class_indict[str(i)])), float('%0.3f' % prediction[i])]
        index_pre.append(s)

    # sort probability
    index_pre.sort(key=lambda x: x[1], reverse=True)
    return_info = index_pre
    return return_info, grayscale_cam


@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == 'POST':
        image = request.files["file"]
        image1_path = './static/image/'
        image_name = image.filename
        img_bytes = image.read()
        image.seek(0)
        image.save(image1_path + '4.jpeg')
        img = cv2.imread(image1_path + '4.jpeg')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_float_np = np.float32(np.array(img)) / 255

        # os.remove(image1_path + '4.jpeg')
        # os.remove(image1_path + '5.jpeg')
        # os.remove(image1_path + '6.jpeg')
        # cv2.imwrite(image1_path + '4.jpeg', img)
        info,  grayscale_cam = get_prediction(image_bytes=img_bytes)
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam),  cv2.COLORMAP_JET)
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
        cv2.imwrite(image1_path + '5.jpeg', cam_image)
        cv2.imwrite(image1_path + '6.jpeg', heatmap)
        name0 = info[0]
        name1 = info[1]
        name2 = info[2]
        name3 = info[3]
        name4 = info[4]
        name5 = info[5]
        name6 = info[6]
        name7 = info[7]
        name8 = info[8]
        # name9 = info[9]
        # name10 = info[10]
        # name11 = info[11]
        # name12 = info[12]
        return render_template("index1.html", name0=name0, name1=name1, name2=name2,
                               name3=name3, name4=name4, name5=name5, name6=name6,
                               name7=name7, name8=name8
                               )
    return render_template('index.html', name='null', hh='null', name2='null')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=50002)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
