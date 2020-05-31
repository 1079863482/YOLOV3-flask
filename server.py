import io
from PIL import Image
from flask import Flask, request, jsonify

import torch
import json
import time
from darknet import Darknet
from detect_yolo import *

# with open('label.json', 'rb') as f:
#     label_id_name_dict = json.load(f)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
input_size = 300

app = Flask(__name__)

transforms = torchvision.transforms.Compose([  # 归一化，Tensor处理
    torchvision.transforms.ToTensor()
])


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # print(data)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        # print("Hello")
        if request.files.get("image"):
            # print("world")
            now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            # read the image in PIL format

            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')
            image.save(now + '.jpg')
            # preprocess the image and prepare it for classification
            W, H, scale, img = narrow_image(image)  # 传入缩放函数中，得到W,H,缩放比例，缩放后的416*416的图片
            img_data = transforms(img).unsqueeze(0)

            # classify the input image and then initialize the list
            # of predictions to return to the client

            box = detector(img_data, 0.25, anchors_cfg.ANCHORS_GROUP)

            box = enlarge_box(W, H, scale, box)

            res = json_text(box,W,H)

            # indicate that the request was a success
            # data["success"] = True
            # print(data["success"])

    # return the data dictionary as a JSON response
            return res
    # return jsonify(data)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Pytorch model and Flask starting server..."
        "please wait until server has fully started"))

    path = r"E:\YOLO\model\yolov3.weights"
    model = Darknet(80)
    model.load_weights(path)
    detector = Detector(model)
    print('..... Finished loading model! ......')
    app.run(host='0.0.0.0', port =5000,debug=True )


