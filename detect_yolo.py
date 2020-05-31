import anchors_cfg
from darknet import Darknet
import torchvision
from utils import *
import cv2
import json

class Detector(torch.nn.Module):

    def __init__(self,model):
        super(Detector, self).__init__()

        self.net = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.net.eval()  # 开始测试

    def forward(self, input, thresh, anchors):        #将图片、置信度阈值、建议框输入
        input_ = input.to(self.device)

        output_13, output_26, output_52 = self.net(input_)        #将图片传入网络中得到三个特征图输出


        idxs_13, vecs_13 = self._filter(output_13, thresh)         #得到置信度大于阈值的索引和输出
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        box = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)
        box = nms(box.cpu())

        return box

    def _filter(self, output, thresh):

        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        mask = torch.sigmoid(output[..., 4]) > thresh      #得到置信度大于阈值的掩码

        idxs = mask.nonzero()               #根据掩码得到索引
        vecs = output[mask]                 #置信度大于阈值的总输出

        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors).to(self.device)          #将建议框转化为张量

        n = idxs[:, 0]  # 所属的图片，批量传入时，这里不会用到
        a = idxs[:, 3]  # 建议框    [N,13,13,3,15]

        cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 1])) * t  # 索引+中心点输出乘以缩放比例得到原图的中心点y
        cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 0])) * t  #索引 + 中心点输出乘以缩放比例得到原图的中心点x
        w = anchors[a, 0] * torch.exp(vecs[:, 2])        #对应的实际框的w
        h = anchors[a, 1] * torch.exp(vecs[:, 3])        #对应的实际框的h

        cls = torch.sigmoid(vecs[:,4])


        if len(vecs[:,5:85]) > 0:
            _,pred = torch.max(vecs[:,5:85],dim=1)              #得到分类情况
            box = torch.stack([n.float(), cx, cy, w, h,pred.float(),cls], dim=1)
        else:
            box = torch.stack([n.float(), cx, cy, w, h, h,cls], dim=1)
        return box


def json_text(box,W,H):
    """
    画图函数：把框在原图上画出
    :param box: 实际框
    :param image: 原图
    :return: 画了框的图
    """

    fp = open(r'coco.names', "r")
    text = fp.read().split("\n")[:-1]
    results = []
    for i in range(len(box)):       #逐个取出
        cx = box[i][1]
        cy = box[i][2]
        w = box[i][3]
        h = box[i][4]

        num_class = int(box[i][5])       #分类标签
        class_name = text[num_class]

        x1 = int(cx - w / 2)     # 边界限定
        if x1 < 0:
            x1 = 0
        y1 = int(cy - h / 2)
        if y1 < 0:
            y1 = 0
        x2 = int(cx + w / 2)
        if x2 > W:
            x2 = W
        y2 = int(cy + h / 2)
        if y2 > H :
            y2 = H

        detect_result = {'class': class_name, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        results.append(detect_result)

        # print("results: ~~~~~~~~~~~~~~~~~~~~~",results)

    data_json = json.dumps(results, sort_keys=True, indent=4, separators=(',', ': '))

    return data_json


