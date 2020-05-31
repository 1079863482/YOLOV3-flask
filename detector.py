import anchors_cfg
from darknet import Darknet
import torchvision
from utils import *
import cv2

class Detector(torch.nn.Module):

    def __init__(self,save_net):
        super(Detector, self).__init__()
        self.net = Darknet(80)

        # self.net.load_state_dict(torch.load("model/yolov3.pth"))
        self.net.load_weights(save_net)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.net.eval()          #开始测试

    def forward(self, input, thresh, anchors):        #将图片、置信度阈值、建议框输入
        input_ = input.to(self.device)

        output_13, output_26, output_52 = self.net(input_)        #将图片传入网络中得到三个特征图输出

        # output_13 = output_13.cpu()
        # output_26 = output_26.cpu()
        # output_52 = output_52.cpu()

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


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = torchvision.transforms.Compose([  # 归一化，Tensor处理
        torchvision.transforms.ToTensor()
    ])
    detector = Detector(r"E:\YOLO\model\yolov3.weights")
    for i in range(1,2):
        image = Image.open(r"E:\YOLO\test_image\timg1.jpg")
        image_c =image.copy()                             #将原图复制
        W, H, scale, img = narrow_image(image)            #传入缩放函数中，得到W,H,缩放比例，缩放后的416*416的图片
        img_data = transforms(img).unsqueeze(0)         
        box = detector(img_data, 0.25, anchors_cfg.ANCHORS_GROUP)          #将缩放后的416*416图像传入探索函数，得到目标框
        box = enlarge_box(W, H, scale, box)                   #将目标框按照缩放比例反算回原图
        print(box)
        image_out = draw(box,image_c)                         #将框在原图画出
        image_out.show()

