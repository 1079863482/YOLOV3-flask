import torch
import torch.nn as nn
import numpy as np


class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )


    def forward(self, x):

        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)


class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        time_channel = out_channels * 2

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, time_channel, 3, 1, 1),

            ConvolutionalLayer(time_channel, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, time_channel, 3, 1, 1),

            ConvolutionalLayer(time_channel, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)



class Darknet(nn.Module):
    def __init__(self, cls=80):
        super(Darknet, self).__init__()

        output_channel = 3 * (5 + cls)

        self.trunk52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownsamplingLayer(32, 64),

            ResidualLayer(64),

            DownsamplingLayer(64, 128),

            ResidualLayer(128),
            ResidualLayer(128),

            DownsamplingLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )

        self.trunk26 = nn.Sequential(
            DownsamplingLayer(256, 512),

            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )

        self.trunk13 = nn.Sequential(
            DownsamplingLayer(512, 1024),

            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
        )

        self.con_set13 = nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.predict_one = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, output_channel, 1, 1, 0)
        )

        self.up_to_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpsampleLayer()
        )

        self.con_set26 = nn.Sequential(
            ConvolutionalSet(768, 256)
        )

        self.predict_two = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, output_channel, 1, 1, 0)
        )

        self.up_to_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpsampleLayer()
        )

        self.con_set52 = nn.Sequential(
            ConvolutionalSet(384, 128)
        )

        self.predict_three = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, output_channel, 1, 1, 0)
        )


    def forward(self, x):
        feature_52 = self.trunk52(x)
        feature_26 = self.trunk26(feature_52)
        feature_13 = self.trunk13(feature_26)

        con_set_13_out = self.con_set13(feature_13)
        detection_13_out = self.predict_one(con_set_13_out)

        up_26_out = self.up_to_26(con_set_13_out)
        route_26_out = torch.cat((up_26_out, feature_26), dim=1)
        con_set_26_out = self.con_set26(route_26_out)
        detection_26_out = self.predict_two(con_set_26_out)

        up_52_out = self.up_to_52(con_set_26_out)
        route_52_out = torch.cat((up_52_out, feature_52), dim=1)
        con_set_52_out = self.con_set52(route_52_out)
        detection_52_out = self.predict_three(con_set_52_out)

        return detection_13_out, detection_26_out, detection_52_out

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)

        weights = np.fromfile(fp, dtype=np.float32)  # 加载 np.ndarray 中的剩余权重，权重是以float32类型存储的
        weights = weights[5:]        # 前五个是头部信息

        model_list = []           
        for model in self.modules():          # 将权重加载到一个list中
            if isinstance(model, nn.BatchNorm2d):
                model_list.append(model)
            if isinstance(model, nn.Conv2d):
                model_list.append(model)

        ptr = 0
        is_continue = False
        for i in range(0, len(model_list)):
            if is_continue:
                is_continue = False
                continue

            conv = model_list[i]
            # print(i // 2, conv)

            if i < len(model_list) - 1 and isinstance(model_list[i + 1], nn.BatchNorm2d):
                is_continue = True

                bn = model_list[i + 1]
                # print(bn)
                num_bn_biases = bn.bias.numel()
                # print(num_bn_biases, weights[ptr:ptr + 4 * num_bn_biases])

                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)

                # print(bn.bias)
                # print(bn.weight)
                # print(bn.running_mean)
                # print(bn.running_var)
            else:
                is_continue = False

                num_biases = conv.bias.numel()
                # print(weights[ptr:ptr + num_biases])

                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                ptr = ptr + num_biases

                conv_biases = conv_biases.view_as(conv.bias.data)

                conv.bias.data.copy_(conv_biases)

            num_weights = conv.weight.numel()
            # print(num_weights, weights[ptr:ptr + num_weights])

            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)

        fp.close()


if __name__ == '__main__':

    net = Darknet(2)

    save_net = r"E:\YOLO\model\converted.weights"
    net.load_weights(save_net)

    # net.load_state_dict(torch.load("model/best-100.pt"))

    x = torch.Tensor(2, 3, 416, 416)

    y_13, y_26, y_52 = net(x)

    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
