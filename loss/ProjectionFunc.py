import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.p1 = nn.Linear(input_size,hidden_size)
        self.act = nn.ReLU()
        self.p2 = nn.Linear(hidden_size,output_size)

    def forward(self,input):
        return self.p2(self.act(self.p1(input)))



class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,proj_img):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.proj_img= proj_img

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        if self.proj_img:
            N,C,H,W = input.shape
            input = input.view(N,C,-1).permute(0,2,1).contiguous()#N,C,H,W --> N,C,HW --> N, HW, C
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        # if self.proj_img:
        #     output = output.permute(0,2,1).contiguous().view(N,-1,H,W)
        return output



class Channel_Spatial_Attention(nn.Module):
    def __init__(self,in_channel,hidden):
        super(Channel_Spatial_Attention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, hidden, (3, 3), 1, 1, bias=False),
                                  nn.BatchNorm2d(hidden),
                                  nn.PReLU(hidden),
                                  nn.Conv2d(hidden, hidden, (3, 3), 1, 1, bias=False),
                                  nn.BatchNorm2d(hidden))
        self.CA = CAModule(hidden,reduction=16)
        self.SA = SAModule()


    def forward(self,x):
        x = self.conv(x)
        ca =self.CA(x)
        sa =self.SA(x)
        return x

class CAModule(nn.Module):
    '''Channel Attention Module'''

    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        x = self.sigmoid(x)
        return input * x


class SAModule(nn.Module):
    '''Spatial Attention Module'''

    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return input * x


class BottleNeck_IR_CBAM(nn.Module):
    '''Improved Residual Bottleneck with Channel Attention Module and Spatial Attention Module'''

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_CBAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))

        self.channel_layer = CAModule(out_channel, 16)
        self.spatial_layer = SAModule()

        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        # Target for A-SKD
        res, att_c = self.channel_layer(res)
        res, att_s = self.spatial_layer(res)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res
