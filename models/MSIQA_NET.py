import torch
import torch.nn as nn
import models.MSIQA_SWIN as swin
import torch.nn.init as init
import models.MSIQA_RESNET as resnet50

class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        self.A0 = torch.eye(hide_channel)
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = x.device
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)

        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)

        self.A0 = self.A0.to(device)
        self.A2 = self.A2.to(device)

        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.conv3(y)
        y = self.relu(y)
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y


class MSIQA_NET(nn.Module):
    def __init__(self, num_outputs=1):
        super(MSIQA_NET, self).__init__()
        self.swin_transformer = swin.SwinTransformerNet().cuda()
        self.resnet = resnet50.ResNet50Net().cuda()
        self.AGCA_block1 = AGCA(176, 4)
        self.AGCA_block2 = AGCA(352, 4)
        self.AGCA_block3 = AGCA(704, 4)
        self.AGCA_block4 = AGCA(1408, 4)
        self.AGCA_block5 = AGCA(768, 4)

        self.conv1 = nn.Sequential(
            nn.Conv2d(352, 176, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(7)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(704, 352, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(7)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1408, 704, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(7)
        )
        self.conv4 = nn.Conv2d(2816, 1408, kernel_size=1, stride=1)
        self.global_pool2d1 = nn.AdaptiveMaxPool2d(1)
        self.global_pool2d2 = nn.AdaptiveAvgPool2d(1)
        self.global_pool1d = nn.AdaptiveMaxPool1d(768)
        self.fc_score = nn.Sequential(
            nn.Linear(768, 384),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(384, num_outputs),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(768, 384),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(384, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, img):
        res_f = self.resnet(img)
        st_f = self.swin_transformer(img)
        feat1 = torch.cat((res_f['Stage1'], st_f['Stage1'].permute(0, 3, 1, 2)), dim=1)
        feat2 = torch.cat((res_f['Stage2'], st_f['Stage2'].permute(0, 3, 1, 2)), dim=1)
        feat3 = torch.cat((res_f['Stage3'], st_f['Stage3'].permute(0, 3, 1, 2)), dim=1)
        feat4 = torch.cat((res_f['Stage4'], st_f['Stage4'].permute(0, 3, 1, 2)), dim=1)

        feat1 = self.conv1(feat1)
        feat2 = self.conv2(feat2)
        feat3 = self.conv3(feat3)
        feat4 = self.conv4(feat4)
        feat1 = self.AGCA_block1(feat1)
        feat2 = self.AGCA_block2(feat2)
        feat3 = self.AGCA_block3(feat3)
        feat4 = self.AGCA_block4(feat4)
        fused_feat = torch.cat((feat1, feat2, feat3, feat4), dim=1)

        swin_final_feat = st_f['Stage4'].permute(0, 3, 1, 2)
        swin_final_feat = self.AGCA_block5(swin_final_feat)

        fused_feat = self.global_pool2d1(fused_feat)
        fused_feat = fused_feat.view(fused_feat.shape[0], -1)
        swin_final_feat = self.global_pool2d2(swin_final_feat)
        swin_final_feat = swin_final_feat.view(swin_final_feat.shape[0], -1)
        fused_feat = self.global_pool1d(fused_feat)

        score = torch.tensor([]).cuda()
        for i in range(fused_feat.shape[0]):
            f = self.fc_score(swin_final_feat[i])
            w = self.fc_weight(fused_feat[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score