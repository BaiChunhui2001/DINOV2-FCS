#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from torch.nn import BatchNorm2d
from torch.nn import init
from .EVCBlock import EVCBlock
from .AKconv import AKConv
from .pagFM import PagFM


class SASPP (nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(SASPP, self).__init__()
        bn_mom = 0.1

        self.scale1 = nn.Sequential(
                                    nn.Conv2d(inplanes, branch_planes, 3, 1, padding=6, dilation=6, bias=True),
                                    )
        self.scale2 = nn.Sequential(
                                    nn.Conv2d(inplanes, branch_planes, 3, 1, padding=12, dilation=12, bias=True),
                                    )
        self.scale3 = nn.Sequential(
                                    nn.Conv2d(inplanes, branch_planes, 3, 1, padding=18, dilation=18, bias=True),
                                    )

        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*3, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*3, inplanes, kernel_size=3, padding=1, groups=4, bias=False),
                                    )
        
        self.akconv = AKConv(inc=inplanes,outc=inplanes, num_param=3)
        
        self.pagfm = PagFM(in_channels=inplanes,mid_channels=inplanes//2)


    def forward(self, x):
               
        scale_list = []
        
        scale_list.append(self.scale1(x))
        scale_list.append(self.scale2(x))
        scale_list.append(self.scale3(x))
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
        scale_ak = self.akconv(x)
        
        out = self.pagfm(scale_out,scale_ak)
        
        return out
    
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.init_weight()
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.init_weight()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=384, embedding_dim=384, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels = in_channels
        c2_in_channels = in_channels 
        c3_in_channels = in_channels 
        c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.conv = nn.Conv2d(embedding_dim*4,
                embedding_dim*4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(embedding_dim*4)
        
        self.evc = EVCBlock(in_channels, embedding_dim)
        
        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )
        self.saspp = SASPP(embedding_dim,embedding_dim//2,embedding_dim)
        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
        self.init_weight()
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        c5 = self.evc(c4)
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) + c5
        _c4 = F.interpolate(_c4, size=(c1.size(2)*2,c1.size(3)*2), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) + c5
        _c3 = F.interpolate(_c3, size=(c1.size(2)*2,c1.size(3)*2), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]) + c5
        _c2 = F.interpolate(_c2, size=(c1.size(2)*2,c1.size(3)*2), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) + c5
        _c1 = F.interpolate(_c1, size=(c1.size(2)*2,c1.size(3)*2), mode='bilinear', align_corners=False)

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        atten = torch.mean(_c, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        _c_atten = torch.mul(_c, atten)
        
        x = self.linear_fuse(_c_atten)
        x = self.saspp(x)
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.dinov2_vits14 = torch.hub.load('/home/bch/code/dinov2', 'dinov2_vitb14',source='local')


    def forward(self, x):
        x = self.dinov2_vits14.get_intermediate_layers(x,n = [2,5,8,11], reshape = True)
        
        return x
                    
class dinov2SegMlp2(nn.Module):

    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(dinov2SegMlp2, self).__init__()
        self.cp = ContextPath()
        self.decode_head = SegFormerHead(3, 768, 768)
        self.aux_mode = aux_mode
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x = self.cp(x)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        if self.aux_mode == 'train':
            return x
        elif self.aux_mode == 'eval':
            return x,
        elif self.aux_mode == 'pred':
            x = x.argmax(dim=1)
            return x
        else:
            raise NotImplementedError



if __name__ == "__main__":
    net = dinov2SegMlp2(3)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 518, 518).cuda()
    out = net(in_ten)
    print(out.shape)
    

    # net.get_params()
    
    # device = torch.device('cuda')
    # model = dinov2Seg(17,aux_mode='eval')
    # model.eval()
    # model.to(device)
    # iterations = None
    
    # input = torch.randn(1, 3, 1024, 1024).cuda()
    # with torch.no_grad():
    #     for _ in range(10):
    #         model(input)
    
    #     if iterations is None:
    #         elapsed_time = 0
    #         iterations = 100
    #         while elapsed_time < 1:
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             t_start = time.time()
    #             for _ in range(iterations):
    #                 model(input)
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             elapsed_time = time.time() - t_start
    #             iterations *= 2
    #         FPS = iterations / elapsed_time
    #         iterations = int(FPS * 6)
    
    #     print('=========Speed Testing=========')
    #     torch.cuda.synchronize()
    #     torch.cuda.synchronize()
    #     t_start = time.time()
    #     for _ in range(iterations):
    #         model(input)
    #     torch.cuda.synchronize()
    #     torch.cuda.synchronize()
    #     elapsed_time = time.time() - t_start
    #     latency = elapsed_time / iterations * 1000
    # torch.cuda.empty_cache()
    # FPS = 1000 / latency
    # print(FPS)
    # num_parameters = sum(p.numel() for p in model.parameters())
    # print(f"模型的参数量为: {num_parameters}") 

