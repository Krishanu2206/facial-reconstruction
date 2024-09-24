import torch
from torch import nn


def cnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0, first_layer = False):

    if first_layer:
        return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
            )

def tcnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0, first_layer = False):
    if first_layer:
        return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding)

    else:
        return nn.Sequential(
           nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )
       
       
class Generator(nn.Module):
    def __init__(self, c_dim: int = 7, gf_dim:int = 64):#input : 256x256
        super(Generator,self).__init__()
        self.e1 = cnn_block(c_dim,gf_dim,4,2,1, first_layer = True)
        self.e2 = cnn_block(gf_dim,gf_dim*2,4,2,1,)
        self.e3 = cnn_block(gf_dim*2,gf_dim*4,4,2,1,)
        self.e4 = cnn_block(gf_dim*4,gf_dim*8,4,2,1,)
        self.e5 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
        self.e6 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
        self.e7 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,)
        self.e8 = cnn_block(gf_dim*8,gf_dim*8,4,2,1, first_layer=True)

        self.d1 = tcnn_block(gf_dim*8,gf_dim*8,4,2,1)
        self.d2 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
        self.d3 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
        self.d4 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1)
        self.d5 = tcnn_block(gf_dim*8*2,gf_dim*4,4,2,1)
        self.d6 = tcnn_block(gf_dim*4*2,gf_dim*2,4,2,1)
        self.d7 = tcnn_block(gf_dim*2*2,gf_dim*1,4,2,1)
        self.d8 = tcnn_block(gf_dim*1*2,c_dim,4,2,1, first_layer = True)#256x256
        self.tanh = nn.Tanh()

    def forward(self,x:torch.Tensor):
        x = x.type(torch.float32)
        e1 = self.e1(x)
        e2 = self.e2(nn.LeakyReLU(0.2)(e1))
        e3 = self.e3(nn.LeakyReLU(0.2)(e2))
        e4 = self.e4(nn.LeakyReLU(0.2)(e3))
        e5 = self.e5(nn.LeakyReLU(0.2)(e4))
        e6 = self.e6(nn.LeakyReLU(0.2)(e5))
        e7 = self.e7(nn.LeakyReLU(0.2)(e6))
        e8 = self.e8(nn.LeakyReLU(0.2)(e7))
        d1 = torch.cat([nn.Dropout(0.5)(self.d1(nn.ReLU()(e8))),e7],1)
        d2 = torch.cat([nn.Dropout(0.5)(self.d2(nn.ReLU()(d1))),e6],1)
        d3 = torch.cat([nn.Dropout(0.5)(self.d3(nn.ReLU()(d2))),e5],1)
        d4 = torch.cat([self.d4(nn.ReLU()(d3)),e4],1)
        d5 = torch.cat([self.d5(nn.ReLU()(d4)),e3],1)
        d6 = torch.cat([self.d6(nn.ReLU()(d5)),e2],1)
        d7 = torch.cat([self.d7(nn.ReLU()(d6)),e1],1)
        d8 = self.d8(nn.ReLU()(d7))

        return self.tanh(d8)[:, :3, :, :]
    
    
    
class Discriminator(nn.Module):
    def __init__(self, c_dim: int = 3, df_dim:int = 64):#input : 256x256
        super(Discriminator,self).__init__()
        self.conv1 = cnn_block(c_dim*2,df_dim,4,2,1, first_layer=True) # 128x128
        self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)# 64x64
        self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)# 32 x 32
        self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)# 31 x 31
        self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)# 30 x 30

        self.sigmoid = nn.Sigmoid()
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        x, y = x.type(torch.float32), y.type(torch.float32)
        O = torch.cat([x,y],dim=1)
        if O.shape[1] != 6:
            print( f"{O.shape}, {x.shape},{ y.shape}")
        O = nn.LeakyReLU(0.2)(self.conv1(O))
        O = nn.LeakyReLU(0.2)(self.conv2(O))
        O = nn.LeakyReLU(0.2)(self.conv3(O))
        O = nn.LeakyReLU(0.2)(self.conv4(O))
        O = self.conv5(O)

        return self.sigmoid(O)