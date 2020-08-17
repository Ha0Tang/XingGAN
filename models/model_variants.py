import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F


class XingBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(XingBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=cated_stream2)

        self.query_conv = nn.Conv2d(in_channels=256, out_channels=256//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=256, out_channels=256//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cated_stream2:
            conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim*2),
                       nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        # print('x2_out', x2_out.size()) [32, 256, 32, 16]

        # Update Image Branch
        m_batchsize, C, height, width = x1_out.size()
        proj_query = self.query_conv(x1_out).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # print('proj_query', proj_query.size()) [32, 512, 32]
        proj_key = self.key_conv(x2_out).view(m_batchsize, -1, width * height)
        # print('proj_key', proj_key.size()) [32, 32, 512]
        energy = torch.bmm(proj_query, proj_key)
        # print('energy', energy.size()) [32, 512, 512]
        attention = self.softmax(energy)
        # print('attention', attention.size()) [32, 512, 512]
        proj_value = self.value_conv(x1_out).view(m_batchsize, -1, width * height)
        # print('proj_value', proj_value.size()) [32, 256, 512]

        x1_out_1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        x1_out_1 = x1_out_1.view(m_batchsize, C, height, width)
        # print('x1_out', x1_out.size()) [32, 256, 32, 16]
        x1_out_update = x1_out + self.gamma*x1_out_1  # connection


        m_batchsize, C, height, width = x2_out.size()
        proj_query = self.query_conv(x2_out).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # print('proj_query', proj_query.size()) [32, 512, 32]
        proj_key = self.key_conv(x1_out).view(m_batchsize, -1, width * height)
        # print('proj_key', proj_key.size()) [32, 32, 512]
        energy = torch.bmm(proj_query, proj_key)
        # print('energy', energy.size()) [32, 512, 512]
        attention = self.softmax(energy)
        # print('attention', attention.size()) [32, 512, 512]
        proj_value = self.value_conv(x2_out).view(m_batchsize, -1, width * height)
        # print('proj_value', proj_value.size()) [32, 256, 512]

        x2_out_1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        x2_out_1 = x2_out_1.view(m_batchsize, C, height, width)
        # print('x1_out', x1_out.size()) [32, 256, 32, 16]
        x2_out_update = x2_out + self.gamma*x2_out_1  # connection
        # Update Image Branch

        # Update Skeleton Branch
        x2_out_update = torch.cat((x1_out_update, x2_out_update), 1)

        # print('after x2_out', x2_out.size())[32, 512, 32, 16]
        return x1_out_update, x2_out_update

class XingModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert(n_blocks >= 0 and type(input_nc) == list)
        super(XingModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream1_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]

        # att_block in place of res_block
        mult = 2**n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(XingBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i]))

        # up_sample
        model_stream1_up = []
        model_stream2_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]

            model_stream2_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, 30, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]

        model_stream2_up += [nn.ReflectionPad2d(3)]
        model_stream2_up += [nn.Conv2d(ngf, 30, kernel_size=7, padding=0)]
        model_stream2_up += [nn.Tanh()]

        # self.model = nn.Sequential(*model)
        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        # self.att = nn.Sequential(*attBlock)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)
        self.stream2_up = nn.Sequential(*model_stream2_up)

        self.x2_con = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.x2_norm = norm_layer(256)
        self.x2_relu = nn.ReLU(True)

        self.atte_con = nn.ConvTranspose2d(512+256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.atte_norm = norm_layer(256)
        self.atte_relu = nn.ReLU(True)
        self.atte_con1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.atte_norm1 = norm_layer(128)
        self.atte_relu1 = nn.ReLU(True)
        self.atte_con2 = nn.ConvTranspose2d(128, 21, kernel_size=1, stride=1, padding=0, bias=use_bias)

    def forward(self, input): # x from stream 1 and stream 2
        # here x should be a tuple
        image, x2 = input
        #print('x1',x1.size()) [32, 3, 128, 64]
        # down_sample
        x1 = self.stream1_down(image)
        x2 = self.stream2_down(x2)
        # att_block
        for model in self.att:
            x1, x2 = model(x1, x2)

        # print('x1', x1.size()) [32, 256, 32, 16]
        # print('x2', x2.size()) [32, 512, 32, 16]
        attention = torch.cat((x1, x2), 1)
        #print('attention', attention.size()) [32, 768, 32, 16]
        attention = self.atte_con(attention)
        #print('attention',attention.size()) [32, 256, 64, 32]
        attention = self.atte_norm(attention)
        attention = self.atte_relu(attention)
        attention = self.atte_con1(attention)
        attention = self.atte_norm1(attention)
        attention = self.atte_relu1(attention)
        attention = self.atte_con2(attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)
        attention1 = attention[:,0:1,:,:]
        attention2 = attention[:,1:2,:,:]
        attention3 = attention[:,2:3,:,:]
        attention4 = attention[:,3:4,:,:]
        attention5 = attention[:,4:5,:,:]
        attention6 = attention[:,5:6,:,:]
        attention7 = attention[:,6:7,:,:]
        attention8 = attention[:,7:8,:,:]
        attention9 = attention[:,8:9,:,:]
        attention10 = attention[:,9:10,:,:]
        attention11 = attention[:,10:11,:,:]
        attention12 = attention[:,11:12,:,:]
        attention13 = attention[:,12:13,:,:]
        attention14 = attention[:,13:14,:,:]
        attention15 = attention[:,14:15,:,:]
        attention16 = attention[:,15:16,:,:]
        attention17 = attention[:,16:17,:,:]
        attention18 = attention[:,17:18,:,:]
        attention19 = attention[:,18:19,:,:]
        attention20 = attention[:,19:20,:,:]
        attention21 = attention[:,20:21,:,:]

        attention1 = attention1.repeat(1,3,1,1)
        attention2 = attention2.repeat(1,3,1,1)
        attention3 = attention3.repeat(1,3,1,1)
        attention4 = attention4.repeat(1,3,1,1)
        attention5 = attention5.repeat(1,3,1,1)
        attention6 = attention6.repeat(1,3,1,1)
        attention7 = attention7.repeat(1,3,1,1)
        attention8 = attention8.repeat(1,3,1,1)
        attention9 = attention9.repeat(1,3,1,1)
        attention10 = attention10.repeat(1,3,1,1)
        attention11 = attention11.repeat(1,3,1,1)
        attention12 = attention12.repeat(1,3,1,1)
        attention13 = attention13.repeat(1,3,1,1)
        attention14 = attention14.repeat(1,3,1,1)
        attention15 = attention15.repeat(1,3,1,1)
        attention16 = attention16.repeat(1,3,1,1)
        attention17 = attention17.repeat(1,3,1,1)
        attention18 = attention18.repeat(1,3,1,1)
        attention19 = attention19.repeat(1,3,1,1)
        attention20 = attention20.repeat(1,3,1,1)
        attention21 = attention21.repeat(1,3,1,1)

        # up_sample
        x1 = self.stream1_up(x1)
        image1 = x1[:, 0:3, :, :]
        image2 = x1[:, 3:6, :, :]
        image3 = x1[:, 6:9, :, :]
        image4 = x1[:, 9:12, :, :]
        image5 = x1[:, 12:15, :, :]
        image6 = x1[:, 15:18, :, :]
        image7 = x1[:, 18:21, :, :]
        image8 = x1[:, 21:24, :, :]
        image9 = x1[:, 24:27, :, :]
        image10 = x1[:, 27:30, :, :]


        x2 = self.x2_con(x2)
        x2 = self.x2_norm(x2)
        x2 = self.x2_relu(x2)
        x2 = self.stream2_up(x2)

        image11 = x2[:, 0:3, :, :]
        image12 = x2[:, 3:6, :, :]
        image13 = x2[:, 6:9, :, :]
        image14 = x2[:, 9:12, :, :]
        image15 = x2[:, 12:15, :, :]
        image16 = x2[:, 15:18, :, :]
        image17 = x2[:, 18:21, :, :]
        image18 = x2[:, 21:24, :, :]
        image19 = x2[:, 24:27, :, :]
        image20 = x2[:, 27:30, :, :]

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        output10 = image10 * attention10
        output11 = image11 * attention11
        output12 = image12 * attention12
        output13 = image13 * attention13
        output14 = image14 * attention14
        output15 = image15 * attention15
        output16 = image16 * attention16
        output17 = image17 * attention17
        output18 = image18 * attention18
        output19 = image19 * attention19
        output20 = image20 * attention20
        output21 = image * attention21

        return output1+ output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10 +\
               output11 + output12 + output13 + output14 + output15 + output16 + output17 + output18 + output19 + output20 + output21


class XingNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(XingNetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = XingModel(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)






