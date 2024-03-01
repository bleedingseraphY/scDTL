import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np

#import scipy.io as sio
from copy import deepcopy
        
        
class AEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(AEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Sequential(
                            nn.Linear(hidden_dims[-1], latent_dim)
                                       ,nn.ReLU()
                            )

        # Build Decoder
        modules = []

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Sigmoid(),
            nn.Dropout(drop_out)
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Sigmoid(),
                    nn.Dropout(drop_out))
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1])
                                       ,nn.Sigmoid()
                            )
        # self.feature_extractor =nn.Sequential(
        #     self.encoder,
        #     self.bottleneck
        # )            

             
    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding = self.bottleneck(result)

        return embedding

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return  output        

# Model of Imputor
class Imputor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(Imputor, self).__init__()

        modules = []

        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Sigmoid(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.predictor = nn.Sequential(*modules)
        #self.output = nn.Linear(hidden_dims[-1], output_dim)

        self.output = nn.Sequential(
                                nn.Linear(hidden_dims[-1],output_dim),
                                nn.BatchNorm1d(output_dim),
                                nn.Sigmoid()
                            )            

    def forward(self, input: Tensor, **kwargs):
        embedding = self.predictor(input)
        output = self.output(embedding)
        return  output

# Model of Pretrained I
class PretrainedImputor(AEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False):
        
        # Construct an autoencoder model
        AEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            bottlenect_reached = False
            for p in self.parameters():
                if ((bottlenect_reached == True)&(p.shape.numel()>self.latent_dim)):
                    break
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
                if p.shape.numel() == self.latent_dim:
                    bottlenect_reached = True
        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Imputor(input_dim=self.latent_dim,
                                 output_dim=output_dim,
                                 h_dims=hidden_dims_predictor,
                                 drop_out=drop_out_predictor)

    def forward(self, input, **kwargs):
        embedding = self.encode(input)
        output = self.predictor(embedding)
        return output
   
    def predict(self, embedding, **kwargs):
        output = self.predictor(embedding)
        return  output 

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.Sigmoid(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):

        x = self.conv(x)
        return x

class conv_decoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):

    def __init__(self, filters=[64,128,256,512], drop_out=0.1):
        super(U_Net, self).__init__()

        #
        self.filters = filters

        hidden_dims = deepcopy(filters)

        modules = []
        # Build Encoder
        for i in range(1,len(hidden_dims)-1):
            # i_dim = hidden_dims[i]
            # o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    conv_block(hidden_dims[i-1], hidden_dims[i])
                )
            )
            # in_channels = h_dim


        # left
        self.Conv1 = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid_1 = nn.Sigmoid()
        self.Conv2 = conv_block(1, filters[0])
        self.encoder = nn.Sequential(*modules)

        self.bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(filters[-2], filters[-1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(filters[-1]),
            nn.Sigmoid()
        )

        decoder_modules = []
        hidden_dims.reverse()
        # Build Decoder
        for i in range(1,len(hidden_dims)):
            # i_dim = hidden_dims[i - 1]
            # o_dim = hidden_dims[i]

            decoder_modules.append(
                nn.Sequential(
                    up_conv(hidden_dims[i-1], hidden_dims[i]),
                    conv_decoder(hidden_dims[i-1], hidden_dims[i])
                )
            )
            # in_channels = h_dim

        self.decoder = nn.Sequential(*decoder_modules)

        self.Conv_o1 = nn.Conv1d(filters[0], 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid_2 = nn.Sigmoid()
        self.Conv_o2 = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid_3 = nn.Sigmoid()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        e1 = self.Conv2(x)
        e_index = []
        e_index.append(e1)
        for i in range(len(self.filters)-2):
            new_e = self.encoder[i](e_index[i])
            e_index.append(new_e)
        new_e = self.bottleneck(e_index[-1])
        e_index.append(new_e)

        new_d = e_index[-1]
        for i in range(len(self.filters)-1):
            new_d = self.decoder[i][0](new_d)
            new_d = torch.cat((e_index[-(2+i)], new_d), dim=1)
            new_d = self.decoder[i][1](new_d)
        out = self.Conv_o1(new_d)
        out = self.sigmoid_2(out)
        out = torch.squeeze(out, dim=1)
        return out

def vae_loss(recon_x, x, mu, logvar,reconstruction_function,weight=1):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD * weight

class VAEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(VAEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
    
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)
        
        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1],
                            nn.Sigmoid())
                            ) 
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )
    
    def encode_(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode(self, input: Tensor,repram=False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encode_(input)

        if (repram==True):
            z = self.reparameterize(mu, log_var)
            return z
        else:
            return mu

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
         M_N = self.params['batch_size']/ self.num_train_imgs,
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 
        # Account for the minibatch samples from the dataset
        # M_N = self.params['batch_size']/ self.num_train_imgs,
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class PretrainedVAEImputor(VAEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False,
                 z_reparam=True):
        
        self.z_reparam=z_reparam
        # Construct an autoencoder model
        VAEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            bottlenect_reached = False
            for p in self.parameters():
                if ((bottlenect_reached == True)&(p.shape[0]>self.latent_dim)):
                    break
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
                if p.shape[0] == self.latent_dim:
                    bottlenect_reached = True

        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Imputor(input_dim=self.latent_dim,
                                 output_dim=output_dim,
                                 h_dims=hidden_dims_predictor,
                                 drop_out=drop_out_predictor)

        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )

    def forward(self, input, **kwargs):
        embedding = self.encode(input,repram=self.z_reparam)
        output = self.predictor(embedding)
        return  output

    def predict(self, embedding, **kwargs):
        output = self.predictor(embedding)
        return  output

class DTL(nn.Module):
    def __init__(self, source_model,target_model,fix_source=False):
        super(DTL, self).__init__()
        self.source_model = source_model
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
        self.target_model = target_model

    def forward(self, X_source, X_target,C_target=None):
     
        x_src_mmd = self.source_model.encode(X_source)

        if(type(C_target)==type(None)):
            x_tar_mmd = self.target_model.encode(X_target)
        else:
            x_tar_mmd = self.target_model.encode(X_target,C_target)

        y_src = self.source_model.predictor(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1   = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=1):
        super(CBAM, self).__init__()
        #Channel Attention
        self.max_pool=nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool1d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #Spatial Attention
        self.conv=nn.Conv1d(in_channels=2,out_channels=1,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #Channel Attention
        maxout=self.max_pool(x)

        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1)
        channel_out=channel_out*x
        #Spatial Attention
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out
class AttentionVotor(nn.Module):
    def __init__(self, filters, in_channel):
        super(AttentionVotor, self).__init__()
        self.unet = U_Net(filters=filters)
        self.cbam = CBAM(in_channel,reduction=16,kernel_size=7)
        self.conV = nn.Conv1d(2,1,kernel_size=1)
        self.ln = nn.Linear(4000,4000)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sc_orgin_data, generate_data):

        orgin_feature = self.unet(sc_orgin_data)
        orgin_feature = torch.unsqueeze(orgin_feature, dim=1)
        generate_data = torch.unsqueeze(generate_data, dim=1)
        combine_feature = torch.cat([orgin_feature,generate_data],dim=1)
        output = self.cbam(combine_feature)
        output = self.conV(output)
        output = torch.squeeze(output,dim=1)
        output = self.ln(output)
        return output


class TargetModel(nn.Module):
    def __init__(self, source_predcitor,target_encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = source_predcitor
        self.target_encoder = target_encoder

    def forward(self, X_target,C_target=None):

        if(type(C_target)==type(None)):
            x_tar = self.target_encoder.encode(X_target)
        else:
            x_tar = self.target_encoder.encode(X_target,C_target)
        y_src = self.source_predcitor.predictor(x_tar)
        return y_src

def g_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD