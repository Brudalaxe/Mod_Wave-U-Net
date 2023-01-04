# Defining the architecture

use_cuda = torch.cuda.is_available()

class DryWaveunet(nn.Module):
    def __init__(self):
        super(DryWaveunet, self).__init__()
        self.enc_num_layers = 10
        self.dec_num_layers = 10
        self.enc_filter_size = 15
        self.dec_filter_size = 1
        self.input_channel = 8
        self.nfilters = 48

        enc_channel_in = [self.input_channel] + [min(self.dec_num_layers, (i + 1)) * self.nfilters for i in range(self.enc_num_layers - 1)]
        enc_channel_out = [min(self.dec_num_layers, (i + 1)) * self.nfilters for i in range(self.enc_num_layers)]
        dec_channel_out = enc_channel_out[:self.dec_num_layers][::-1]
        dec_channel_in = [enc_channel_out[-1]*2 + self.nfilters] + [enc_channel_out[-i-1] + dec_channel_out[i-1] for i in range(1, self.dec_num_layers)]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(self.enc_num_layers):
            self.encoder.append(nn.Conv1d(enc_channel_in[i], enc_channel_out[i], self.enc_filter_size))

        for i in range(self.dec_num_layers):
            self.decoder.append(nn.Conv1d(dec_channel_in[i], dec_channel_out[i], self.dec_filter_size))

        self.middle_layer = nn.Sequential(
            nn.Conv1d(enc_channel_out[-1], enc_channel_out[-1] + self.nfilters, self.enc_filter_size),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.nfilters + self.input_channel, 2, kernel_size=1)
        )

    def forward(self,x):
        encoder = list()
        input = x

        # Downsampling
        for i in range(self.enc_num_layers):
            x = self.encoder[i](x)
            x = F.leaky_relu(x,0.2)
            encoder.append(x) # Append for concatenation
            x = x[:,:,::2] # Decimation

        x = self.middle_layer(x)

        # Upsampling
        for i in range(self.dec_num_layers):
            x = F.interpolate(x, size=x.shape[-1]*2-1, mode='linear', align_corners=True)
            x = self.crop_and_concat(x, encoder[self.enc_num_layers - i - 1])
            x = self.decoder[i](x)
            x = F.leaky_relu(x,0.2)

        # Concat with original input
        x = self.crop_and_concat(x, input)

        # Output prediction
        output = self.output_layer(x)
        return output

    def crop_and_concat(self, x1, x2):
        crop_x2 = self.crop(x2, x1.shape[-1])
        x = torch.cat([x1,crop_x2],dim=1)
        return x

    def crop(self, tensor, target_shape):
        # Center crop
        shape = tensor.shape[-1]
        diff = shape - target_shape
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:,:,crop_start:-crop_end]
