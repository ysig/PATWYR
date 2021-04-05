import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import math

class ResNetFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Input images x of handwritten text-lines, which might have
        # arbitrary lengths, are first processed by a Convolutional
        # Neural Network. We obtain an intermediate visual feature
        # representation Fc of size f. We use the ResNet50 [26] as
        # our backbone convolutional architecture. 
        # Such visual feature representation has a contextualized global view of the
        # whole input image while remaining compact.
        self.resnet = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        # From https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1))
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x.squeeze(0)

# class TransformerLayer(nn.Module):
#     def __init__(self, f=1024):
#         super().__init__()
#         self.dropout = nn.Dropout(p=0.1)
#         self.layer_norm = nn.LayerNorm(f)
#         self.multi_head = torch.nn.MultiheadAttention(f, num_heads=8)
#         self.fc = nn.Linear(f, f)

#     def forward(self, x, x1=None, x2=None):
#         xp = self.layer_norm(x)
        
#         if x1 is None:
#             x1 = xp
#         if x2 is None:
#             x2 = xp

#         x = self.multi_head(xp, x1, x2)[0]
#         x = self.dropout(x)
#         xp = xp + x
#         x = self.layer_norm(xp)
#         x = F.relu(self.fc(x))
#         x = self.dropout(x)
#         return xp + x

# class TransformerDecoder(nn.Module):
#     def __init__(self, f=1024, N=4):
#         super().__init__()
#         modules = nn.ModuleList()
#         for _ in range(N):
#             modules.append(TransformerLayer())
#         self.layers = modules

#     def forward(self, x, x1=None, x2=None):
#         for l in self.layers:
#             x = l(x, x1, x2)
#         return x

class VisualFeatureEncoder(nn.Module):
    def __init__(self, f=1024, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.resnet = ResNetFeatures()
        self.fc = nn.Linear(f*4, f)
        self.pe = PositionalEncoding(f)
        self.fc_bar = nn.Linear(f, f)
        # self.trans = TransformerDecoder(f)
        self.fc_hat = nn.Linear(140, 89)
        self.layer_norm = nn.LayerNorm(f)
        self.layer_norm2 = nn.LayerNorm(89)
        encoder_layers = nn.TransformerEncoderLayer(f, num_heads, f, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)


    def forward(self, x):
        # Question: input-size?
        x = self.resnet(x)
        f, h, w = x.size()
        x = x.view(f*h, w).permute(1, 0).contiguous()
        x = F.relu(self.fc(x))
        x = self.pe(x.unsqueeze(1))
        x = self.layer_norm(F.relu(self.fc_bar(x)))
        x = F.softmax(self.transformer_encoder(x), dim=2)
        x = F.relu(self.fc_hat(x.squeeze(1).permute(1, 0)))
        x = self.layer_norm2(x).permute(1, 0).unsqueeze(1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TextTranscriber(nn.Module):
    def __init__(self, dict_size=83, f=1024, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.ebl = nn.Embedding(dict_size, f)
        self.pe = PositionalEncoding(f)
        encoder_layers = nn.TransformerEncoderLayer(f, num_heads, f, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=f, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(f, dict_size)
        

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y):
        x = self.ebl(x).unsqueeze(1)
        x = self.pe(x)
        a = self.generate_square_subsequent_mask(x.size()[0])
        x = F.softmax(self.transformer_encoder(x, a), dim=2)
        x = self.transformer_decoder(y, x, a, a)
        return self.linear(x).squeeze(1)


if __name__ == "__main__":
    import torchvision
    import numpy as np
    from torchvision.transforms.functional import resize, pil_to_tensor
    import os
    import PIL

    # load two images
    def load_image(path):
        img = PIL.Image.open(os.path.join('debug-data', path))
        array = pil_to_tensor(img).squeeze(0).permute(1, 0).unsqueeze(0).float()/255.0
        return resize(array, size=64).squeeze(0).permute(1, 0)

    def load_batch_image(max_len=2227):
        # Each batch should have 
        c = 0
        batch = []
        while True:
            img = load_image('1.png')
            if c + img.size()[1] <= max_len:
                c += img.size()[1]
                batch.append(img)
            else:
                break
        batch_img = torch.cat(batch, dim=1)
        if batch_img.size()[0] != max_len:
            batch_img = nn.ZeroPad2d((0, max_len-batch_img.size()[1], 0, 0))(batch_img)
        return batch_img

    def load_text():
        inp = "A|MOVE|to|stop|Mr.|Gaitskell|from"
        return ["<S>"] + list(inp.replace("|", " ")) + ["<E>"]

    character_dict = dict()
    def get(x):
        a = character_dict.get(x, None)
        if a is None:
            idx = len(character_dict)
            character_dict[x] = idx
            return idx
        else:
            return a

    def load_batch_text(max_len=90):
        batch_text = []
        while True:
            q = load_text()
            if len(batch_text) + len(q) <= max_len:
                batch_text += q
            else:
                break
        for i in range(max_len - len(batch_text)):
            batch_text.append("<P>")
        return torch.LongTensor([get(b) for b in batch_text])

    vfe = VisualFeatureEncoder()
    tt = TextTranscriber()
    a = vfe(load_batch_image())
    bt = load_batch_text()
    b = tt(bt[0:89], a)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(b, bt[1:])
    loss.backward()
    