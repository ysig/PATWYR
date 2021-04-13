import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import math
from torchvision.models.resnet import Bottleneck
from dataset import load_image, load_text, ALPHABET, MAX_LEN


class ResNetFeatures(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # Input images x of handwritten text-lines, which might have
        # arbitrary lengths, are first processed by a Convolutional
        # Neural Network. We obtain an intermediate visual feature
        # representation Fc of size f. We use the ResNet50 [26] as
        # our backbone convolutional architecture. 
        # Such visual feature representation has a contextualized global view of the
        # whole input image while remaining compact.
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        # self.layer4 = self.resnet._make_layer(Bottleneck, 512, 3, stride=1, dilate=False)

    def forward(self, x):
        # From https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x.repeat(1, 3, 1, 1))
        x = self.resnet.bn1(x)
        # print(x.size())
        x = self.resnet.relu(x)
        # print(x.size())
        x = self.resnet.maxpool(x)
        # print(x.size())
        x = self.resnet.layer1(x)
        # print(x.size())
        x = self.resnet.layer2(x)
        # print(x.size())
        x = self.resnet.layer3(x)
        # print(x.size())
        # print(self.resnet.layer3)
        # print(self.resnet.layer4(x).size())
        # x = self.layer3(x)
        # x = self.layer4(x)
        # print(x.size())
        
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

class VisualFeatureEncoder(nn.Module):
    def __init__(self, f=1024, num_heads=8, num_layers=4, dropout=0.1, text_len=100):
        super(VisualFeatureEncoder, self).__init__()
        self.resnet = ResNetFeatures()
        self.fc = nn.Linear(f*4, f)
        self.pe = PositionalEncoding(f)
        self.fc_bar = nn.Linear(f, f)
        # self.trans = TransformerDecoder(f)
        # self.fc_hat = nn.Linear(140, text_len)
        self.layer_norm = nn.LayerNorm(f)
        # self.layer_norm2 = nn.LayerNorm(text_len)
        # self.layer_norm2 = nn.LayerNorm(140)
        encoder_layers = nn.TransformerEncoderLayer(f, num_heads, f, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)


    def forward(self, x):
        # Question: input-size?
        x = self.resnet(x)
        b, f, h, w = x.size()
        x = x.view(b, f*h, w).permute(0, 2, 1).contiguous()
        x = F.relu(self.fc(x))
        # x = self.fc(x)
        x = self.pe(x.permute(1, 0, 2).contiguous())
        x = self.layer_norm(F.relu(self.fc_bar(x)))
        # x = self.layer_norm(self.fc_bar(x))
        # x = F.softmax(self.transformer_encoder(x), dim=2)
        x = self.transformer_encoder(x)
        # x = F.relu(self.fc_hat(x.permute(2, 1, 0)))
        # x = self.layer_norm2(x).permute(2, 1, 0)
        return self.layer_norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2228):
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
    def __init__(self, alphabet, dict_size=83, f=1024, num_layers=4, num_heads=8, dropout=0.1, text_len=100):
        super(TextTranscriber, self).__init__()
        self.ebl = nn.Embedding(dict_size, f)
        self.pe = PositionalEncoding(f)
        # encoder_layers = nn.TransformerEncoderLayer(f, num_heads, f, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=f, nhead=num_heads, dim_feedforward=f, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(f, dict_size)
        self.alphabet = alphabet
        self.f = f
        self.inv_alphabet = {j: i for i, j in alphabet.items()}
        self.text_len = text_len
        self.init_weights()
        self.mask_x = self.generate_square_subsequent_mask(text_len)

    def init_weights(self):
        initrange = 0.1
        self.ebl.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y):
        x = self.ebl(x)*math.sqrt(self.f)
        x = self.pe(x)
        dim = x.size()[0]
        a = self.mask_x[:dim, :dim]
        # x = F.softmax(self.transformer_encoder(x, a), dim=2)
        # x = self.transformer_encoder(x, a)
        x = self.transformer_decoder(x, y, a)
        # print(x.size())
        return self.linear(x).permute(1, 0, 2)#.contiguous()

    @torch.no_grad()
    def to_text_(self, x):
        txt = []
        p = {self.alphabet["<E>"]}
        s = {self.alphabet["<S>"], self.alphabet["<P>"]}
        for idx in x:
            if idx in p:
                break
            if idx in s:
                continue
            txt.append(self.inv_alphabet[idx])
        return "".join(txt)

    @torch.no_grad()
    def to_text(self, x):
        x = x.cpu().numpy()
        if len(x.shape) == 2:
            return [self.to_text_(x[i]) for i in range(x.shape[0])]
        else:
            return self.to_text_(x)

    @torch.no_grad()
    def gen(self, y):
        output_tokens = torch.full((y.size()[1], self.text_len), self.alphabet["<P>"]).long().to(y.device)
        output_tokens[:, 0] = self.alphabet["<S>"]
        for j in range(1, self.text_len):
            # xp = output_tokens[:, :j].permute(1, 0)
            x = output_tokens[:, :j].permute(1, 0)
            x = self.forward(x, y)
            # x = self.ebl(xp)*math.sqrt(self.f)
            # x = self.pe(x)
            # x = F.softmax(self.transformer_encoder(x), dim=2)
            # x = self.transformer_encoder(x)
            # x = self.transformer_decoder(x, y)
            # x = self.linear(x).permute(1, 0, 2)#.contiguous()
            a = torch.argmax(x, dim=-1)
            output_tokens[:, j] = a[:,-1]
            # if xp[-1] == fs:
            #     break
        print(output_tokens)
        return self.to_text(output_tokens)

    # cpu-only
    # @torch.no_grad()
    # def gen(self, y):
    #     out = []
    #     fs = self.alphabet["<E>"]
    #     for i in range(y.size()[1]):
    #         img = y[:,i].unsqueeze(1)
    #         xp = torch.LongTensor([self.alphabet["<S>"]]).to(y.device)
    #         for j in range(self.text_len):
    #             x = self.ebl(xp)
    #             x = self.pe(x)
    #             x = F.softmax(self.transformer_encoder(x), dim=2)
    #             x = self.transformer_decoder(img[:j+1, :, :], x)
    #             x = self.linear(x).permute(1, 0, 2).contiguous()
    #             a = torch.argmax(x, keepdim=True, dim=2).squeeze(2).squeeze(0)
    #             xp = torch.cat([torch.LongTensor([self.alphabet["<S>"]]).to(a.device), a], dim=0)
    #             if xp[-1] == fs:
    #                 break
    #         out.append(self.to_text(xp))
    #     return out

# DEBUG
import os
import torchvision
import numpy as np
from torchvision.transforms.functional import resize, pil_to_tensor
import PIL

def load_batch_image():
    # Each batch should have 
    return torch.cat([load_image(os.path.join('debug-data', f"{i}.png")) for i in range(1, 3)], dim=0).unsqueeze(1)

character_dict = dict()
def get(x):
    a = character_dict.get(x, None)
    if a is None:
        idx = len(character_dict)
        character_dict[x] = idx
        return idx
    else:
        return a

TXT = ["A|MOVE|to|stop|Mr.|Gaitskell|from", "nominating|any|more|Labour|life|Peers"]
def load_text_tensor(txt):
    return torch.LongTensor([ALPHABET[t] for t in load_text(txt)]).unsqueeze(1)

def load_batch_text():
    return torch.cat([load_text_tensor(TXT[i]) for i in range(2)], dim=1)

if __name__ == "__main__":
    # load two images
    vfe = VisualFeatureEncoder(text_len=MAX_LEN)
    tt = TextTranscriber(ALPHABET, text_len=MAX_LEN)
    a = vfe(load_batch_image())
    bt = load_batch_text()
    print(bt.size())
    b = tt(bt[0:tt.text_len, :], a)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    trgt = bt[1:, :]
    for i in range(trgt.size()[1]):
        loss += criterion(b[i], trgt[:, i])
    loss.backward()
    out = tt.gen(a)
    print(out)