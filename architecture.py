import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import math
from torchvision.models.resnet import Bottleneck
from dataset import load_image, load_text, ALPHABET, MAX_LEN


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
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        # self.layer4 = self.resnet._make_layer(Bottleneck, 512, 3, stride=1, dilate=False)

    def forward(self, x):
        # From https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x.repeat(1, 3, 1, 1))
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerHTR(nn.Module):
    def __init__(self, alphabet, dict_size=83, f=1024, num_layers=4, num_heads=8, dropout=0.1, text_len=100):
        super(TransformerHTR, self).__init__()
        # (Visual Feature) Encoder
        self.resnet = ResNetFeatures()
        self.fc = nn.Linear(f*4, f)
        self.pe_encode = PositionalEncoding(f, 140, dropout)
        self.fc_bar = nn.Linear(f, f)
        encoder_layers = nn.TransformerEncoderLayer(f, num_heads, f, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # (Text Transcriber) Decoder
        self.ebl = nn.Embedding(dict_size, f)
        self.pe_decode = PositionalEncoding(f, text_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=f, nhead=num_heads, dim_feedforward=f, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(f, dict_size)

        # General
        self.f = f
        self.text_len = text_len
        self.alphabet = alphabet
        self.inv_alphabet = {j: i for i, j in alphabet.items()}
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc_bar.bias.data.zero_()
        self.fc_bar.weight.data.uniform_(-initrange, initrange)
        self.ebl.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, x):
        x = self.resnet(x)
        b, f, h, w = x.size()
        x = x.view(b, f*h, w).permute(0, 2, 1)
        x = F.relu(self.fc(x))
        x = self.pe_encode(x.permute(1, 0, 2))
        x = F.relu(self.fc_bar(x))
        x = self.transformer_encoder(x)
        return x

    def decode(self, x, y):
        kpm = (x == self.alphabet['<P>']).transpose(1, 0)
        x = self.ebl(x)*math.sqrt(self.f)
        x = self.pe_decode(x)
        dim = x.size()[0]
        a = self.generate_square_subsequent_mask(dim).to(x.device)
        x = self.transformer_decoder(x, y, a, tgt_key_padding_mask=kpm)
        return self.linear(x).permute(1, 0, 2)

    def forward(self, x, y):
        return self.decode(x, self.encode(y))

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
        y = self.encode(y)
        output_tokens = torch.full((y.size()[1], self.text_len), self.alphabet["<P>"]).long()
        output_tokens[:, 0] = self.alphabet["<S>"]
        output_tokens = output_tokens.to(y.device)
        for j in range(1, self.text_len):
            x = output_tokens[:, :j].permute(1, 0)
            x = self.decode(x, y)
            a = torch.argmax(x, dim=-1)
            output_tokens[:, j] = a[:,-1]
        return self.to_text(output_tokens)


# DEBUG
import os
import torchvision
import numpy as np
from torchvision.transforms.functional import resize, pil_to_tensor
import PIL

def load_batch_image(max_img=2):
    # Each batch should have 
    return torch.cat([load_image(os.path.join('debug-data', f"{i}.png")) for i in range(1, max_img+1)], dim=0).unsqueeze(1)

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
    transformer = TransformerHTR(ALPHABET, text_len=MAX_LEN)
    bt = load_batch_text()
    print(bt.size())
    b = transformer(bt[0:transformer.text_len, :], load_batch_image())
    criterion = nn.CrossEntropyLoss()
    loss = 0
    trgt = bt[1:, :]
    for i in range(trgt.size()[1]):
        loss += criterion(b[i], trgt[:, i])
    loss.backward()
    out = transformer.gen(load_batch_image())
    print(out)