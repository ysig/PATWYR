import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import math
from dataset import load_image

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
        x = self.resnet.conv1(x.unsqueeze(1).repeat(1, 3, 1, 1))
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
        return x

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
        b, f, h, w = x.size()
        x = x.view(b, f*h, w).permute(0, 2, 1).contiguous()
        x = F.relu(self.fc(x))
        x = self.pe(x.permute(1, 0, 2).contiguous())
        x = self.layer_norm(F.relu(self.fc_bar(x)))
        x = F.softmax(self.transformer_encoder(x), dim=2)
        x = F.relu(self.fc_hat(x.permute(2, 1, 0)))
        x = self.layer_norm2(x).permute(2, 1, 0)
        return x


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
    def __init__(self, alphabet, dict_size=83, f=1024, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.ebl = nn.Embedding(dict_size, f)
        self.pe = PositionalEncoding(f)
        encoder_layers = nn.TransformerEncoderLayer(f, num_heads, f, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=f, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(f, dict_size)
        self.alphabet = alphabet
        self.inv_alphabet = {j: i for i, j in alphabet.items()}
        

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y):
        x = self.ebl(x)
        x = self.pe(x)
        a = self.generate_square_subsequent_mask(x.size()[0])
        x = F.softmax(self.transformer_encoder(x, a), dim=2)
        x = self.transformer_decoder(y, x, a, a)
        return self.linear(x).permute(1, 0, 2).contiguous()

    @torch.no_grad()
    def to_text_(self, x):
        txt = []
        p = {self.alphabet["<F>"], self.alphabet["<P>"]}
        s = self.alphabet["<S>"]
        for idx in x.cpu().numpy():
            if idx in p:
                break
            if idx == s:
                continue
            txt.append(self.inv_alphabet[idx])
        return "".join(txt)

    @torch.no_grad()
    def to_text(self, x):
        if len(x.size()) == 2:
            return [self.to_text_(x[i]) for i in range(x.size()[0])]
        else:
            return self.to_text_(x)

    @torch.no_grad()
    def gen(self, y):
        out = []
        fs = self.alphabet["<F>"]
        for i in range(y.size()[1]):
            img = y[:,i].unsqueeze(1)
            xp = torch.LongTensor([alphabet["<S>"]])
            for j in range(89):
                x = self.ebl(xp)
                x = self.pe(x)
                x = F.softmax(self.transformer_encoder(x), dim=2)
                x = self.transformer_decoder(img[:j+1, :, :], x)
                x = self.linear(x).permute(1, 0, 2).contiguous()
                a = torch.argmax(x, keepdim=True, dim=2).squeeze(2).squeeze(0)
                xp = torch.cat([torch.LongTensor([alphabet["<S>"]]), a], dim=0)
                if xp[-1] == fs:
                    break
            out.append(self.to_text(xp))
        return out


if __name__ == "__main__":
    import torchvision
    import numpy as np
    from torchvision.transforms.functional import resize, pil_to_tensor
    import os
    import PIL

    # load two images

    def load_batch_image():
        # Each batch should have 
        return torch.cat([load_image(os.path.join('debug-data', f"{i}.png")) for i in range(1, 3)], dim=0)

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
    def load_text(i, max_len=90):
        inp = TXT[i]
        txt = ["<S>"] + list(inp) + ["<E>"]
        for i in range(max_len - len(txt)):
            txt.append("<P>")        
        t = torch.LongTensor([get(b) for b in txt])
        return t.unsqueeze(1)

    def load_batch_text():
        return torch.cat([load_text(i) for i in range(2)], dim=1)

    alphabet = {' ': 0, '!': 1, '"': 2, '#': 3, '&': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '<F>': 26, '<P>': 27, '<S>': 28, '?': 29, 'A': 30, 'B': 31, 'C': 32, 'D': 33, 'E': 34, 'F': 35, 'G': 36, 'H': 37, 'I': 38, 'J': 39, 'K': 40, 'L': 41, 'M': 42, 'N': 43, 'O': 44, 'P': 45, 'Q': 46, 'R': 47, 'S': 48, 'T': 49, 'U': 50, 'V': 51, 'W': 52, 'X': 53, 'Y': 54, 'Z': 55, 'a': 56, 'b': 57, 'c': 58, 'd': 59, 'e': 60, 'f': 61, 'g': 62, 'h': 63, 'i': 64, 'j': 65, 'k': 66, 'l': 67, 'm': 68, 'n': 69, 'o': 70, 'p': 71, 'q': 72, 'r': 73, 's': 74, 't': 75, 'u': 76, 'v': 77, 'w': 78, 'x': 79, 'y': 80, 'z': 81, '|': 82}
    vfe = VisualFeatureEncoder()
    tt = TextTranscriber(alphabet)
    a = vfe(load_batch_image())
    bt = load_batch_text()
    b = tt(bt[0:89, :], a)
    criterion = nn.CrossEntropyLoss()
    cs, bs = bt[1:, :].size()
    N = cs*bs
    loss = criterion(b.view(N, -1), bt[1:, :].view(N))
    loss.backward()
    out = tt.gen(a)
    print(out)