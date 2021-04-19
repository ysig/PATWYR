import torch
from torch import nn
from torch.utils.data import Subset, Dataset, DataLoader
from torch.nn import functional as F
import torchvision
import math
import numpy as np
from torchvision.transforms.functional import resize, pil_to_tensor
import os
import PIL

MAX_LEN = 95
ALPHABET = {' ': 0, '!': 1, '"': 2, '#': 3, '&': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '<E>': 26, '<P>': 27, '<S>': 28, '?': 29, 'A': 30, 'B': 31, 'C': 32, 'D': 33, 'E': 34, 'F': 35, 'G': 36, 'H': 37, 'I': 38, 'J': 39, 'K': 40, 'L': 41, 'M': 42, 'N': 43, 'O': 44, 'P': 45, 'Q': 46, 'R': 47, 'S': 48, 'T': 49, 'U': 50, 'V': 51, 'W': 52, 'X': 53, 'Y': 54, 'Z': 55, 'a': 56, 'b': 57, 'c': 58, 'd': 59, 'e': 60, 'f': 61, 'g': 62, 'h': 63, 'i': 64, 'j': 65, 'k': 66, 'l': 67, 'm': 68, 'n': 69, 'o': 70, 'p': 71, 'q': 72, 'r': 73, 's': 74, 't': 75, 'u': 76, 'v': 77, 'w': 78, 'x': 79, 'y': 80, 'z': 81, '|': 82}

def load_image(path, max_len=2227):
    img = PIL.Image.open(path).convert('L')
    array = torch.Tensor(np.array(img)).unsqueeze(0).permute(0, 2, 1).float()/255.0
    img = resize(array, size=64).permute(0, 2, 1)
    a = nn.ZeroPad2d((0, max_len-img.size()[2], 0, 0))(img)
    return a

def gen_alphabet(data):
    data_ = set()
    for _, y in data:
        data_ |= set(y)
    data_.add('<S>')
    data_.add('<E>')
    data_.add('<P>')
    return {d: i for i, d in enumerate(sorted(list(data_)))}

def read_lines_text(annotation_txt):
    data = []
    with open(annotation_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line.startswith('#'):
                continue
            else:
                spl = line.split(' ')
                image_dir = spl[0].split('-')
                data.append((os.path.join(image_dir[0], image_dir[0]+'-'+image_dir[1], spl[0]+'.png'), ' '.join(spl[8:])))
    return data

def load_text(inp):
    txt = ["<S>"] + list(inp) + ["<E>"]
    for i in range(MAX_LEN + 1 - len(txt)):
        txt.append("<P>")
    return txt


class IAM(Dataset):
    def __init__(self, annotation_txt, image_folder, alphabet):
        self.data = read_lines_text(annotation_txt)
        self.image_folder = image_folder
        self.alphabet = alphabet

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, txt = self.data[idx]
        img = load_image(os.path.join(self.image_folder, img_path))
        txt = torch.LongTensor([self.alphabet[t] for t in load_text(txt)]).unsqueeze(0)
        return img, txt

def synthetic_make_date(image_folder):
    return [(f, str(os.path.splitext(f)[0].split('_')[0]).replace(" ", "|")) for f in os.listdir(image_folder) if len(os.path.splitext(f)[0]) < MAX_LEN-2]

class Synthetic(IAM):
    def __init__(self, image_folder, alphabet):
        self.data = synthetic_make_date(image_folder)
        self.image_folder = image_folder
        self.alphabet = alphabet

def make_dataloader(dataset, batch_size, num_workers, pin_memory, subset_indices=None):
    if subset_indices is not None:
        dataset = Subset(dataset, subset_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
