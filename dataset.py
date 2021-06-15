import torch
from torch import nn
from torch.utils.data import Subset, Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
import torchvision
import math
import numpy as np
from torchvision.transforms.functional import resize, pil_to_tensor, normalize
import os
import PIL
import copy
import random
from augment import augmentor
import cv2

MAX_LEN = 95
ALPHABET = {' ': 0, '!': 1, '"': 2, '#': 3, '&': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '<E>': 26, '<P>': 27, '<S>': 28, '?': 29, 'A': 30, 'B': 31, 'C': 32, 'D': 33, 'E': 34, 'F': 35, 'G': 36, 'H': 37, 'I': 38, 'J': 39, 'K': 40, 'L': 41, 'M': 42, 'N': 43, 'O': 44, 'P': 45, 'Q': 46, 'R': 47, 'S': 48, 'T': 49, 'U': 50, 'V': 51, 'W': 52, 'X': 53, 'Y': 54, 'Z': 55, 'a': 56, 'b': 57, 'c': 58, 'd': 59, 'e': 60, 'f': 61, 'g': 62, 'h': 63, 'i': 64, 'j': 65, 'k': 66, 'l': 67, 'm': 68, 'n': 69, 'o': 70, 'p': 71, 'q': 72, 'r': 73, 's': 74, 't': 75, 'u': 76, 'v': 77, 'w': 78, 'x': 79, 'y': 80, 'z': 81, '|': 82}
ignore_files = {'a05/a05-116/a05-116-09.png'}

def load_image(path, max_len=2227, transform=False):
    array = cv2.imread(path, 0)
    if transform:
        array = cv2.resize(augmentor(array), (array.shape[1], array.shape[0]), interpolation=cv2.INTER_AREA)

    array = cv2.resize(array, (int(array.shape[1]*(64/array.shape[0])), 64), interpolation=cv2.INTER_AREA)
    if array.shape[1] >= max_len:
        array = cv2.resize(array, (max_len, 64), interpolation=cv2.INTER_AREA)    
    else:
        right = max_len - array.shape[1]
        array =  cv2.copyMakeBorder(array, 0, 0, 0, right, cv2.BORDER_CONSTANT)
    

    assert array.shape == (64, max_len)
    array = array.astype(np.float32)/255.0
    return torch.from_numpy(array).unsqueeze(0)

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
                dt = (os.path.join(image_dir[0], image_dir[0]+'-'+image_dir[1], spl[0]+'.png'), ' '.join(spl[8:]))
                # print(dt[0])
                if dt[0] not in ignore_files:
                    data.append(dt)
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
        self.transform = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, txt = self.data[idx]
        img = load_image(os.path.join(self.image_folder, img_path), transform=self.transform)
        txt = torch.LongTensor([self.alphabet[t] for t in load_text(txt)]).unsqueeze(0)
        return img, txt

    def subset(self, file):
        new_dataset = copy.deepcopy(self)
        valid_key = {str(l.strip('\n')) for l in open(file, 'r').readlines()}
        new_data = []
        for d, i in new_dataset.data:
            ds = os.path.split(os.path.split(d)[0])[1]
            if ds in valid_key:
                new_data.append((d, i))
        new_dataset.data = new_data
        return new_dataset


def synthetic_make_date(image_folder):
    return [(f, str(os.path.splitext(f)[0].split('_')[0]).replace(" ", "|")) for f in os.listdir(image_folder) if len(os.path.splitext(f)[0]) < MAX_LEN-2]

class Synthetic(IAM):
    def __init__(self, image_folder, alphabet):
        self.data = synthetic_make_date(image_folder)
        self.image_folder = image_folder
        self.alphabet = alphabet
        self.transform = False

def make_iam(dataset, batch_size, num_workers, pin_memory, split_file=None, transform=False):
    if split_file is not None:
        dataset = dataset.subset(split_file)
    dataset.transform = transform
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

def make_dataloader(dataset, batch_size, num_workers, pin_memory, subset_indices=None, transform=False):
    if subset_indices is not None:
        dataset = Subset(dataset, subset_indices)
    dataset.transform = transform
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

if __name__ == "__main__":
    import sys
    PP = os.path.dirname(__file__)
    iam = IAM(sys.argv[1], sys.argv[2], ALPHABET)
    print(len(iam.subset(os.path.join(PP, 'splits', 'train.uttlist'))))
    print(len(iam.subset(os.path.join(PP, 'splits', 'validation.uttlist'))))
    print(len(iam.subset(os.path.join(PP, 'splits', 'test.uttlist'))))
    iam[0]