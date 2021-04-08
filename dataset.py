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

def load_image(path, max_len=2227):
    img = PIL.Image.open(path)
    array = torch.Tensor(np.array(img)).unsqueeze(0).permute(0, 2, 1).float()/255.0
    img = resize(array, size=64).permute(0, 2, 1)
    a = nn.ZeroPad2d((0, max_len-img.size()[2], 0, 0))(img)
    return a

def gen_alphabet(data):
    data_ = set()
    for _, y in data:
        data_ |= set(y)
    data_.add('<S>')
    data_.add('<F>')
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

def load_text(inp, max_len=90):
    txt = ["<S>"] + list(inp) + ["<E>"]
    for i in range(max_len - len(txt)):
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
        return image, txt

def iam_dataloader(dataset, batch_size, num_workers, pin_memory, subset_indices=None):
    if subset_indices is not None:
        dataset = Subset(dataset, subset_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
