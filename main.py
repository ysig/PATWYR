import math
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from tqdm import tqdm, trange
from metrics import Metrics
from architecture import VisualFeatureEncoder, TextTranscriber
from dataset import IAM, iam_dataloader
import wandb
import os

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, eps, len_A, reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing_a = float(eps/len_A)
        self.smoothing_b = float(1 - ((len_A - 1)*eps/len_A))
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

class PATWYR(object):
    def __init__(self, checkpoint=None, lr=0.0002, device="cpu", wandb=False):
        # self.device
        self.alphabet = {' ': 0, '!': 1, '"': 2, '#': 3, '&': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '<F>': 26, '<P>': 27, '<S>': 28, '?': 29, 'A': 30, 'B': 31, 'C': 32, 'D': 33, 'E': 34, 'F': 35, 'G': 36, 'H': 37, 'I': 38, 'J': 39, 'K': 40, 'L': 41, 'M': 42, 'N': 43, 'O': 44, 'P': 45, 'Q': 46, 'R': 47, 'S': 48, 'T': 49, 'U': 50, 'V': 51, 'W': 52, 'X': 53, 'Y': 54, 'Z': 55, 'a': 56, 'b': 57, 'c': 58, 'd': 59, 'e': 60, 'f': 61, 'g': 62, 'h': 63, 'i': 64, 'j': 65, 'k': 66, 'l': 67, 'm': 68, 'n': 69, 'o': 70, 'p': 71, 'q': 72, 'r': 73, 's': 74, 't': 75, 'u': 76, 'v': 77, 'w': 78, 'x': 79, 'y': 80, 'z': 81, '|': 82}
        self.metrics = Metrics()
        self.wandb = False
        self.device = torch.device(device)
        self.load_model(checkpoint)

    def dataloader(self, purpose, batch_size, num_workers, pin_memory):
        if purpose == 'train':
            indices = range(6482)
        elif purpose == 'val':
            inidices = range(6482, 6482 + 976)
        else:
            inidices = range(6482 + 976, 6482 + 976 + 2914)
        dl = iam_dataloader(self.iam_dataset, batch_size, num_workers, pin_memory, indices)
        return dl

    def iam_dataset_init(self, annotation_txt, image_folder):
        if not hasattr(self, 'iam_dataset'):
            self.iam_dataset = IAM(annotation_txt, image_folder, self.alphabet)

    def metrics(self, hypo, ref):
        # Corpus-Level WER: 40.0
        wer, cer = self.metrics.wer(hypo, ref), self.metrics.cer(hypo, ref)
        return wer, cer

    def train_(self):
        self.vfe.train()
        self.tt.train()

    def eval_(self):
        self.vfe.eval()
        self.tt.eval()

    def adjust_learning_rate(self, lr, lr_decay):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.5 ** (epoch // lr_decay))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, checkpoint_dir, annotation_txt, image_folder, epochs, lr, lr_decay, batch_size, num_workers, pin_memory, smoothing_eps):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.iam_dataset_init(annotation_txt, image_folder)
        criterion = LabelSmoothingLoss(smoothing_eps, len(self.alphabet))
        train_loader = self.dataloader('train', batch_size, num_workers, pin_memory)
        val_loader = self.dataloader('val', 1, num_workers, False)
        if self.wandb:
            wandb.watch(self.vfe)
            wandb.watch(self.tt)
        for i in trange(self.epochs, epochs):
            self.adjust_learning_rate(lr, lr_decay)
            hypo, ref = [], []
            total_loss = 0
            dim1 = 0
            self.train_()
            for img, txt in train_loader:
                dim1 += img.size()[0]
                optimizer.zero_grad()
                a, bt = self.vfe(img.to(self.device)), txt.permute(1, 0).to(self.device)
                b = self.tt(bt[0:89], a)
                loss = criterion(b, bt[1:])
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                hypo += self.tt.to_text(torch.argmax(b, dim=2))
                ref += self.tt.to_text(bt)
                dataset = IAM(annotation_txt, image_folder)
            twer, tcer = self.metrics(hypo, ref)
            mean_loss_train = total_loss/dim1
            
            self.eval_()
            hypo, ref = [], []
            for img, txt in val_loader:
                hypo += self.tt.gen(self.vfe(img))
                ref += self.tt.to_text(txt)
            vwer, vcer = self.metrics(hypo, ref)
            self.checkpoint({'train_wer': twer, 'train_cer': tcer, 'val_wer': vwer, 'val_cer': vcer, 'train_loss': mean_loss_train}, i)

    def test(self, annotation_txt, image_folder):
        test_loader = self.dataloader('test', 1, num_workers, False)
        self.eval_()
        hypo, ref = [], []
        for img, txt in val_loader:
            hypo += self.tt.gen(self.vfe(img))
            ref += self.tt.to_text(txt)
        wer, cer = self.metrics(hypo, ref)

    def log(self, metrics, step):
        print(metrics)
        if self.wandb:
            wandb.log(metrics, step=step)

    def load_model(self, checkpoint):
        vfe = VisualFeatureEncoder()
        tt = TextTranscriber(self.alphabet)
        optimizer = optim.Adam(list(vfe.parameters()) + list(tt.parameters()), lr=0.001)
        if checkpoint is not None:
            a, b, c, self.metrics_ = torch.load(checkpoint)
            vfe.load_state_dict(a)
            tt.load_state_dict(b)
            optimizer.load_state_dict(c)
        else:
            self.metrics_ = {}
        self.vfe, self.tt, self.optim = vfe.to(self.device), tt.to(self.device), optimizer

    def checkpoint(self, metric, step):
        self.log(metrics, step)
        if self.metrics_.get('val_cer', 10000) > metrics['val_cer']:
            self.metrics_ = metrics
            torch.save((self.vfe, self.tt, self.optimizer, metrics, step), os.path.join(self.output_folder, 'best_model.pkl'))
        torch.save((self.vfe, self.tt, self.optimizer, metrics, step), os.path.join(self.output_folder, 'model.pkl'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog= "main.py", description = 'Train/Test/Gen for PATWYR', epilog='Type "main.py <command> -h" for more information.')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % ("main.py", example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('train', 'main.py', 'train -a ascii/lines.txt -i ')
    p.add_argument('-a', '--iam-annotation-txt', required=True, help='Annotation txt file')
    p.add_argument('-i', '--iam-image-folder', required=True, help='Image Folder')
    p.add_argument('--lr', default=0.0002, help='Directory containing dataset')
    p.add_argument('--lr_decay', default=20, type=int, help='Directory containing dataset')
    p.add_argument('--epochs', default=60, type=int, help='Directory containing dataset')
    p.add_argument('-d', '--device', default='cuda', help='Directory containing dataset')
    p.add_argument('-wp', '--wandb-project', type=str, default=None, help='Wandb-ID')
    p.add_argument('-we', '--wandb-entity', type=str, default=None, help='Wandb-ID')
    p.add_argument('-r', '--resume-checkpoint', default=None, help='Directory containing dataset')
    p.add_argument('-w', '--num_workers', type=int, default=4)
    p.add_argument('-se', '--smoothing_eps', type=float, default=0.4)
    p.add_argument('-c', '--checkpoint-dir', required=True, help='Directory containing dataset')
    p.add_argument('-bs', '--batch-size', default=32, help='Directory containing dataset')
    p.add_argument('-pm', '--pin-memory', action='store_true', help='Directory containing dataset')
    
    p = add_command('test', 'main.py', 'test -a ascii/lines.txt -i ')
    p.add_argument('-a', '--annotation-txt', required=True, help='Annotation txt file')
    p.add_argument('-i', '--image-folder', required=True, help='Image Folder')
    p.add_argument('-d', '--device', default='cuda', help='Directory containing dataset')
    p.add_argument('-c', '--checkpoint-dir', default=None, help='Directory containing dataset')

    p = add_command('gen', 'main.py', 'gen -a ascii/lines.txt -i ')
    p.add_argument('-i', '--image-dir', required=True, help='Image Folder')
    p.add_argument('-d', '--device', default='cuda', help='Directory containing dataset')
    p.add_argument('-c', '--checkpoint-pth', default=None, help='Directory containing dataset')
    p.add_argument('-p', '--predictions', default=None, help='Directory containing dataset')

    args = parser.parse_args()

    if args.command == 'train':
        log_wandb = False
        if args.wandb_project is not None:
            assert args.wandb_entity is not None 
            conf = vars(args)
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=conf)
            del conf['wandb_project']
            del conf['wandb_entity']
            log_wandb = True
        model = PATWYR(checkpoint=args.resume_checkpoint, device=args.device, wandb=log_wandb)
        model.train(args.checkpoint_dir, args.iam_annotation_txt, args.iam_image_folder, args.epochs, args.lr, args.lr_decay, args.batch_size, args.num_workers, bool(args.pin_memory), args.smoothing_eps)

    elif args.command == 'test':
        model = PATWYR(checkpoint=os.path.join(args.checkpoint_dir, 'best_model'), device=args.device)
        model.test(args.iam_annotation_txt, args.iam_image_folder)

    elif args.command == 'gen':
        model = PATWYR(checkpoint=args.resume_checkpoint, device=args.device)
