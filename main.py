import math
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from tqdm.auto import tqdm, trange
from metrics import Metrics
from architecture import TransformerHTR, load_batch_image
from dataset import IAM, iam_dataloader, ALPHABET, MAX_LEN
import torchvision.transforms.functional as FTV
import wandb
import os
import matplotlib.pyplot as plt

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
        return self.smoothing_a * x + self.smoothing_b * y

    def forward(self, preds, target):
        # assert 0 <= self.smoothing < 1
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)
        return self.linear_combination(loss / n, nll)

class Trainer(object):
    def __init__(self, checkpoint=None, lr=0.0002, device="cpu", wandb=False):
        self.alphabet = ALPHABET
        self.metrics_obj = Metrics()
        self.wandb = wandb
        self.device = torch.device(device)
        self.load_model(checkpoint)

    def dataloader(self, purpose, batch_size, num_workers, pin_memory):
        if purpose == 'train':
            indices = range(6482)
        elif purpose == 'val':
            indices = range(6482, 6482 + 976)
        else:
            indices = range(6482 + 976, 6482 + 976 + 2914)
        return iam_dataloader(self.iam_dataset, batch_size, num_workers, pin_memory, indices)

    def iam_dataset_init(self, annotation_txt, image_folder):
        if not hasattr(self, 'iam_dataset'):
            self.iam_dataset = IAM(annotation_txt, image_folder, self.alphabet)

    def metrics(self, hypo, ref):
        # Corpus-Level WER: 40.0
        wer, cer = self.metrics_obj.wer(hypo, ref), self.metrics_obj.cer(hypo, ref)
        return wer, cer

    def adjust_learning_rate(self, epoch, lr, lr_decay):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.5 ** (epoch // lr_decay))
        # print('epoch:',epoch,'lr:',lr)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def train(self, checkpoint_dir, annotation_txt, image_folder, epochs, lr, lr_decay, batch_size, num_workers, pin_memory, smoothing_eps, save_optimizer=False, no_save=False):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.iam_dataset_init(annotation_txt, image_folder)
        NA = len(self.alphabet)
        # criterion = LabelSmoothingLoss(smoothing_eps, NA)
        criterion = nn.CrossEntropyLoss()
        train_loader = self.dataloader('train', batch_size, num_workers, pin_memory)
        val_loader = self.dataloader('val', batch_size, num_workers, False)
        if self.wandb:
            wandb.watch(self.model)
        for i in trange(self.epochs, epochs):
            self.adjust_learning_rate(i, lr, lr_decay)
            hypo, ref = [], []
            total_loss, dim1 = 0, 0
            self.model.train()
            for img, txt in tqdm(train_loader, total=len(train_loader), desc='Training'):
                self.optim.zero_grad()
                bt = txt.squeeze(1).permute(1, 0).to(self.device)
                b = self.model(bt[0:MAX_LEN], img.to(self.device))
                trgt = bt[1:].permute(1, 0)
                loss = 0
                for i in range(trgt.size()[0]):
                    loss += criterion(b[i], trgt[i])
                (loss/trgt.size()[0]).backward()
                self.optim.step()
                total_loss += loss.detach().item()
                dim1 += trgt.size()[0]
                hypo += self.model.to_text(torch.argmax(b, dim=2))
                ref += self.model.to_text(trgt)
            twer, tcer = self.metrics(hypo, ref)
            mean_loss_train = total_loss/dim1
            
            self.model.eval()
            hypo, ref = [], []
            with torch.no_grad():
                for img, txt in tqdm(val_loader, total=len(val_loader), desc='Validation'):
                    bt = txt.squeeze(1).permute(1, 0).to(self.device)
                    b = self.model(bt[0:MAX_LEN], img.to(self.device))
                    trgt = bt[1:].permute(1, 0)
                    hypo += self.model.to_text(torch.argmax(b, dim=2))
                    ref += self.model.to_text(trgt)
            vwer, vcer = self.metrics(hypo, ref)
            self.checkpoint({'train_wer': twer, 'train_cer': tcer, 'val_wer': vwer, 'val_cer': vcer, 'train_loss': mean_loss_train}, checkpoint_dir, i, save_optimizer, no_save)

    def test(self, annotation_txt, image_folder):
        test_loader = self.dataloader('test', 1, num_workers, False)
        self.eval_()
        hypo, ref = [], []
        for img, txt in val_loader:
            hypo += self.model.gen(img.to(self.device))
            ref += self.model.to_text(txt.squeeze(1))
        wer, cer = self.metrics(hypo, ref)

    def log(self, metrics, step):
        print(metrics)
        with torch.no_grad():
            img = load_batch_image().to(self.device)
            out = self.model.gen(img)
            imgs = []
            for i in range(img.size()[0]):
                imgs.append((FTV.to_pil_image(img[i]), str(out[i])))
        fig, axs = plt.subplots(len(imgs), 1)
        for ax, (img, t) in zip(axs, imgs):
            ax.imshow(img)
            ax.set_title(t)
            # print(t)
            ax.axis('off')
        fig.tight_layout()
        plt.show()

        if self.wandb:
            # print('Wandb logging')
            # print(metrics)
            images = {'images': [wandb.Image(x, caption=t) for x, t in imgs]}
            wandb.log(metrics, step=step)
            # wandb.log(images, step=step)

    def load_model(self, checkpoint):
        model = TransformerHTR(self.alphabet, text_len=MAX_LEN)
        optimizer = optim.Adam(list(model.parameters()), lr=0.001)
        if checkpoint is not None:
            d = torch.load(checkpoint)
            self.metrics_ = d['metrics']
            self.epochs = d['epochs']
            model.load_state_dict(d['model'])
            if 'optimizer' in d:
                optimizer.load_state_dict(d['optimizer'])
        else:
            self.metrics_, self.epochs = {}, 0
        self.model, self.optim = model.to(self.device), optimizer

    def save(self, epochs, metrics, save_optimizer, save_pkl):
        d = {'model': self.model, 'metrics': metrics, 'epochs': epochs}
        if save_optimizer:
            d['optimizer'] = self.optim
        torch.save(d, save_pkl)

    def checkpoint(self, metrics, checkpoint_dir, step, save_optimizer, no_save):
        self.log(metrics, step)
        if self.metrics_.get('val_cer', 10000) > metrics['val_cer']:
            self.metrics_ = metrics
            if not no_save:
                self.save(step, metrics, save_optimizer, os.path.join(checkpoint_dir, 'best_model.pkl'))
        if not no_save:
            self.save(step, metrics, save_optimizer, os.path.join(checkpoint_dir, 'model.pkl'))

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
    p.add_argument('--lr', default=0.0002, type=float, help='Directory containing dataset')
    p.add_argument('--lr_decay', default=20, type=int, help='Directory containing dataset')
    p.add_argument('--epochs', default=60, type=int, help='Directory containing dataset')
    p.add_argument('-d', '--device', default='cuda', help='Directory containing dataset')
    p.add_argument('-wp', '--wandb-project', type=str, default=None, help='Wandb-ID')
    p.add_argument('-we', '--wandb-entity', type=str, default=None, help='Wandb-ID')
    p.add_argument('-r', '--resume-checkpoint', default=None, help='Directory containing dataset')
    p.add_argument('-w', '--num_workers', type=int, default=4)
    p.add_argument('-se', '--smoothing_eps', type=float, default=0.4)
    p.add_argument('-c', '--checkpoint-dir', required=True, help='Directory containing dataset')
    p.add_argument('-bs', '--batch-size', default=32, type=int, help='Directory containing dataset')
    p.add_argument('-pm', '--pin-memory', action='store_true', help='Directory containing dataset')
    p.add_argument('--save-optimizer', action='store_true', help='Directory containing dataset')
    p.add_argument('--no-save', action='store_true', help='Directory containing dataset')

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
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "lr_decay": args.lr_decay, "smoothing_eps": args.smoothing_eps})
            del conf['wandb_project']
            del conf['wandb_entity']
            log_wandb = True
        model = Trainer(checkpoint=args.resume_checkpoint, device=args.device, wandb=log_wandb)
        model.train(args.checkpoint_dir, args.iam_annotation_txt, args.iam_image_folder, args.epochs, args.lr, args.lr_decay, args.batch_size, args.num_workers, bool(args.pin_memory), args.smoothing_eps, save_optimizer=bool(args.save_optimizer), no_save=bool(args.no_save))

    elif args.command == 'test':
        model = Trainer(checkpoint=os.path.join(args.checkpoint_dir, 'best_model'), device=args.device)
        model.test(args.iam_annotation_txt, args.iam_image_folder)

    elif args.command == 'gen':
        model = Trainer(checkpoint=args.resume_checkpoint, device=args.device)
