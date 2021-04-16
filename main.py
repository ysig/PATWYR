import math
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from tqdm.auto import tqdm, trange
from metrics import ocr_metrics
from architecture import TransformerHTR, load_batch_image
from dataset import IAM, make_dataloader, ALPHABET, MAX_LEN
import torchvision.transforms.functional as FTV
import wandb
import os
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps, len_A, dim=-1):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = float(1 - ((len_A - 1)*eps/len_A))
        self.smoothing = float(eps/len_A)
        assert 0 <= self.smoothing < 1
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.full_like(pred, self.smoothing)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Engine(object):
    def __init__(self, checkpoint=None, lr=0.0002, device="cpu", wandb=False):
        self.alphabet = ALPHABET
        # self.metrics_obj = Metrics()
        self.wandb = wandb
        self.device = torch.device(device)
        self.load_model(checkpoint)

    def dataloader(self, dataset_type, purpose, batch_size, num_workers, pin_memory):
        if dataset_type == "IAM":
            if purpose == 'train':
                indices = range(6482)
            elif purpose == 'val':
                indices = range(6482, 6482 + 976)
            else:
                indices = range(6482 + 976, 6482 + 976 + 2914)
            return make_dataloader(self.iam_dataset, batch_size, num_workers, pin_memory, indices)
        elif dataset_type == "Synthetic":
            synthetic_dataset = self.synthetic_dataset
            if purpose == 'train':
                indices = range(int(0.8*len(synthetic_dataset)))
            else:
                indices = range(int(0.8*len(synthetic_dataset)), len(synthetic_dataset))
            return make_dataloader(synthetic_dataset, batch_size, num_workers, pin_memory, indices)
        else:
            return ValueError("Unrecognized Dataset")

    def dataset_init(self, dataset):
        if dataset[0] == "IAM":
            annotation_txt, image_folder = dataset[1]
            if not hasattr(self, 'iam_dataset'):
                self.iam_dataset = IAM(annotation_txt, image_folder, self.alphabet)
        
        elif dataset[0] == "Synthetic":
            if not hasattr(self, 'synthetic_dataset'):
                self.synthetic_dataset = Sythetic(dataset[1], self.alphabet)
        else:
            raise ValueError("Unrecognized Dataset")

    def metrics(self, hypo, ref):
        # wer, cer = self.metrics_obj.wer(hypo, ref), self.metrics_obj.cer(hypo, ref)
        return ocr_metrics(hypo, ref)

    def adjust_learning_rate(self, epoch, lr, lr_decay, decay_factor=0.5):
        if lr_decay is not None:
            lr = lr * (decay_factor ** (epoch // lr_decay))
        # print('epoch:',epoch,'lr:',lr)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def loss(self, criterion, b, trgt):
        loss = 0
        for j in range(trgt.size()[0]):
            loss += criterion(b[j], trgt[j])
        return loss, trgt.size()[0]

    def train(self, checkpoint_dir, dataset, epochs=60, lr=0.0001, lr_decay=None, batch_size=32, num_workers=1, pin_memory=False, label_smoothing=True, smoothing_eps=0.4, verbose=False, save_optimizer=False, no_save=False, log_after=10):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.dataset_init(dataset)
        train_loader = self.dataloader(dataset[0], 'train', batch_size, num_workers, pin_memory)
        val_loader = self.dataloader(dataset[0], 'val', batch_size, num_workers, False)

        NA = len(self.alphabet)
        if label_smoothing:
            criterion = LabelSmoothingLoss(smoothing_eps, NA)
        else:
            criterion = nn.CrossEntropyLoss()

        if self.wandb:
            wandb.watch(self.model)

        for i in trange(self.epochs, epochs):
            self.adjust_learning_rate(i, lr, lr_decay)
    
            hypo, ref = [], []
            total_loss, dim1 = 0, 0
    
            self.model.train()
    
            if verbose:
                iterator = tqdm(train_loader, total=len(train_loader), desc='Training')
            else:
                iterator = train_loader
    
            for img, txt in iterator:
                self.optim.zero_grad()
                bt = txt.squeeze(1).permute(1, 0).to(self.device)
                b = self.model(bt[0:MAX_LEN], img.to(self.device))
                trgt = bt[1:].permute(1, 0)
                loss, bs = self.loss(criterion, b, trgt)
                (loss/bs).backward()
                self.optim.step()
                total_loss += loss.detach().item()
                dim1 += bs
                hypo += self.model.to_text(torch.argmax(b, dim=2))
                ref += self.model.to_text(trgt)

            twer, tcer = self.metrics(hypo, ref)
            mean_loss_train = total_loss/dim1

            self.model.eval()
    
            if i <= log_after:
                self.log({'WER-train': twer, 'CER-train': tcer, 'LOSS-train': mean_loss_train}, i, verbose)
                continue

            hypo, hypo_greedy, ref = [], [], []
            with torch.no_grad():
    
                if verbose:
                    iterator = tqdm(val_loader, total=len(val_loader), desc='Validation')
                else:
                    iterator = val_loader
    
                total_loss, dim1 = 0, 0
                for img, txt in iterator:
                    bt = txt.squeeze(1).permute(1, 0).to(self.device)
                    b = self.model(bt[0:MAX_LEN], img.to(self.device))
                    trgt = bt[1:].permute(1, 0)
                    hypo += self.model.to_text(torch.argmax(b, dim=2))
                    hypo_greedy += self.model.gen(img.to(self.device))
                    ref += self.model.to_text(trgt)
                    loss, bs = self.loss(criterion, b, trgt)
                    total_loss += loss.detach().item()
                    dim1 += bs

            mean_loss_val = total_loss/dim1
            vwer, vcer = self.metrics(hypo, ref)
            vwer_greedy, vcer_greedy = self.metrics(hypo_greedy, ref)
    
            metrics = {'WER-train': twer, 'CER-train': tcer, 'WER-val': vwer, 'CER-val': vcer, 'WER-val-greedy': vwer_greedy,
                       'CER-val-greedy': vcer_greedy, 'LOSS-train': mean_loss_train, 'LOSSval-loss': mean_loss_val}
            self.checkpoint(metrics, checkpoint_dir, i, save_optimizer, no_save, verbose)

    def test(self, annotation_txt, image_folder):
        test_loader = self.dataloader('test', 1, num_workers, False)
        self.eval_()
        hypo, ref = [], []
        for img, txt in val_loader:
            hypo += self.model.gen(img.to(self.device))
            ref += self.model.to_text(txt.squeeze(1))
        wer, cer = self.metrics(hypo, ref)

    def log(self, metrics, step, verbose):
        if verbose:
            print(f"Epoch {step}:\n", metrics)
        with torch.no_grad():
            img = load_batch_image(4).to(self.device)
            out = self.model.gen(img)
            imgs = []
            for i in range(img.size()[0]):
                imgs.append((FTV.to_pil_image(img[i]), str(out[i])))
        if verbose:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(imgs), 1)
            for ax, (img, t) in zip(axs, imgs):
                ax.imshow(img)
                ax.set_title(t)
                ax.axis('off')
            plt.subplots_adjust(hspace=0)
            plt.show()

        if self.wandb:
            images = {'images': [wandb.Image(x, caption=t) for x, t in imgs]}
            wandb.log(metrics, step=step)
            wandb.log(images, step=step)

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

    def checkpoint(self, metrics, checkpoint_dir, step, save_optimizer, no_save, verbose):
        self.log(metrics, step, verbose)
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

    p = add_command('pretrain', 'main.py', 'train -a ascii/lines.txt -i ')
    p.add_argument('--synthetic-data', required=True, help='Image Folder')
    p.add_argument('--lr', default=0.0002, type=float, help='Directory containing dataset')
    p.add_argument('--lr_decay', default=None, type=int, help='Directory containing dataset')
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
    p.add_argument('--verbose', action='store_true', help='Directory containing dataset')
    p.add_argument('--log-after', default=-1, help='Directory containing dataset')
    p.add_argument('--label-smoothing', action='store_true', help='Directory containing dataset')

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
    p.add_argument('--log-after', default=-1, help='Directory containing dataset')
    p.add_argument('--no-save', action='store_true', help='Directory containing dataset')
    p.add_argument('--verbose', action='store_true', help='Directory containing dataset')
    p.add_argument('--label-smoothing', action='store_true', help='Directory containing dataset')

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
    if args.command in {'train', 'pretrain'}:
        log_wandb = False
        if args.wandb_project is not None:
            assert args.wandb_entity is not None 
            conf = vars(args)
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config={"command": args.command, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "lr_decay": args.lr_decay, "smoothing_eps": args.smoothing_eps, 'label_smoothing': bool(args.label_smoothing)})
            del conf['wandb_project']
            del conf['wandb_entity']
            log_wandb = True
        engine = Engine(checkpoint=args.resume_checkpoint, device=args.device, wandb=log_wandb)
        if args.command == "train":
            engine.train(args.checkpoint_dir, ("IAM", (args.iam_annotation_txt, args.iam_image_folder)), args.epochs, args.lr, args.lr_decay, args.batch_size, args.num_workers, bool(args.pin_memory), bool(args.label_smoothing), args.smoothing_eps, verbose=bool(args.verbose), save_optimizer=bool(args.save_optimizer), no_save=bool(args.no_save), log_after=int(args.log_after))
        elif args.command == "pretrain":
            engine.train(args.checkpoint_dir, ("Synthetic", (args.pretrained)), args.epochs, args.lr, args.lr_decay, args.batch_size, args.num_workers, bool(args.pin_memory), bool(args.label_smoothing), args.smoothing_eps, verbose=bool(args.verbose), save_optimizer=bool(args.save_optimizer), no_save=False, log_after=int(args.log_after))

    elif args.command == 'test':
        engine = Engine(checkpoint=os.path.join(args.checkpoint_dir, 'best_model'), device=args.device)
        engine.test(args.iam_annotation_txt, args.iam_image_folder)

    elif args.command == 'gen':
        engine = Engine(checkpoint=args.resume_checkpoint, device=args.device)
