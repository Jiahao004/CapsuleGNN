import os
import torch

import logging

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from model import Model


class Trainer(object):
    def __init__(self, optimizer, patience=False, device='cuda'):
        super(Trainer, self).__init__()

        self.optimizer = optimizer
        self.device = device
        self.patience = patience

    def train(self, model, n_epochs, trainloader, testloader, output_path):
        self.max_f1 = 0
        patience_count = 0
        for epoch in range(n_epochs):
            loss = self.train_epoch(model, trainloader)
            a, f, ff = self.eval(model, testloader)
            f1 = (f + ff) / 2
            if f1 > self.max_f1:
                patience_count = 0
                self.max_f1 = f1
                self.f = f
                self.ff = ff
                self.a = a
                self.save_model(model, output_path)
            else:
                if self.patience:
                    if self.patience == patience_count:
                        print('early stopping...')
                        break
                    else:
                        patience_count += 1

            logging.info('epoch:{}    loss:{:.4f}    acc:{:.4f}    macro-f1:{:.4f}    micro-f1:{:.4f}'.format(epoch, loss, a, f, ff))
            print('epoch:{}    loss:{:.4f}    acc:{:.4f}    macro-f1:{:.4f}    micro-f1:{:.4f}'.format(epoch, loss, a, f, ff))

        logging.info('dev max f1:{:.4f}'.format(self.max_f1))
        print('dev max f1:{:.4f}'.format(self.max_f1))
        return self.a, self.f, self.ff

    def train_epoch(self, model, trainloader):
        n_steps, epoch_loss = 0, 0
        for batch in tqdm(trainloader, desc='training'):
            epoch_loss += self.train_step(model, batch)
            n_steps += 1
        return epoch_loss / n_steps

    def train_step(self, model, batch):
        assert isinstance(model, Model)
        x, edge_index, batch, y = batch.x, batch.edge_index, batch.batch, batch.y
        loss = model(x.to(self.device), edge_index.to(self.device), batch.to(self.device), y.to(self.device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def eval(self, model, testloader):
        model.eval()
        with torch.no_grad():
            tbar = tqdm(testloader, desc='eval')
            outputs, targets = [], []
            for b in tbar:
                x, edge_index, batch, y = b.x, b.edge_index, b.batch, b.y
                cls_caps_output = model.infer(x.to(self.device), edge_index.to(self.device), batch.to(self.device))
                assert isinstance(cls_caps_output, torch.Tensor)
                output = torch.norm(cls_caps_output, p=2, dim=-1).argmax(dim=-1)
                outputs += output.cpu().tolist()
                targets += y.tolist()
        model.train()
        f = f1_score(targets, outputs, average='macro')
        a = accuracy_score(targets, outputs)
        ff = f1_score(targets, outputs, average='micro')
        return a, f, ff

    def save_model(self, model, output_path):
        cp_path = os.path.join(output_path, 'checkpoint.pt')
        print(f'saving to {cp_path}')
        logging.info(f'saving to {cp_path}')
        cp = {'model': model}
        torch.save(cp, cp_path)

    def load_model(self, output_path):
        cp_path = os.path.join(output_path, 'checkpoint.pt')
        print(f'loading from {cp_path}')
        logging.info(f'loading from {cp_path}')
        cp = torch.load(cp_path)
        return cp['model']
