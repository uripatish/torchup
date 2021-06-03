import torch
import math

from tqdm import tqdm

from torchup.windowlr import WindowLR

import torchup.utils

class Predictor:

    def __init__(self, net, device):
        self.device = device      
        def todevice(x):
            return x.to(self.device)
        self.todevice = todevice

        self.net = net.to(device)


    def train(self, data_loader, loss_fn, target_num,
            batch_loss_fn = torchup.utils.input_output_batch_loss,
            optim = torch.optim.Adam,
            sched = WindowLR,
            sched_loss = True, 
            epochs = 100,
            pre_batch = None,
            post_batch = None,
            post_epoch = None):

        optim = optim(self.net.parameters())
        sched = sched(optim)

        for epoch_num in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            count = 0
            for batch_num, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                inputs = map(self.todevice, batch[:-target_num])
                targets = map(self.todevice, batch[-target_num:])

                if pre_batch:
                  pre_batch()

                optim.zero_grad()
                loss = batch_loss_fn(self.net, loss_fn, *inputs, *targets)
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                count += 1

                if post_batch:
                    post_batch()

            epoch_loss /= count
            print('{}/{}, loss: {}'.format(epoch_num+1, epochs, epoch_loss))
            if sched_loss:
                sched.step(epoch_loss)
            else:
                sched.step()

            if post_epoch:
                post_epoch(loss = epoch_loss)

    def eval_loss(self, data_loader, loss_fn, target_num,
                batch_loss_fn = torchup.utils.input_output_batch_loss):

        self.net.eval()
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():

            for batch_num, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                inputs = map(self.todevice, batch[:-target_num])
                targets = map(self.todevice, batch[-target_num:])

                loss = batch_loss_fn(self.net, loss_fn, *inputs, *targets)
                # outputs = self.net(*inputs)
                # loss = loss_fn(outputs, *targets)

                epoch_loss += loss.item()
                count += 1
        
        return (epoch_loss/count)


    def apply(self, data_loader, target_num, apply_fn = torch.nn.Identity()):      

        self.net.eval()
        outputs = []
        with torch.no_grad():

            for batch_num, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # batch_num, batch = next(iter(enumerate(data_loader)))
                inputs = map(self.todevice, batch[:-target_num])
                targets = map(self.todevice, batch[-target_num:])

                new_output = apply_fn(self.net(*inputs))

                outputs.append(new_output)
        
        return outputs
