import torch

from tqdm import tqdm

import torchup.utils

def sched_3steps(optim, epochs: int, **kwargs):    
  return torch.optim.lr_scheduler.StepLR(optim, step_size = epochs//3)

def fit(params, data_loader, batch_loss_fn, device,
        epochs = 100,
        optim_fn = torch.optim.Adam,
        sched_fn = sched_3steps,
        sched_batch = False,

        # hooks
        pre_batch = None,
        post_batch = None,
        pre_epoch = None,
        post_epoch = None):

    optim = optim_fn(params)
    sched = sched_fn(optim, epochs = epochs)

    def todevice(x):
        return x.to(device)

    for epoch_num in range(epochs):
        
        if pre_epoch:
            pre_epoch()

        epoch_loss = 0.0
        count = 0
        for batch_num, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
            if pre_batch:
               pre_batch()

            # batch = next(tr_itr) # for debug
            optim.zero_grad()
            loss = batch_loss_fn(*map(todevice, batch))
            assert (not loss.isnan().item()) and (not loss.isinf().item())
            loss.backward()
            optim.step()

            if sched_batch:
                sched.step()

            epoch_loss += loss.item()
            count += 1

            if post_batch:
                post_batch()

        epoch_loss /= count
        print('{}/{}, loss: {}'.format(epoch_num+1, epochs, epoch_loss))

        if (not sched_batch) and (epochs > 1):
            sched.step()

        if post_epoch:
            post_epoch(loss = epoch_loss)
