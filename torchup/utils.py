import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

def counts(V):
  return [(v.item(), (V == v).sum().item()) for v in V.unique()]

def paramcount(net):
  sum(p.numel() for p in net.parameters())

def err_loss(outputs, targets):
  return (outputs.argmax(1) != targets).float().mean()

def sign_err_loss(outputs, targets):
  return (outputs.sign() != targets.unsqueeze(1)).float().mean()

def logistic_loss(outputs, targets):
  return (-(outputs * targets.unsqueeze(1)).sigmoid().log()).mean()

def logloss(class_prob, labels, eps = 1e-8):
  # loss = -(class_prob * functional.one_hot(labels, num_classes = class_dim)).sum(1).clamp(eps, 1-eps).log().mean()
  # loss = -torch.gather(class_prob, 1, labels.unsqueeze(1)).clamp(eps, 1-eps).log().mean()
  loss = -torch.gather(class_prob, 1, labels.unsqueeze(1)).log().mean()
  return loss

# Estimate channels' mean and standard deviation of an image dataset.
# To estimate the standard deviation, using the identity: Var[X] = E[X**2] - E[X]**2
# Since expectation is linear, the estimation over the dataset is done by averaging averages.
def channel_stats(dataset, device, *args, **kwargs):
    
    # build dataloader
    kwargs['drop_last'] = True # drop the last batch if it is not full, so as not to skew the statistics 
    dataloader = DataLoader(dataset, *args, **kwargs)
    # init values
    channels_sum, channels_sqrd_sum, batch_num = 0, 0, 0

    # iterate only over the images, and ignore the labels
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        batch_num += 1

    # finalize
    mean = channels_sum / batch_num
    std = ((channels_sqrd_sum / batch_num) - (mean ** 2)) ** 0.5

    mean = mean.tolist()
    std = std.tolist()

    return mean, std

# elements to be used in predictor training
def input_output_batch_loss(net, loss_fn, inputs, targets):
    outputs = net(inputs)
    loss = loss_fn(outputs, targets)

    return loss
