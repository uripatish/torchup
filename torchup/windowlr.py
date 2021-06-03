import torch

from torch._six import inf

from torch.optim import Optimizer

class WindowLR(object):
    """Reduce learning rate when a metric has stopped improving. 
    Comparison is done with an moving window statistic that is 
    calculated over a fixed sized moving. 
    If the optimization gets stuck around some value of the objective,
    it can be effective to reduce the learning rate. For this purpose, 
    this scheduler reads a metrics quantity and if no improvement is 
    seen compared to a window statistic, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        win_size (int): Number of objective values for calculating 
            the window statistic. Default: 10.
        win_fn (function): Function that recieves a window tensor, and 
            returns a scalar statistic. Default: mean. 
        threshold (float): Threshold for measuring a change in the window            
            statistic. Default: 0.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = win_stat * ( 1 + threshold ) in 'max'
            mode or win_stat * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = win_stat + threshold in
            `max` mode or win_stat - threshold in `min` mode. Default: 'rel'.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``True``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = WindowLR(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, 
                 win_size=10, win_fn = torch.mean, reset_after_reduce=True,
                 threshold=0., threshold_mode='rel', 
                 min_lr=1e-5, eps=1e-8, verbose=True):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.win_size = win_size
        self.win_fn = win_fn
        self.reset_after_reduce = reset_after_reduce

        self.verbose = verbose
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.win_stat = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets window."""
        self.window = []        
        self.win_stat = self.mode_worse

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        # add current value and calculate new average if window has enough values        
        self.window.append(current)
        # remove last item if needed
        if len(self.window) > self.win_size:
            self.window.pop(0)
        # update win_stat
        if len(self.window) == self.win_size:
            self.win_stat = self.win_fn(torch.Tensor(self.window)).item()

        # perform comparisons    
        if not self.is_better(current, self.win_stat):
            self._reduce_lr(epoch)
            if self.reset_after_reduce:
                self._reset()

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def is_better(self, a, win_stat):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < win_stat * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < win_stat - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > win_stat * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > win_stat + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
