import torch
import math

class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, warmup_iters=1000, base_lr=1e-4, plateau_scheduler_kwargs=None):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.base_lr = base_lr
        self.current_step = 0
        self.in_warmup = True
        self.init_lrs = [group['lr'] for group in optimizer.param_groups]
        
        if plateau_scheduler_kwargs is None:
            plateau_scheduler_kwargs = dict(mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_scheduler_kwargs)

    def step(self, metrics=None):
        if self.current_step < self.warmup_iters:
            self.in_warmup = True
            lr_scale = (self.current_step + 1) / self.warmup_iters
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lr * lr_scale
            self.current_step += 1
        else:
            if self.in_warmup:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr
                self.in_warmup = False

            self.plateau_scheduler.step(metrics)


def make_optimizer(cfg, model):
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    cfg.SOLVER.BASE_LR,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'ADAM' or 'ADAMcos':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    elif cfg.SOLVER.OPTIMIZER == 'ADAMW':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                      amsgrad=cfg.SOLVER.AMSGRAD)
    else:
        raise NotImplementedError()
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.OPTIMIZER == 'ADAMcos':
        t = cfg.SOLVER.STATIC_STEP
        max_ep = cfg.SOLVER.MAX_EPOCH
        lambda1 = lambda epoch: 1 if epoch < t else 0.00001 \
            if 0.5 * (1 + math.cos(math.pi * (epoch - t) / (max_ep - t))) < 0.00001 \
            else 0.5 * (1 + math.cos(math.pi * (epoch - t) / (max_ep - t)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif cfg.SOLVER.LR_SCHEDULER == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg.SOLVER.LR_PATIENCE,
            factor=cfg.SOLVER.LR_FACTOR,
            threshold=cfg.SOLVER.LR_THRESHOLD,
            verbose=True)
    elif cfg.SOLVER.LR_SCHEDULER == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            base_lr=cfg.SOLVER.BASE_LR,
            plateau_scheduler_kwargs={
                'mode': 'min',
                'factor': cfg.SOLVER.LR_FACTOR,
                'patience': cfg.SOLVER.LR_PATIENCE,
                'min_lr': cfg.SOLVER.LR_MIN,
                'verbose': True
            }
        )
    else:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)
    

