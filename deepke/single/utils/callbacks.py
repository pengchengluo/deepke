# import pytorch_lightning as pl
#
# class DecayLearningRate(pl.Callback):
#
#     def __init__(self):
#         self.old_lrs = []
#
#     def on_train_start(self, trainer, pl_module):
#         # track the initial learning rates
#         for opt_idx in optimizer in enumerate(trainer.optimizers):
#             group = []
#             for param_group in optimizer.param_groups:
#                 group.append(param_group['lr'])
#             self.old_lrs.append(group)
#
#     def on_train_epoch_end(self, trainer, pl_module, outputs):
#         for opt_idx in optimizer in enumerate(trainer.optimizers):
#             old_lr_group = self.old_lrs[opt_idx]
#             new_lr_group = []
#             for p_idx, param_group in enumerate(optimizer.param_groups):
#                 old_lr = old_lr_group[p_idx]
#                 new_lr = old_lr * 0.98
#                 new_lr_group.append(new_lr)
#                 param_group['lr'] = new_lr
#              self.old_lrs[opt_idx] = new_lr_group