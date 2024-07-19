from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class TrainModeControllerHook(Hook):
    """
    Args:
        num_epoch (List), train_mode (List)
    """
    def __init__(self, num_epoch, train_mode):
        self.train_mode = train_mode
        self.switch_point = num_epoch
        for i in range(len(num_epoch) - 1):
            self.switch_point[i + 1] += self.switch_point[i]
    
    def before_run(self, runner):
        if hasattr(runner.model, 'module'):
            self.model = runner.model.module
        else:
            self.model = runner.model

    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        idx = -1
        for i, point in enumerate(self.switch_point):
            if point >= epoch:
                idx = i
                break
        train_mode = self.train_mode[idx]

        assert train_mode in ['crux_learner', 'catch_up_learner', 'alter']
        if train_mode == 'alter':
            if epoch == 1:
                train_mode = self.model.train_mode
            else:
                train_mode = 'crux_learner' if self.model.train_mode == 'catch_up_learner' else 'catch_up_learner'
        self.model.switch_train_mode(train_mode)
        runner.logger.info(f'train mode: {self.model.train_mode}')

    # TODO
    # def after_train_iter(self, runner):
    #     """print grad"""
    #     from matplotlib import pyplot as plt
    #     import numpy as np
    #     names = []
    #     off_x, on_x, grads = [], [], []
    #     for i, (name, par) in enumerate(self.model.named_parameters()):
    #         names.append(name)
    #         if par.requires_grad:
    #             grads.append(par.grad.clone().detach().mean().abs().cpu().numpy())
    #             on_x.append(i)
    #             print(name, par.grad.detach().mean().item())
    #         else:
    #             off_x.append(i)
        
    #     grads = np.stack(grads).reshape(-1)
    #     plt.figure(1, figsize=(100, 40))
    #     plt.scatter(on_x, grads, c='red', s=50)
    #     plt.scatter(off_x, np.zeros(len(off_x)))
    #     plt.xticks(range(len(names)), names, rotation=90)
    #     plt.yticks(np.linspace(0, grads.max(), 20))
    #     plt.grid()
    #     plt.savefig('./grad_fig.jpg')
    #     print('save_fig')
    #     input()
        