import nntools.tracker.metrics as NNmetrics
import torch
import torch.distributed as dist
import tqdm
from nntools.dataset import class_weighting
from nntools.nnet import FuseLoss, SUPPORTED_LOSS, BINARY_MODE, MULTICLASS_MODE
from nntools.utils import reduce_tensor
from nntools.utils.misc import call_with_filtered_kwargs
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from .experiment import Experiment


class SupervisedExperiment(Experiment):
    def __init__(self, config, run_id=None):
        super(SupervisedExperiment, self).__init__(config, run_id=run_id)
        if 'ignore_index' in self.config['Training']:
            self.ignore_index = self.config['Training']['ignore_index']
        else:
            self.ignore_index = -100

        self.n_classes = config['Network'].get('n_classes', -1)
        self.class_weights = None
        self.gt_name = 'mask'

    def initial_tracking(self):
        if 'Optimizer' in self.config:
            self.log_params(**self.config['Optimizer'])
        if 'Learning_rate_scheduler' in self.config:
            self.log_params(**self.config['Learning_rate_scheduler'])

        self.log_params(**self.config['Network'])
        self.log_params(Loss=self.config['Loss'].get('type', 'custom'))
        if 'fusion' in self.config['Loss']:
            self.log_params(Loss_fusion=self.config['Loss']['fusion'])
        if 'params_loss' in self.config['Loss']:
            self.log_params(**self.config['Loss']['params_loss'])
        if 'weighted_loss' in self.config['Loss']:
            self.log_params(weighted_loss=self.config['Loss']['weighted_loss'])
        if 'params_weighting' in self.config['Loss'] and self.config['Loss'].get('weighted_loss', False):
            self.log_params(**self.config['Loss']['params_weighting'])
        if 'Preprocessing' in self.config:
            self.log_params(**self.config['Preprocessing'])

        super(SupervisedExperiment, self).initial_tracking()

    def start(self, run_id=None):
        if self.config['Loss'].get('weighted_loss', False) and self.class_weights is None:
            class_weights = self.get_class_weights()
            self.setup_class_weights(weights=class_weights)
        super(SupervisedExperiment, self).start(run_id)

    def get_loss(self, weights=None, rank=0):
        config = self.config['Loss']
        mode = MULTICLASS_MODE if self.n_classes > 2 else BINARY_MODE
        fuse_loss = FuseLoss(fusion=config.get('fusion', 'mean'), mode=mode)

        list_losses = config['type'].split('|')

        if weights is not None:
            weights = weights.cuda(self.get_gpu_from_rank(rank))
        kwargs = {'ignore_index': self.ignore_index, 'mode': mode}

        for k in list_losses:
            k = k.strip()
            loss = SUPPORTED_LOSS[k]
            loss_args = kwargs.copy()
            loss_args['weight'] = weights
            if k in config.get('params_loss', {}):
                loss_args.update(config['params_loss'][k])

            fuse_loss.add(call_with_filtered_kwargs(loss, loss_args))

        return fuse_loss

    def get_class_weights(self):
        class_count = self.train_dataset.get_class_count()
        kwargs = self.config['Loss'].get('params_weighting', {})
        return torch.tensor(class_weighting(class_count, ignore_index=self.ignore_index, **kwargs))

    def setup_class_weights(self, weights):
        if self.config['Manager']['amp']:
            self.class_weights = weights.half()
        else:
            self.class_weights = weights

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.class_weights, rank=rank)
        optimizer = self.partial_optimizer(
            model.get_trainable_parameters(self.config['Optimizer']['params_solver']['lr']))

        if self.partial_lr_scheduler is not None:
            lr_scheduler = self.partial_lr_scheduler(optimizer)
        else:
            lr_scheduler = None
        iteration = self.tracker.current_iteration - 1
        train_loader, train_sampler = self.get_dataloader(self.train_dataset, rank=rank)

        if self.validation_dataset is not None:
            valid_loader, valid_sampler = self.get_dataloader(self.validation_dataset,
                                                              batch_size=1,
                                                              shuffle=False, rank=rank)
            self.ctx_train['valid_loader'] = valid_loader
            self.ctx_train['valid_sampler'] = valid_sampler

        scaler = GradScaler(enabled=self.config['Manager'].get('grad_scaling', True))

        self.ctx_train['loss_function'] = loss_function
        self.ctx_train['lr_scheduler'] = lr_scheduler
        self.ctx_train['iteration'] = iteration

        self.ctx_train['scaler'] = scaler

        self.ctx_train['train_loader'] = train_loader
        self.ctx_train['train_sampler'] = train_sampler

        self.ctx_train['optimizer'] = optimizer
        self.ctx_train['model'] = model

        self.epoch_loop(rank=rank)

    def in_epoch(self, epoch, rank=0):
        model = self.ctx_train['model']
        optimizer = self.ctx_train['optimizer']
        clip_grad = self.config['Training'].get('grad_clipping', False)
        scaler = self.ctx_train['scaler']
        lr_scheduler = self.ctx_train['lr_scheduler']
        train_loader = self.ctx_train['train_loader']
        iters_to_accumulate = self.config['Training'].get('iters_to_accumulate', 1)

        if self.is_main_process(rank):
            progressBar = tqdm.tqdm(total=len(train_loader))

        if self.validation_dataset is not None:
            valid_loader = self.ctx_train['valid_loader']
            valid_sampler = self.ctx_train['valid_sampler']

        if self.ctx_train['train_sampler'] is not None:
            self.ctx_train['train_sampler'].set_epoch(epoch)

        for i, batch in (enumerate(train_loader)):
            self.ctx_train['iteration'] += 1
            with autocast(enabled=self.config['Manager']['amp']):
                loss = self.forward_train(self.ctx_train['model'], self.ctx_train['loss_function'], rank, batch)

                loss = loss / iters_to_accumulate
                self.ctx_train['scaler'].scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if clip_grad:
                        clip_grad_norm_(model.parameters(), float(clip_grad))
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()

                    if self.ctx_train['scheduler_opt'].call_on == 'on_iteration':
                        self.lr_scheduler_step(lr_scheduler, epoch, i, len(train_loader))

            """
            Validation step
            """
            if self.ctx_train['iteration'] % self.config['Validation']['log_interval'] == 0:
                if self.validation_dataset is not None:
                    with torch.no_grad():
                        with autocast(enabled=self.config['Manager']['amp']):
                            valid_metric = self.validate(model, valid_loader,
                                                         self.ctx_train['iteration'],
                                                         rank,
                                                         self.ctx_train['loss_function'])

                if self.ctx_train['scheduler_opt'].call_on == 'on_validation':
                    self.lr_scheduler_step(lr_scheduler, epoch, i, len(train_loader), valid_metric)

                if self.is_main_process(rank):
                    self.log_metrics(self.ctx_train['iteration'], trainining_loss=loss.detach().item())
                    self.save_model(model, filename='last')

            if self.multi_gpu:
                dist.barrier()

            if self.is_main_process(rank):
                progressBar.update(1)

        if self.is_main_process(rank):
            progressBar.close()

        """ 
        If the validation set is not provided, we save the model once per epoch
        """
        if self.validation_dataset is None:
            if self.is_main_process(rank):
                self.save_model(model,
                                filename='iteration_%i_loss_%f' % (self.ctx_train['iteration'], loss.detach().item()))

        if self.ctx_train['scheduler_opt'].call_on == 'on_epoch':
            self.lr_scheduler_step(lr_scheduler, epoch, self.ctx_train['iteration'], len(train_loader))

        if self.multi_gpu:
            dist.barrier()

    def forward_train(self, model, loss_function, rank, batch):
        batch = self.batch_to_device(batch, rank)
        pred = model(batch['image'])
        if isinstance(pred, tuple):
            loss = loss_function(*pred, batch[self.gt_name])
        else:
            loss = loss_function(pred, batch[self.gt_name])
        return loss

    def validate(self, model, valid_loader, iteration, rank=0, loss_function=None):
        gpu = self.get_gpu_from_rank(rank)
        confMat = torch.zeros(self.n_classes, self.n_classes).cuda(gpu)
        losses = 0
        model.eval()
        for n, batch in enumerate(valid_loader):
            batch = self.batch_to_device(batch, rank)
            img = batch['image']
            gt = batch[self.gt_name]
            proba = model(img)

            losses += loss_function(proba, gt).detach()
            pred = torch.argmax(proba, 1)
            confMat += NNmetrics.confusion_matrix(pred, gt, num_classes=self.n_classes)

        if self.multi_gpu:
            confMat = reduce_tensor(confMat, self.world_size, mode='sum')
            losses = reduce_tensor(losses, self.world_size, mode='sum') / self.world_size

        losses = losses / n

        confMat = NNmetrics.filter_index_cm(confMat, self.ignore_index)
        mIoU = NNmetrics.mIoU_cm(confMat)
        if self.is_main_process(rank):
            stats = NNmetrics.report_cm(confMat)
            stats['mIoU'] = mIoU
            stats['validation_loss'] = losses.item()

            self.log_metrics(step=iteration, **stats)
            if self.tracked_metric is None:
                self.tracked_metric = mIoU
            else:
                if mIoU > self.tracked_metric:
                    self.tracked_metric = mIoU
                    filename = ('best_valid_iteration_%i_mIoU_%.3f' % (iteration, mIoU)).replace('.', '')
                    self.save_model(model, filename=filename)
        model.train()
        return mIoU
