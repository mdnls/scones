import torch
import logging
from compatibility.models.imagecritic import FCImageCritic, ImageCritic
from compatibility.models.compatibility import Compatibility
from compatibility.models import get_compatibility
from torch.utils.data import DataLoader
from ncsn.losses import get_optimizer
from compatibility.models import get_cost
from datasets import get_dataset
from tqdm import tqdm
import os
from datasets import data_transform


class CpatRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def train(self):
        source_dataset, source_test_dataset = get_dataset(self.args, self.config.source)
        source_loader = DataLoader(source_dataset, batch_size=self.config.training.batch_size,
                                   shuffle=True, num_workers=self.config.source.data.num_workers, drop_last=True)
        source_batches = iter(source_loader)

        target_dataset, target_test_dataset = get_dataset(self.args, self.config.target)
        target_loader = DataLoader(target_dataset, batch_size=self.config.training.batch_size,
                                   shuffle=True, num_workers=self.config.target.data.num_workers, drop_last=True)
        target_batches = iter(target_loader)

        cpat = get_compatibility(self.config)
        cpat_opt = get_optimizer(self.config, cpat.parameters())

        if(self.args.resume_training):
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'))
            cpat.load_state_dict(states[0])
            cpat_opt.load_state_dict(states[1])
            logging.info(f"Resuming training after {states[2]} steps.")


        logging.info("Optimizing the compatibility function.")
        with tqdm(total=self.config.training.n_iters) as progress:
            for d_step in range(self.config.training.n_iters):

                try:
                    (Xs, ys) = next(source_batches)
                    (Xt, yt) = next(target_batches)
                except StopIteration:
                    # Refresh after one epoch
                    source_batches = iter(source_loader)
                    target_batches = iter(target_loader)
                    (Xs, ys) = next(source_batches)
                    (Xt, yt) = next(target_batches)

                Xs = data_transform(self.config.source, Xs)
                Xs = Xs.to(self.config.device)

                Xt = data_transform(self.config.target, Xt)
                Xt = Xt.to(self.config.device)

                obj = cpat_opt.step(lambda: self._cpat_closure(Xs, Xt, cpat, cpat_opt))
                avg_density = torch.mean(cpat.forward(Xs, Xt))

                obj_val = round(obj.item(), 5)
                avg_density_val = round(avg_density.item(), 5)
                progress.update(1)
                progress.set_description_str(f"Average Density: {avg_density_val}")
                self.config.tb_logger.add_scalars('Optimization', {
                    'Objective': obj_val,
                    'Average Density': avg_density_val
                }, d_step)

                if(d_step % self.config.training.snapshot_freq == 0):
                    states = [
                        cpat.state_dict(),
                        cpat_opt.state_dict(),
                        d_step
                    ]

                    torch.save(states, os.path.join(self.args.log_path, f'checkpoint_{d_step}.pth'))
                    torch.save(states, os.path.join(self.args.log_path, f'checkpoint.pth'))

    def _cpat_closure(self, Xs, Xt, cpat, cpat_opt):
        cpat_opt.zero_grad()
        density_real_inp = cpat.inp_density_param(Xs)
        density_real_outp = cpat.outp_density_param(Xt)
        density_reg = cpat.penalty(Xs, Xt)
        obj = torch.mean(density_real_inp + density_real_outp - density_reg)
        (-obj).backward() # for gradient ascent rather than descent
        return obj

