import numpy as np
import tqdm
import logging
import torch
import copy
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from scones.models import GaussianScore
from datasets import get_dataset
from scones.models.langevin_dynamics import Langevin_dynamics

from compatibility.models import get_compatibility as _get_compatibility

__all__ = ['GaussianRunner']


def get_compatibility(config):
    # TODO: redesign cpat to avoid translation
    cnf_for_cpat = copy.deepcopy(config.compatibility)
    cnf_for_cpat.source = config.source
    cnf_for_cpat.target = config.target
    cnf_for_cpat.transport = config.transport
    return _get_compatibility(cnf_for_cpat)

def get_scorenet(config):
    cnf_for_ncsn = copy.deepcopy(config.ncsn)
    cnf_for_ncsn.data = config.target.data
    return GaussianScore(cnf_for_ncsn)

class GaussianRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample(self):
        score = get_scorenet(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        score.eval()

        if self.config.compatibility.ckpt_id is None:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, 'checkpoint.pth'),
                                     map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path,
                                                  f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                     map_location=self.config.device)

        cpat = get_compatibility(self.config)
        cpat.load_state_dict(cpat_states[0])

        source_dataset, _ = get_dataset(self.args, self.config.source)
        dataloader = DataLoader(source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        (Xs, _) = next(iter(dataloader))
        Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device).type(torch.float)

        if self.config.ncsn.sampling.data_init:
            init_Xt = torch.clone(Xs_global)
            init_Xt.requires_grad = True
            init_Xt = init_Xt.to(self.config.device)

        else:
            init_Xt = torch.rand(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1,
                                 device=self.config.device)
            init_Xt.requires_grad = True
            init_Xt = init_Xt.to(self.config.device)

        all_samples = Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                           self.config.ncsn.sampling.n_steps_each,
                                           self.config.ncsn.sampling.step_lr,
                                           verbose=True,
                                           final_only=self.config.ncsn.sampling.final_only)

        all_samples = torch.stack(all_samples, dim=0)

        all_samples = all_samples.view((-1,
                                        self.config.ncsn.sampling.sources_per_batch,
                                        self.config.ncsn.sampling.samples_per_source,
                                        self.config.target.data.dim))
        np.save(os.path.join(self.args.image_folder, 'all_samples.npy'), all_samples.detach().cpu().numpy())

        sample = all_samples[-1].view(self.config.ncsn.sampling.sources_per_batch *
                                      self.config.ncsn.sampling.samples_per_source,
                                      self.config.target.data.dim)

        np.save(os.path.join(self.args.image_folder, 'sources.npy'), Xs.detach().cpu().numpy().reshape((-1, self.config.source.data.dim)))
        np.save(os.path.join(self.args.image_folder, 'sample.npy'), sample.detach().cpu().numpy())

