import numpy as np
import tqdm
import logging
import torch
import copy
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from scones.models import GaussianScore, GaussianCpat, JointMarginalGaussianCpat
from datasets import get_dataset
from scones.models.langevin_dynamics import Langevin_dynamics, compare_Langevin_dynamics, joint_marginal_Langevin_dynamics
from baryproj.models import get_bary as _get_bary
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from compatibility.models import get_compatibility as _get_compatibility

__all__ = ['GaussianRunner']


def get_compatibility(args, config):
    # TODO: redesign cpat to avoid translation
    print("USING AD HOC MODIFIED GAUSSIAN CPAT. REMEMBER TO REMOVE ARGS")
    if(config.compatibility.model.architecture == "comparison"):
        source_dataset, _ = get_dataset(args, config.source)
        target_dataset, _ = get_dataset(args, config.target)
        mu_source = source_dataset.mean
        cov_source = source_dataset.cov
        mu_target = target_dataset.mean
        cov_target = target_dataset.cov

        cnf_for_cpat = copy.deepcopy(config.compatibility)
        cnf_for_cpat.source = config.source
        cnf_for_cpat.target = config.target
        cnf_for_cpat.transport = config.transport
        cnf_for_cpat.model.architecture = "fcn"
        return _get_compatibility(cnf_for_cpat), GaussianCpat(config, mu_source, cov_source, mu_target, cov_target)
    if(config.compatibility.model.architecture == "true_gaussian"):
        source_dataset, _ = get_dataset(args, config.source)
        target_dataset, _ = get_dataset(args, config.target)
        mu_source = source_dataset.mean
        cov_source = source_dataset.cov
        mu_target = target_dataset.mean
        cov_target = target_dataset.cov
        return GaussianCpat(config, mu_source, cov_source, mu_target, cov_target)
    if(config.compatibility.model.architecture == "joint_marginal"):
        source_dataset, _ = get_dataset(args, config.source)
        target_dataset, _ = get_dataset(args, config.target)
        mu_source = source_dataset.mean
        cov_source = source_dataset.cov
        mu_target = target_dataset.mean
        cov_target = target_dataset.cov
        return JointMarginalGaussianCpat(config, mu_source, cov_source, mu_target, cov_target)

    cnf_for_cpat = copy.deepcopy(config.compatibility)
    cnf_for_cpat.source = config.source
    cnf_for_cpat.target = config.target
    cnf_for_cpat.transport = config.transport
    return _get_compatibility(cnf_for_cpat)

def get_scorenet(config, target):
    cnf_for_ncsn = copy.deepcopy(config.ncsn)
    cnf_for_ncsn.data = config.target.data
    cnf_for_ncsn.data.mean = target.mean.detach().cpu().numpy()
    cnf_for_ncsn.data.cov = target.cov.detach().cpu().numpy()
    return GaussianScore(cnf_for_ncsn)

def get_bary(config):
    cnf_for_bproj = copy.deepcopy(config.compatibility)
    cnf_for_bproj.source = config.source
    cnf_for_bproj.target = config.target
    cnf_for_bproj.transport = config.transport
    return _get_bary(cnf_for_bproj)

class GaussianRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample(self):
        if self.config.compatibility.ckpt_id is None:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, 'checkpoint.pth'),
                                     map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path,
                                                  f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                     map_location=self.config.device)

        cpat = get_compatibility(self.args, self.config)
        print("REMOVE LINE 83 COMPARISON")
        if(self.config.compatibility.model.architecture == "comparison"):
            cpat[0].load_state_dict(cpat_states[0])
        else:
            cpat.load_state_dict(cpat_states[0])


        baryproj_data_init = (hasattr(self.config, "baryproj") and self.config.ncsn.sampling.data_init)

        if(baryproj_data_init):
            if(self.config.baryproj.ckpt_id is None):
                bproj_states = torch.load(
                    os.path.join('scones', self.config.baryproj.log_path, 'checkpoint.pth'),
                    map_location=self.config.device)
            else:
                bproj_states = torch.load(os.path.join('scones', self.config.baryproj.log_path,
                                                      f'checkpoint_{self.config.baryproj.ckpt_id}.pth'),
                                         map_location=self.config.device)

            bproj = get_bary(self.config)
            bproj.load_state_dict(bproj_states[0])
            bproj = torch.nn.DataParallel(bproj)
            bproj.eval()

        source_dataset, _ = get_dataset(self.args, self.config.source)
        dataloader = DataLoader(source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        target_dataset, _ = get_dataset(self.args, self.config.target)
        target_dataloader = DataLoader(target_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        score = get_scorenet(self.config, target_dataset).to(self.config.device)
        score = torch.nn.DataParallel(score)
        score.eval()

        (Xs, _) = next(iter(dataloader))
        Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device).type(torch.float)
        Xs_global = Xs_global.view(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1)



        if baryproj_data_init:
            Xs_init = torch.clone(Xs_global).to(self.config.device).view((-1, self.config.source.data.dim))
            init_Xt = bproj(Xs_init).detach()


            xs = Xs_init.detach().cpu().numpy()
            xt = init_Xt.cpu().numpy()
            pair = np.concatenate([xs, xt], axis=1)
            print(np.cov(pair, rowvar=False))
            lines = [[(xs[i][0], xt[i][0]), (xs[i][1], xt[i][1])] for i in range(100)]
            lc = LineCollection(lines)
            fig, ax = plt.subplots()
            ax.add_collection(lc)
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.show()


            init_Xt = init_Xt.view((self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1))
            init_Xt.requires_grad = True
        elif self.config.ncsn.sampling.data_init:
            init_Xt = torch.clone(Xs_global)
            init_Xt.requires_grad = True
            init_Xt = init_Xt.to(self.config.device)

        else:
            init_Xt = torch.zeros(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1,
                                 device=self.config.device)
            init_Xt.requires_grad = True
            init_Xt = init_Xt.to(self.config.device)

        print("REMOVE COMPARISOn 116")
        if(self.config.compatibility.model.architecture == "comparison"):
            all_samples = compare_Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                            self.config.ncsn.sampling.n_steps_each,
                                            self.config.ncsn.sampling.step_lr,
                                            verbose=True,
                                            sample_every=self.config.ncsn.sampling.sample_every,
                                            final_only=self.config.ncsn.sampling.final_only)
        elif(self.config.compatibility.model.architecture == "joint_marginal"):
            all_samples = joint_marginal_Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                                    self.config.ncsn.sampling.n_steps_each,
                                                    self.config.ncsn.sampling.step_lr,
                                                    verbose=True,
                                                    sample_every=self.config.ncsn.sampling.sample_every,
                                                    final_only=self.config.ncsn.sampling.final_only)
        else:
            all_samples = Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                               self.config.ncsn.sampling.n_steps_each,
                                               self.config.ncsn.sampling.step_lr,
                                               verbose=True,
                                               sample_every=self.config.ncsn.sampling.sample_every,
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

