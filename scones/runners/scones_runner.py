import numpy as np
import tqdm
import logging
import torch
import copy
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from ncsn.models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from ncsn.models.ncsn import NCSN, NCSNdeeper
from datasets import get_dataset, data_transform, inverse_data_transform
from scones.models import (anneal_Langevin_dynamics,
                                               anneal_Langevin_dynamics_inpainting,
                                               anneal_Langevin_dynamics_interpolation)

from ncsn.models import get_sigmas
from ncsn.models.ema import EMAHelper
from compatibility.models import get_compatibility as _get_compatibility
from baryproj.models import get_bary as _get_bary
from ncsn.runners import NCSNRunner
__all__ = ['SCONESRunner']

# todo: redesign the config to avoid translation in get_bary, get_compatibility, etc

def get_bary(config):
    cnf_for_bary = copy.deepcopy(config.baryproj)
    cnf_for_bary.source = config.source
    cnf_for_bary.target = config.target
    cnf_for_bary.transport = config.transport
    cnf_for_bary.device = config.device
    return _get_bary(cnf_for_bary)


def get_compatibility(config):
    cnf_for_cpat = copy.deepcopy(config.compatibility)
    cnf_for_cpat.source = config.source
    cnf_for_cpat.target = config.target
    cnf_for_cpat.transport = config.transport
    cnf_for_cpat.device = config.device
    return _get_compatibility(cnf_for_cpat)

def get_scorenet(config):
    cnf_for_ncsn = copy.deepcopy(config.ncsn)
    cnf_for_ncsn.data = config.target.data
    if config.target.data.dataset == 'CIFAR10' or config.target.data.dataset == 'CELEBA':
        return NCSNv2(cnf_for_ncsn).to(config.device)
    elif config.target.data.dataset == "FFHQ":
        return NCSNv2Deepest(cnf_for_ncsn).to(config.device)
    elif config.target.data.dataset == 'LSUN':
        return NCSNv2Deeper(cnf_for_ncsn).to(config.device)
    elif config.target.data.dataset == 'MNIST':
        return NCSN(cnf_for_ncsn).to(config.device)

class SCONESRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample(self):
        if self.config.ncsn.sampling.ckpt_id is None:
            ncsn_states = torch.load(os.path.join('scones',
                                                  self.config.ncsn.sampling.log_path,
                                                  'checkpoint.pth'),
                                     map_location=self.config.device)
        else:
            ncsn_states = torch.load(os.path.join('scones',
                                                  self.config.ncsn.sampling.log_path,
                                                  f'checkpoint_{self.config.ncsn.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_scorenet(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config.ncsn)
        sigmas = sigmas_th.cpu().numpy()

        ncsn_states[0]["module.sigmas"] = sigmas_th
        score.load_state_dict(ncsn_states[0], strict=True)

        score.eval()

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


        if self.config.compatibility.ckpt_id is None:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                map_location=self.config.device)

        cpat = get_compatibility(self.config)
        cpat.load_state_dict(cpat_states[0])

        if self.config.ncsn.model.ema:
            ema_helper = EMAHelper(mu=self.config.ncsn.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(ncsn_states[-1])
            ema_helper.ema(score)



        source_dataset, _ = get_dataset(self.args, self.config.source)
        dataloader = DataLoader(source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        (Xs, _) = next(iter(dataloader))
        Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)
        Xs_global = data_transform(self.config.source, Xs_global)

        if not self.config.ncsn.sampling.fid:
            if self.config.ncsn.sampling.inpainting:

                ''' NCSN INPAINTING CODE. EITHER PATCH THIS FOR SCONES OR REMOVE IT. 
                
                data_iter = iter(dataloader)
                refer_images, _ = next(data_iter)
                refer_images = refer_images.to(self.config.device)
                width = int(np.sqrt(self.config.sampling.batch_size))
                init_samples = torch.rand(width, width, self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)
                all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score,
                                                                  sigmas,
                                                                  self.config.data.image_size,
                                                                  self.config.sampling.n_steps_each,
                                                                  self.config.sampling.step_lr)

                torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

                if not self.config.sampling.final_only:
                    for i, sample in enumerate(tqdm.tqdm(all_samples)):
                        sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.ncsn.sampling.ckpt_id)))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'completion_{}.pth'.format(self.config.sampling.ckpt_id)))
                '''

                raise NotImplementedError("Inpainting with SCONES is not currently implemented.")
            elif self.config.ncsn.sampling.interpolation:
                ''' NCSN INTERPOLATION CODE. EITHER PATCH THIS FOR SCONES OR REMOVE IT. 
                
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics_interpolation(init_samples, score, sigmas,
                                                                     self.config.sampling.n_interpolations,
                                                                     self.config.sampling.n_steps_each,
                                                                     self.config.sampling.step_lr, verbose=True,
                                                                     final_only=self.config.sampling.final_only)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, nrow=self.config.sampling.n_interpolations)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, self.config.sampling.n_interpolations)
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                '''
                raise NotImplementedError("Interpolation with SCONES is not currently implemented.")
            else:
                if self.config.ncsn.sampling.data_init:
                    if(baryproj_data_init):
                        init_Xt = bproj(Xs_global) + sigmas_th[0] * torch.randn_like(Xs_global)
                    else:
                        init_Xt = Xs_global + sigmas_th[0] * torch.randn_like(Xs_global)

                    init_Xt.requires_grad = True
                    init_Xt = init_Xt.to(self.config.device)

                else:
                    init_Xt = torch.rand(self.config.ncsn.sampling.sources_per_batch *
                                              self.config.ncsn.sampling.samples_per_source,
                                              self.config.target.data.channels,
                                              self.config.target.data.image_size,
                                              self.config.target.data.image_size,
                                              device=self.config.device)
                    init_Xt = data_transform(self.config.target, init_Xt)
                    init_Xt.requires_grad = True
                    init_Xt = init_Xt.to(self.config.device)

                all_samples = anneal_Langevin_dynamics(init_Xt, Xs_global, score, cpat, sigmas,
                                                       self.config.ncsn.sampling.n_steps_each,
                                                       self.config.ncsn.sampling.step_lr,
                                                       verbose=True,
                                                       final_only=self.config.ncsn.sampling.final_only,
                                                       denoise=self.config.ncsn.sampling.denoise)

                all_samples = torch.stack(all_samples, dim=0)

                if not self.config.ncsn.sampling.final_only:
                    all_samples = all_samples.view((-1,
                                                    self.config.ncsn.sampling.sources_per_batch,
                                                    self.config.ncsn.sampling.samples_per_source,
                                                    self.config.target.data.channels,
                                                    self.config.target.data.image_size,
                                                    self.config.target.data.image_size))
                    np.save( os.path.join(self.args.image_folder, 'all_samples.npy'), all_samples.detach().cpu().numpy())

                sample = all_samples[-1].view(self.config.ncsn.sampling.sources_per_batch *
                                              self.config.ncsn.sampling.samples_per_source,
                                              self.config.target.data.channels,
                                              self.config.target.data.image_size,
                                              self.config.target.data.image_size)

                sample = inverse_data_transform(self.config.target, sample)

                image_grid = make_grid(sample, nrow=self.config.ncsn.sampling.sources_per_batch)
                save_image(image_grid, os.path.join(self.args.image_folder, 'sample_grid.png'))

                source_grid = make_grid(Xs, nrow=self.config.ncsn.sampling.sources_per_batch)
                save_image(source_grid, os.path.join(self.args.image_folder, 'source_grid.png'))

                np.save(os.path.join(self.args.image_folder, 'sources.npy'), Xs.detach().cpu().numpy())

        else:
            batch_size = self.config.ncsn.sampling.sources_per_batch * self.config.ncsn.sampling.samples_per_source
            total_n_samples = self.config.ncsn.sampling.num_samples4fid
            n_rounds = total_n_samples // batch_size
            if self.config.ncsn.sampling.data_init:
                dataloader = DataLoader(source_dataset,
                                        batch_size=self.config.ncsn.sampling.sources_per_batch,
                                        shuffle=True,
                                        num_workers=self.config.source.data.num_workers)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.ncsn.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                        samples = torch.cat([samples] * self.config.ncsn.sampling.samples_per_source, dim=0)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                        samples = torch.cat([samples] * self.config.ncsn.sampling.samples_per_source, dim=0)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config.target, samples)

                    if(baryproj_data_init):
                        with torch.no_grad():
                            samples = bproj(samples).detach()

                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                    samples.requires_grad = True
                    samples = samples.to(self.config.device)
                else:
                    samples = torch.rand(batch_size,
                                         self.config.target.data.channels,
                                         self.config.target.data.image_size,
                                         self.config.target.data.image_size, device=self.config.device)
                    samples = data_transform(self.config.target, samples)
                    samples.requires_grad = True
                    samples = samples.to(self.config.device)

                all_samples = anneal_Langevin_dynamics(samples, Xs_global, score, cpat, sigmas,
                                                       self.config.ncsn.sampling.n_steps_each,
                                                       self.config.ncsn.sampling.step_lr,
                                                       verbose=True,
                                                       final_only=self.config.ncsn.sampling.final_only,
                                                       denoise=self.config.ncsn.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config.target, img)
                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1

    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for pretrained with ema

        if self.config.ncsn.fast_fid.ensemble:
            if self.config.ncsn.model.ema:
                raise RuntimeError("Cannot apply ensembling to pretrained with EMA.")
            self.fast_ensemble_fid()
            return

        from ncsn.evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        source_dataset, _ = get_dataset(self.args, self.config.source)
        source_dataloader = DataLoader(source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        score = get_scorenet(self.config.ncsn)
        score = torch.nn.DataParallel(score)

        if self.config.compatibility.ckpt_id is None:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                map_location=self.config.device)

        cpat = get_compatibility(self.config)
        cpat.load_state_dict(cpat_states[0])

        sigmas_th = get_sigmas(self.config.ncsn)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.ncsn.fast_fid.begin_ckpt, self.config.ncsn.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.ncsn.model.ema:
                ema_helper = EMAHelper(mu=self.config.ncsn.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.ncsn.fast_fid.num_samples // self.config.ncsn.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                try:
                    (Xs, _) = next(iter(source_dataloader))
                    Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)
                except StopIteration:
                    source_dataloader = iter(source_dataset)
                    (Xs, _) = next(iter(source_dataloader))
                    Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)

                init_samples = torch.rand(self.config.ncsn.fast_fid.batch_size, self.config.target.data.channels,
                                          self.config.target.data.image_size, self.config.target.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config.target, init_samples)
                init_samples.requires_grad = True
                init_samples = init_samples.to(self.config.device)

                all_samples = anneal_Langevin_dynamics(init_samples, Xs_global, score, cpat, sigmas,
                                                       self.config.ncsn.fast_fid.n_steps_each,
                                                       self.config.ncsn.fast_fid.step_lr,
                                                       verbose=self.config.ncsn.fast_fid.verbose,
                                                       final_only=self.config.ncsn.sampling.final_only,
                                                       denoise=self.config.ncsn.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.target.data.channels,
                                         self.config.target.data.image_size,
                                         self.config.target.data.image_size)

                    sample = inverse_data_transform(self.config.target, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config.ncsn, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def fast_ensemble_fid(self):
        from ncsn.evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        num_ensembles = 5
        scores = [NCSN(self.config.ncsn).to(self.config.device) for _ in range(num_ensembles)]
        scores = [torch.nn.DataParallel(score) for score in scores]

        sigmas_th = get_sigmas(self.config.ncsn)
        sigmas = sigmas_th.cpu().numpy()

        if self.config.compatibility.ckpt_id is None:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                map_location=self.config.device)

        cpat = get_compatibility(self.config)
        cpat.load_state_dict(cpat_states[0])

        source_dataset, _ = get_dataset(self.args, self.config.source)
        source_dataloader = DataLoader(source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.ncsn.fast_fid.begin_ckpt, self.config.ncsn.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            begin_ckpt = max(self.config.ncsn.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * 5000)
            index = 0
            for i in range(begin_ckpt, ckpt + 5000, 5000):
                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{i}.pth'),
                                    map_location=self.config.device)
                scores[index].load_state_dict(states[0])
                scores[index].eval()
                index += 1

            def scorenet(x, labels):
                num_ckpts = (ckpt - begin_ckpt) // 5000 + 1
                return sum([scores[i](x, labels) for i in range(num_ckpts)]) / num_ckpts

            num_iters = self.config.ncsn.fast_fid.num_samples // self.config.ncsn.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                try:
                    (Xs, _) = next(iter(source_dataloader))
                    Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)
                except StopIteration:
                    source_dataloader = iter(source_dataset)
                    (Xs, _) = next(iter(source_dataloader))
                    Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)

                init_samples = torch.rand(self.config.ncsn.fast_fid.batch_size, self.config.target.data.channels,
                                          self.config.target.data.image_size, self.config.target.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config.target, init_samples)
                init_samples.requires_grad = True
                init_samples = init_samples.to(self.config.device)

                all_samples = anneal_Langevin_dynamics(init_samples, Xs_global, scorenet, cpat, sigmas,
                                                       self.config.ncsn.fast_fid.n_steps_each,
                                                       self.config.ncsn.fast_fid.step_lr,
                                                       verbose=self.config.ncsn.fast_fid.verbose,
                                                       final_only=self.config.ncsn.sampling.final_only,
                                                       denoise=self.config.ncsn.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.target.data.channels,
                                         self.config.target.data.image_size,
                                         self.config.target.data.image_size)

                    sample = inverse_data_transform(self.config.target, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config.ncsn, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)
