import numpy as np
import glob
from tqdm import tqdm
import torch.nn.functional as F
import logging
import torch
import torchvision
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from datasets import get_dataset, data_transform, inverse_data_transform
from scones.runners.scones_runner import get_compatibility
from baryproj.models import get_bary
from ncsn.losses import get_optimizer

__all__ = ['BPRunner']

class BPRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        source_dataset, source_test_dataset = get_dataset(self.args, self.config.source)
        source_loader = DataLoader(source_dataset, batch_size=self.config.training.batch_size,
                                   shuffle=True, num_workers=self.config.source.data.num_workers, drop_last=True)
        source_batches = iter(source_loader)

        target_dataset, target_test_dataset = get_dataset(self.args, self.config.target)
        target_loader = DataLoader(target_dataset, batch_size=self.config.training.batch_size,
                                   shuffle=True, num_workers=self.config.target.data.num_workers, drop_last=True)
        target_batches = iter(target_loader)

        if self.config.compatibility.ckpt_id is None:
            states = torch.load(os.path.join('baryproj', self.config.compatibility.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join('baryproj', self.config.compatibility.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                map_location=self.config.device)

        cpat = get_compatibility(self.config)
        cpat.load_state_dict(states[0])
        cpat.eval()

        baryproj = get_bary(self.config)
        bp_opt = get_optimizer(self.config, baryproj.parameters())

        if(self.args.resume_training):
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pt'))
            baryproj.load_state_dict(states[0])
            bp_opt.load_state_dict(states[1])
            logging.info(f"Resuming training after {states[2]} steps.")

        logging.info("Optimizing the barycentric projection of the OT map.")

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

                obj = bp_opt.step(lambda: self._bp_closure(Xs, Xt, cpat, baryproj, bp_opt))

                obj_val = round(obj.item(), 5)
                progress.update(1)
                progress.set_description_str(f"L2 Error: {obj_val}")
                self.config.tb_logger.add_scalars('Optimization', {
                    'Objective': obj_val
                }, d_step)

                if(d_step % self.config.training.sample_freq == 0):
                    with torch.no_grad():
                        samples = baryproj(Xs)
                    img_grid1 = torchvision.utils.make_grid(torch.clamp(samples, 0, 1))
                    img_grid2 = torchvision.utils.make_grid(torch.clamp(Xs, 0, 1))
                    self.config.tb_logger.add_image('Samples', img_grid1, d_step)
                    self.config.tb_logger.add_image('Sources', img_grid2, d_step)

                if(d_step % self.config.training.snapshot_freq == 0):
                    states = [
                        baryproj.state_dict(),
                        d_step
                    ]

                    torch.save(states, os.path.join(self.args.log_path, f'checkpoint_{d_step}.pth'))
                    torch.save(states, os.path.join(self.args.log_path, f'checkpoint.pth'))

    def _bp_closure(self, Xs, Xt, cpat, bp, bp_opt):
        bp_opt.zero_grad()
        dx = cpat(Xs, Xt)
        Xt_hat = bp(Xs)
        transport_cost = ((Xt - Xt_hat)**2).flatten(start_dim=1)
        cost = torch.mean(transport_cost, dim=1, keepdim=True) * dx
        obj = torch.mean(cost)
        obj.backward()
        return obj

    def sample(self):
        if self.config.compatibility.ckpt_id is None:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join('scones', self.config.compatibility.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                map_location=self.config.device)

        cpat = get_compatibility(self.config)
        cpat.load_state_dict(cpat_states[0])

        source_dataset, _ = get_dataset(self.args, self.config.source)
        dataloader = DataLoader(source_dataset,
                                batch_size=self.config.sampling.batch_size,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)

        baryproj = get_bary(self.config)
        baryproj.eval()

        if self.config.sampling.ckpt_id is None:
            bp_states = torch.load(os.path.join('bp', self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            bp_states = torch.load(os.path.join('bp', self.args.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                     map_location=self.config.device)

        baryproj.load_state_dict(bp_states[0])

        batch_samples = []
        for i in range(self.config.sampling.n_batches):
            (Xs, _) = next(iter(dataloader))
            Xs = data_transform(self.config.source, Xs)
            transport = baryproj(Xs)
            batch_samples.append(inverse_data_transform(self.config, transport))

        sample = torch.cat(batch_samples, dim=0)

        image_grid = make_grid(sample[:min(64, len(sample))], nrow=8)
        save_image(image_grid, os.path.join(self.args.image_folder, 'sample_grid.png'))

        source_grid = make_grid(Xs[:min(64, len(Xs))], nrow=8)
        save_image(source_grid, os.path.join(self.args.image_folder, 'source_grid.png'))

        np.save(os.path.join(self.args.image_folder, 'sample.npy'), sample.detach().cpu().numpy())
        np.save(os.path.join(self.args.image_folder, 'sources.npy'), Xs.detach().cpu().numpy())

    def fast_fid(self):
        '''
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for pretrained with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to pretrained with EMA.")
            self.fast_ensemble_fid()
            return

        from ncsn.evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''

    def fast_ensemble_fid(self):
        '''
        from ncsn.evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        num_ensembles = 5
        scores = [NCSN(self.config).to(self.config.device) for _ in range(num_ensembles)]
        scores = [torch.nn.DataParallel(score) for score in scores]

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            begin_ckpt = max(self.config.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * 5000)
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

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, scorenet, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''