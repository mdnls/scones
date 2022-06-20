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

                progress.update(1)
                progress.set_description_str("L2 Error: {:.4e}".format(obj.item()))
                self.config.tb_logger.add_scalars('Optimization', {
                    'Objective': obj.item()
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
        nnz = (dx > 1e-20).flatten()
        Xt_hat = bp(Xs)
        transport_cost = ((Xt[nnz] - Xt_hat[nnz])**2).flatten(start_dim=1)
        cost = torch.mean(transport_cost, dim=1, keepdim=True) * dx[nnz]
        obj = torch.mean(cost)
        obj.backward()
        return obj

    def sample(self):
        source_dataset, _ = get_dataset(self.args, self.config.source)

        baryproj = get_bary(self.config)
        baryproj.eval()

        if self.config.sampling.ckpt_id is None:
            bp_states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            bp_states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.compatibility.ckpt_id}.pth'),
                                     map_location=self.config.device)

        baryproj.load_state_dict(bp_states[0])

        if(not self.config.sampling.fid):
            dataloader = DataLoader(source_dataset,
                                    batch_size=self.config.sampling.batch_size,
                                    shuffle=True,
                                    num_workers=self.config.source.data.num_workers)

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

        else:
            batch_size = self.config.sampling.samples_per_batch
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // batch_size

            dataloader = DataLoader(source_dataset,
                                    batch_size=self.config.sampling.samples_per_batch,
                                    shuffle=True,
                                    num_workers=self.config.source.data.num_workers)
            data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation.'):
                with torch.no_grad():
                    (Xs, _) = next(data_iter)
                    Xs = data_transform(self.config.source, Xs).to(self.config.device)
                    transport = baryproj(Xs)
                for img in transport:
                    img = inverse_data_transform(self.config.target, img)
                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1
                del Xs
                del transport

