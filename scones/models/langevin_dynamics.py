import torch
import numpy as np

def anneal_Langevin_dynamics(tgt, source, scorenet, cpat, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, n_sigmas_skip=0):
    images = []

    for c, sigma in enumerate(sigmas):
        if(c < n_sigmas_skip):
            continue
        labels = torch.ones(tgt.shape[0], device=tgt.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(tgt, labels)
                noise = torch.randn_like(tgt)
            cpat_grad = cpat.score(source, tgt)

            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            cpat_grad_norm = torch.norm(cpat_grad.view(cpat_grad.shape[0], -1), dim=-1).mean()

            tgt = tgt + step_size * (grad + cpat_grad) + noise * np.sqrt(step_size * 2)

            image_norm = torch.norm(tgt.view(tgt.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(tgt.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, cpat_grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), cpat_grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        cpat_score = cpat.score(source, tgt).detach()
        with torch.no_grad():
            last_noise = (len(sigmas) - 1) * torch.ones(tgt.shape[0], device=tgt.device)
            last_noise = last_noise.long()
            tgt = tgt + sigmas[-1] ** 2 * (scorenet(tgt, last_noise) + cpat_score)
            images.append(tgt.to('cpu'))

    if final_only:
        return [tgt.to('cpu')]
    else:
        return images


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            if(c < n_sigma_skip):
                continue
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images


def Langevin_dynamics(tgt, source, scorenet, cpat, n_steps, step_lr=0.000008, final_only=False, sample_every=1, verbose=False):
    images = []

    for s in range(n_steps):
        with torch.no_grad():
            grad = scorenet(tgt)
            noise = torch.randn_like(tgt)
        cpat_grad = cpat.score(source, tgt)

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
        cpat_grad_norm = torch.norm(cpat_grad.view(cpat_grad.shape[0], -1), dim=-1).mean()

        tgt = tgt + step_lr * (grad + cpat_grad) + noise * np.sqrt(step_lr * 2)

        image_norm = torch.norm(tgt.view(tgt.shape[0], -1), dim=-1).mean()
        snr = np.sqrt(step_lr / 2.) * grad_norm / noise_norm
        grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2

        if not final_only and (s % sample_every == 0):
            images.append(tgt.to('cpu'))
        if verbose:
            print("step: {}, grad_norm: {}, cpat_grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                s, grad_norm.item(), cpat_grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if final_only:
        return [tgt.to('cpu')]
    else:
        return images