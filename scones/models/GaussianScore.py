import numpy as np
import torch.nn as nn
import torch
import scipy

class GaussianScore(nn.Module):
    def __init__(self, config):
        super(GaussianScore, self).__init__()
        mean = torch.FloatTensor(config.data.mean).to(config.device).view((1, -1, 1))
        cov_pinv = torch.FloatTensor(np.linalg.pinv(config.data.cov)).to(config.device)
        self.mean = torch.nn.Parameter(mean)
        self.precision = torch.nn.Parameter(cov_pinv)
        self.config = config

    def forward(self, x):
        dim = self.config.data.dim
        r = - self.precision @ (x.view((-1, dim, 1)) - self.mean)
        return r.view((-1, dim, 1, 1))

class GaussianCpat(nn.Module):
    def __init__(self, config, mu_source, cov_source, mu_target, cov_target):
        super(GaussianCpat, self).__init__()

        def _2np(x):
            return x.detach().cpu().numpy()

        def sinkhorn(mu_a, cov_a, mu_b, cov_b, lmbda):
            d = len(mu_a)  # dimension
            # lmbda is SCONES regularization parameter,
            #   2 sigma**2 = lambda is used in the paper
            sigma = np.sqrt(lmbda / 2)
            Asq = scipy.linalg.sqrtm(cov_a)
            Asqinv = np.linalg.pinv(Asq)
            Ds = scipy.linalg.sqrtm(4 * Asq @ cov_b @ Asq + sigma ** 4 * np.eye(d))
            Cs = (1 / 2) * Asq @ Ds @ Asqinv - (sigma ** 2 / 2) * np.eye(d)
            sinkhorn_mean = np.concatenate((mu_a, mu_b), axis=0)
            sinkhorn_cov = np.real_if_close(np.block([[cov_a, Cs], [Cs.T, cov_b]]))
            return sinkhorn_mean, sinkhorn_cov

        _, joint_cov = sinkhorn(_2np(mu_source), _2np(cov_source),
                                _2np(mu_target), _2np(cov_target), config.transport.coeff)

        joint_precision = np.linalg.pinv(joint_cov)

        cross_block = joint_precision[:len(mu_source), len(mu_source):]
        target_block = joint_precision[len(mu_source):, len(mu_source):]

        self.source_mean = mu_source.view((1, -1, 1))
        self.target_mean = mu_target.view((1, -1, 1))
        self.joint_precision = torch.FloatTensor(joint_precision).to(mu_target.device)
        self.joint_precision_cross = torch.FloatTensor(cross_block).to(mu_target.device)
        self.joint_precision_target = torch.FloatTensor(target_block).to(mu_target.device)
        self.marginal_precision_target = torch.FloatTensor(np.linalg.pinv(_2np(cov_target))).to(mu_target.device)

        self.config = config

    def score(self, source, tgt):
        dim = self.config.source.data.dim
        r = - self.joint_precision_cross.T @ (source.view((-1, dim, 1)) - self.source_mean)
        s = - (self.joint_precision_target - self.marginal_precision_target) @ (tgt.view((-1, dim, 1)) - self.target_mean)
        return r.view((-1, dim, 1, 1)), s.view((-1, dim, 1, 1))

    def load_state_dict(self, state_dict, strict=False):
        pass

class JointMarginalGaussianCpat(nn.Module):
    def __init__(self, config, mu_source, cov_source, mu_target, cov_target):
        super(JointMarginalGaussianCpat, self).__init__()

        def _2np(x):
            return x.detach().cpu().numpy()

        def sinkhorn(mu_a, cov_a, mu_b, cov_b, lmbda):
            d = len(mu_a)  # dimension
            # lmbda is SCONES regularization parameter,
            #   2 sigma**2 = lambda is used in the paper
            sigma = np.sqrt(lmbda / 2)
            Asq = scipy.linalg.sqrtm(cov_a)
            Asqinv = np.linalg.pinv(Asq)
            Ds = scipy.linalg.sqrtm(4 * Asq @ cov_b @ Asq + sigma ** 4 * np.eye(d))
            Cs = (1 / 2) * Asq @ Ds @ Asqinv - (sigma ** 2 / 2) * np.eye(d)
            sinkhorn_mean = np.concatenate((mu_a, mu_b), axis=0)
            sinkhorn_cov = np.real_if_close(np.block([[cov_a, Cs], [Cs.T, cov_b]]))
            return sinkhorn_mean, sinkhorn_cov

        _, joint_cov = sinkhorn(_2np(mu_source), _2np(cov_source),
                                _2np(mu_target), _2np(cov_target), config.transport.coeff)

        joint_precision = np.linalg.pinv(joint_cov)

        cross_block = joint_precision[:len(mu_source), len(mu_source):]
        target_block = joint_precision[:len(mu_source), :len(mu_source)]

        self.source_mean = mu_source.view((1, -1, 1))
        self.target_mean = mu_target.view((1, -1, 1))
        self.joint_precision = torch.FloatTensor(joint_precision).to(mu_target.device)
        self.joint_precision_cross = torch.FloatTensor(cross_block).to(mu_target.device)
        self.joint_precision_target = torch.FloatTensor(target_block).to(mu_target.device)
        self.marginal_precision_target = torch.FloatTensor(np.linalg.pinv(_2np(cov_target))).to(mu_target.device)

        self.config = config

    def score(self, source, tgt):
        dim = self.config.source.data.dim
        joint = torch.cat((source.view((-1, dim, 1)) - self.source_mean, tgt.view((-1, dim, 1)) - self.target_mean), dim=1)
        r = - self.joint_precision @ joint
        return r[:, dim:, :].view((-1, dim, 1, 1))

    def load_state_dict(self, state_dict, strict=False):
        pass
