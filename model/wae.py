# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 上午8:53
# @Author  : WuDiDaBinGe
# @FileName: dataloader.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class WAE(nn.Module):
    def __init__(self, encode_dims=[2000, 1024, 512, 20], decode_dim=[20, 1024, 2000], dropout=0.0, nonlin='relu'):
        super(WAE, self).__init__()
        self.encoder = nn.ModuleDict({
            f'encode_{i}': nn.Linear(encode_dims[i], encode_dims[i + 1])
            for i in range(len(encode_dims) - 1)
        })

        self.decoder = nn.ModuleDict({
            f'decode_{i}': nn.Linear(decode_dim[i], decode_dim[i + 1])
            for i in range(len(decode_dim) - 1)
        })
        assert encode_dims[-1] == decode_dim[0]
        self.latent_dim = decode_dim[1]
        self.dropout = nn.Dropout(p=dropout)
        self.nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[nonlin]

    def encode(self, x):
        hid = x
        for i, (_, layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder) - 1:
                hid = self.nonlin(hid)
        return hid

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder) - 1:
                hid = self.nonlin(self.dropout(hid))
        return hid

    def forward(self, x):
        theta = self.encode(x)
        theta = F.softmax(theta, dim=-1)
        x_reconst = self.decode(theta)
        return x_reconst, theta

    def sample(self, dist='dirichlet', batch_size=256, dirichlet_alpha=0.1, ori_data=None):
        if dist == 'dirichlet':
            z_true = np.random.dirichlet(
                np.ones(self.latent_dim) * dirichlet_alpha, size=batch_size)
            z_true = torch.from_numpy(z_true).float()
        elif dist == 'gaussian':
            z_true = np.random.randn(batch_size, self.z_dim)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true
        elif dist == 'gmm_std':
            odes = np.eye(self.z_dim) * 20
            ides = np.random.randint(low=0, high=self.z_dim, size=batch_size)
            mus = odes[ides]
            sigmas = np.ones((batch_size, self.z_dim)) * 0.2 * 20
            z_true = np.random.normal(mus, sigmas)
            z_true = F.softmax(torch.from_numpy(z_true).float(), dim=1)
            return z_true
        elif dist == 'gmm_ctm' and ori_data is not None:
            with torch.no_grad():
                hid_vecs = self.encode(ori_data).cpu().numpy()
                # 高斯混合分布 由多个子高斯模型混合
                gmm = GaussianMixture(n_components=self.latent_dim, covariance_type='full', max_iter=200)
                gmm.fit(hid_vecs)
                gmm_spls, _spl_lbs = gmm.sample(n_samples=len(ori_data))
                theta_prior = torch.from_numpy(gmm_spls).float()
                theta_prior = F.softmax(theta_prior, dim=1)
                return theta_prior
        else:
            return self.sample(dist='dirichlet', batch_size=batch_size)

    def mmd_loss(self, x, y, device, t=0.1, kernel='diffusion'):
        '''
                computes the mmd loss with information diffusion kernel
                :param x: batch_size * latent dimension
                :param y:
                :param t:
                :return:
                '''
        eps = 1e-6
        n, d = x.shape
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i + 1, n):
                    sum_xx = sum_xx + torch.norm(x[i] - x[j], p=1).to(device)
            sum_xx = sum_xx / (n * (n - 1))

            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i + 1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i] - y[j], p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0] - 1))

            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i] - y[j], p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt, dim):
                # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
                return torch.exp(-torch.acos(a).pow(2)) / tmpt

            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1 - eps), t, d - 1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1 - eps), t, d - 1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1 - eps), t, d - 1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n - 1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n - 1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy


if __name__ == '__main__':
    model = WAE(encode_dims=[1024, 512, 256, 20],
                decode_dim=[20, 128, 768, 1024])
    model = model.cuda()
    inpt = torch.randn(16, 1024).cuda()
    out, mu = model(inpt)
    print(torch.sum(mu, dim=-1))
    print(out.shape)
    print(mu.shape)
