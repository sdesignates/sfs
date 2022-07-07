import sys, os

sys.path.append('./')
import numpy as np
import torch
from torch_sqrtm import MatrixSquareRoot


import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_sqrtm = MatrixSquareRoot.apply



def isPSD(A, tol=1e-5):

    E, V = torch.eig(A, eigenvectors=False)
    E = E[:, 0].squeeze()
    return torch.all(E > -tol) & torch.allclose(A, A.T, atol=tol), torch.min(E)


def find_clostest_PSD(A):
    """\
    https://math.stackexchange.com/questions/648809/how-to-find-closest-positive-definite-matrix-of-non-symmetric-matrix
    Positive definite matrices are a open set (Positive semidefinite is closed), no cloest point, uses 1e-4 for the minimum eigenvalue
    """
    S, U = torch.eig(A, eigenvectors=True)
    S = S[:, 0]
    S[S <= 0] = 1e-3
    return U @ torch.diag(S) @ U.t()


def weighted_kendall_tau(xs, ys, ws=None):
    """\
    Description:
    ------------
        Calculate weighted kendall tau score, mxi, mxj, myi, myj are use to account for the missing values
        Updated matrix version
    Parameters:
    -----------
        xs: the first array
        ys: the second array
        ws: the weight
    """
    n = xs.shape[0]
    if ws is None:
        ws = torch.ones(n)  # .to(device)
    assert ys.shape[0] == n
    assert ws.shape[0] == n
    # mask array
    mx = (xs != 0)
    my = (ys != 0)

    signed_diffx = torch.sign(xs[:, None] - xs[None, :])
    signed_diffy = torch.sign(ys[:, None] - ys[None, :])
    w_mat = mx[:, None] * my[:, None] * mx[None, :] * my[None, :] * ws[:, None] * ws[None, :]
    w_mat.fill_diagonal_(0)
    kt = torch.sum(signed_diffx * signed_diffy * w_mat) / (torch.sum(w_mat) + 1e-12)
    return kt


def est_cov(X, K_trun, weighted_kt=True):
    start_time = time.time()
    X = torch.FloatTensor(X)
    ntimes = X.shape[0]
    ngenes = X.shape[1]

    empir_cov = torch.zeros(ntimes, ngenes, ngenes)  # .to(device)

    # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
    for t in range(ntimes):
        weight = torch.FloatTensor(K_trun[t, :])  # .to(device)
        if weighted_kt:
            weight = (weight > 0)

        for i in range(ngenes):
            for j in range(i, ngenes):
                if weighted_kt:
                    kt = weighted_kendall_tau(X[(weight > 0), i].squeeze(), X[(weight > 0), j].squeeze(),
                                              weight[weight > 0])
                    if i != j:
                        empir_cov[t, i, j] = torch.sin(np.pi / 2 * kt)
                    else:
                        empir_cov[t, i, j] = 1

                else:
                    kt = weighted_kendall_tau(X[(weight > 0), i].squeeze(), X[(weight > 0), j].squeeze(),
                                              weight[weight > 0])
                    if i != j:
                        empir_cov[t, i, j] = torch.sin(np.pi / 2 * kt)
                        assert empir_cov[t, i, j] < 1
                    else:
                        empir_cov[t, i, j] = 1
        empir_cov[t, :, :] = empir_cov[t, :, :] + empir_cov[t, :, :].T - torch.diag(torch.diag(empir_cov[t, :, :]))
        # check positive definite
        Flag, min_eig = isPSD(empir_cov[t, :, :])

        if not Flag:
            empir_cov[t, :, :] = find_clostest_PSD(empir_cov[t, :, :])
            Flag, min_eig = isPSD(empir_cov[t, :, :])

            assert Flag

    print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))
    return empir_cov


class G_admm_batch():
    def __init__(self, X, K, TF=None, seed=0, pre_cov=None, batchsize=None):
        super(G_admm_batch, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, nsamples, ngenes)
        self.X = torch.FloatTensor(X)
        self.ntimes, self.nsamples, self.ngenes = self.X.shape

        # calculate batchsize
        if batchsize is None:
            self.batchsize = int(self.ntimes / 10)
        else:
            self.batchsize = batchsize
        # calculate empirical covariance matrix
        if pre_cov is None:
            # shape (ntimes, nsamples, ngenes)
            self.epir_mean = self.X.mean(dim=1, keepdim=True)
            X = self.X - self.epir_mean
            # (ntimes * nsamples, ngenes, ngenes)
            self.empir_cov = torch.bmm(X.reshape((self.ntimes * self.nsamples, self.ngenes, 1)),
                                       X.reshape((self.ntimes * self.nsamples, 1, self.ngenes)))
            # (ntimes, ngenes, ngenes)
            self.empir_cov = torch.sum(self.empir_cov.reshape((self.ntimes, self.nsamples, self.ngenes, self.ngenes)),
                                       dim=1) / (self.nsamples - 1)
        else:
            self.empir_cov = pre_cov

        # weight kernel function, shape (ntimes, ntimes)
        self.weights = torch.FloatTensor(K)
        # weighted average of empricial covariance matrix
        assert torch.all(torch.sum(self.weights, dim=1) - 1 < 1e-6)
        self.w_empir_cov = torch.stack([torch.sum(weight[:, None, None] * self.empir_cov, axis=0) for weight in self.weights], dim=0)
        assert self.w_empir_cov.shape[0] == self.ntimes
        assert self.w_empir_cov.shape[1] == self.ngenes
        assert self.w_empir_cov.shape[2] == self.ngenes
        # store the result
        self.thetas = np.zeros((self.ntimes, self.ngenes, self.ngenes))

    @staticmethod
    def neg_lkl_loss(thetas, S):

        t1 = -1 * torch.logdet(thetas)
        t2 = torch.stack([torch.trace(mat) for mat in torch.bmm(S, thetas)])
        return t1 + t2

    # alpha = 2
    def train(self, max_iters=50, n_intervals=1, lamb=2.1e-4, alpha=1, rho=1.7, beta=0, theta_init_offset=0.1):
        n_batches = int(np.ceil(self.ntimes / self.batchsize))
        for batch in range(n_batches):
            print("start running batch " + str(batch))
            start_time = time.time()
            # select a minibatch
            start_idx = batch * self.batchsize
            if batch < n_batches - 1:
                end_idx = (batch + 1) * self.batchsize
                w_empir_cov = self.w_empir_cov[start_idx:end_idx, :, :].to(device)
            else:
                w_empir_cov = self.w_empir_cov[start_idx:, :, :].to(device)



            Z = torch.diag_embed(1 / (torch.diagonal(w_empir_cov, offset=0, dim1=-2, dim2=-1) + theta_init_offset))

            ll = torch.cholesky(Z)
            Z = torch.matmul(ll, ll.transpose(-1, -2))
            U = torch.zeros(Z.shape).to(device)
            I = torch.eye(self.ngenes).expand(Z.shape).to(device)

            it = 0

            if rho is None:
                updating_rho = True
                # rho of the shape (ntimes, 1, 1)
                b_rho = torch.ones((Z.shape[0], 1, 1)).to(device) * 1.7
            else:
                b_rho = torch.FloatTensor([rho] * Z.shape[0])[:, None, None].to(device)
                updating_rho = False
            empir_cov_all = w_empir_cov / b_rho
            W_ij = torch.zeros((w_empir_cov.shape[0], w_empir_cov.shape[0]), dtype=torch.int).to(device)
            for i in range(empir_cov_all.shape[0]):
                for j in range(i + 1, empir_cov_all.shape[0]):
                    cov1 = empir_cov_all[i, :, :]
                    cov2 = empir_cov_all[j, :, :]
                    if torch.sum(abs(cov1 - cov2)) / (self.ngenes * self.ngenes) < 0.04:
                        W_ij[i, j] = 1
                        W_ij[j, i] = 1

            b_alpha = alpha
            b_beta = beta
            b_lamb = lamb
            ################# kernel idea############
            while (it < max_iters):
                # Primal
                Y = U - Z + w_empir_cov / b_rho  # (ntimes, ngenes, ngenes)
                thetas = - 0.5 * Y + torch.stack(
                    [torch_sqrtm(mat) for mat in (torch.transpose(Y, 1, 2) @ Y * 0.25 + I / b_rho)])
                Z_pre = Z.detach().clone()
                # over-relaxation
                thetas = b_alpha * thetas + (1 - b_alpha) * Z_pre
                A = torch.stack([torch.sum(weight[:, None, None] * thetas, axis=0) for weight in W_ij], dim=0)
                B = torch.stack([torch.sum(weight[:, None, None], axis=0) for weight in W_ij], dim=0)
                Z = torch.sign(thetas + U) * torch.max(
                    (b_rho * (thetas + U).abs()  + b_beta * A - b_lamb) / (b_rho + b_beta * B),
                    torch.Tensor([0]).to(device))  # mask

                # Dual
                U = U + thetas - Z

                # calculate residual

                primal_residual = torch.sqrt((thetas - Z).pow(2).sum(1).sum(1))
                dual_residual = b_rho.squeeze() * torch.sqrt((Z - Z_pre).pow(2).sum(1).sum(1))

                # updating rho, rho should be of shape (ntimes, 1, 1)
                if updating_rho:
                    mask_inc = (primal_residual > 10 * dual_residual)
                    b_rho[mask_inc, :, :] = b_rho[mask_inc, :, :] * 2
                    mask_dec = (dual_residual > 10 * primal_residual)
                    b_rho[mask_dec, :, :] = b_rho[mask_dec, :, :] / 2

                # free-up memory
                del Z_pre

                # Stopping criteria
                if (it + 1) % n_intervals == 0:


                    # simplify min of all duality gap
                    duality_gap = b_rho.squeeze() * torch.stack(
                        [torch.trace(mat) for mat in torch.bmm(U.permute(0, 2, 1), Z - thetas)])
                    duality_gap = duality_gap.abs()
                    print(
                        "n_iter: {}, duality gap: {:.4e}, primal residual: {:.4e}, dual residual: {:4e}".format(it + 1,
                                                                                                                duality_gap.max().item(),
                                                                                                                primal_residual.max().item(),
                                                                                                                dual_residual.max().item()))


                    primal_eps = 1e-3  # 1e-6
                    dual_eps = 1e-3  # 1e-6
                    if (primal_residual.max() < primal_eps) and (dual_residual.max() < dual_eps):
                        break
                it += 1

            loss1 = self.neg_lkl_loss(Z, w_empir_cov).sum()
            loss2 = Z.abs().sum()
            print("Batch loss: loss1: {:.5f}, loss2: {:.5f}".format(loss1.item(), loss2.item()))
            # store values
            if batch < n_batches - 1:
                self.thetas[start_idx:end_idx] = Z.detach().cpu().numpy()
            else:
                self.thetas[start_idx:] = Z.detach().cpu().numpy()
            del thetas, U, I, Y, ll, Z

            print("Finished running batch {:d}, running time: {:.2f} sec".format(batch, time.time() - start_time))

        return self.thetas

