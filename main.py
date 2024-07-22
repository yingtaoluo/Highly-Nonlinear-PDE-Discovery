import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from scipy.spatial import distance
import time
import torch.utils.data as data
from pyDOE import lhs
import argparse
from tqdm import tqdm
import copy
from numpy.linalg import norm as norm
import pdb
import os

lib_descr = []

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ours', help='choose which model to train')
parser.add_argument('--seed', type=int, default=42, help='choose which seed to load')
parser.add_argument('--gpu', type=int, default=-1, help='choose which gpu device to use')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--batch_size', type=int, help='choose a batch size')
aux_args.add_argument('--hidden_size', type=int, help='choose a hidden size')
aux_args.add_argument('--reg', type=float, help='choose a regularization coefficient')
parser.set_defaults(batch_size=24, hidden_size=20, reg=1e-5)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

GPU = args.gpu
gamma = 10
win = 10
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
print(device)


def Ridge(A, b, lam=0):
    # A (h, g), b (h)
    if lam != 0:
        return np.linalg.solve(A.T.dot(A) + lam * np.eye(A.shape[1]), A.T.dot(b))
    else:
        return np.linalg.lstsq(A, b)[0]


def plot(correct_ix, x, y, true_coefs, upd_coefs, rhs, do=False):
    return None if not do else None

    # set proper axis lim
    MAX, MIN = [], []

    MIN.append(get_min(true_coefs, upd_coefs, correct_ix[0]))
    MIN.append(get_min(true_coefs, upd_coefs, correct_ix[1]))
    MIN.append(get_min(true_coefs, upd_coefs, correct_ix[2]))
    MIN.append(get_min(true_coefs, upd_coefs, correct_ix[3]))

    MAX.append(get_max(true_coefs, upd_coefs, correct_ix[0]))
    MAX.append(get_max(true_coefs, upd_coefs, correct_ix[1]))
    MAX.append(get_max(true_coefs, upd_coefs, correct_ix[2]))
    MAX.append(get_max(true_coefs, upd_coefs, correct_ix[3]))

    for ix, ii in enumerate(correct_ix):
        # print(rhs[ii])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim(MIN[ix], MAX[ix])
        X, Y = np.meshgrid(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.plot_surface(X, Y, true_coefs[:, :, ii], rstride=1, cstride=1, cmap='rainbow')
        plt.savefig('./figure/kopde/Correct_Coefficient_' + rhs[ii] + '.png', dpi=600)
        # plt.show()
        plt.close()

    for ix, ii in enumerate(correct_ix):
        # print(rhs[ii])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim(MIN[ix], MAX[ix])
        X, Y = np.meshgrid(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.plot_surface(X, Y, upd_coefs[:, :, ii], rstride=1, cstride=1, cmap='rainbow')
        plt.savefig('./figure/kopde/Update_Coefficient_' + rhs[ii] + '.png', dpi=600)
        # plt.show()
        plt.close()

    for ix, ii in enumerate(correct_ix):
        # print(rhs[ii])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim(MIN[ix], MAX[ix])
        X, Y = np.meshgrid(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.plot_surface(X, Y, true_coefs[:, :, ii] - upd_coefs[:, :, ii],
                        rstride=1, cstride=1, cmap='rainbow')
        plt.savefig('./figure/kopde/Res_Coefficient_' + rhs[ii] + '.png', dpi=600)
        # plt.show()
        plt.close()


# JUST FOR plotting
def get_min(cor, upd, ix):
    I_MIN = np.min(cor[:, :, ix])
    II_MIN = np.min(upd[:, :, ix])
    III_MIN = np.min(cor[:, :, ix] - upd[:, :, ix])
    flag = np.minimum(I_MIN, II_MIN)
    return np.minimum(flag, III_MIN)


def get_max(cor, upd, ix):
    I_MAX = np.max(cor[:, :, ix])
    II_MAX = np.max(upd[:, :, ix])
    III_MAX = np.max(cor[:, :, ix] - upd[:, :, ix])
    flag = np.maximum(I_MAX, II_MAX)
    return np.maximum(flag, III_MAX)


def to_npy(x):
    return x.cpu().data.numpy() if torch.cuda.is_available() else x.detach().numpy()


def FiniteDiff(u, dx, d):
    n = u.size
    ux = np.zeros(n, dtype=u.dtype)

    if d == 1:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

        ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
        ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
        return ux

    if d == 2:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2

        ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
        ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
        return ux

    if d == 3:
        for i in range(2, n - 2):
            ux[i] = (u[i + 2] / 2 - u[i + 1] + u[i - 1] - u[i - 2] / 2) / dx ** 3

        ux[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / dx ** 3
        ux[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / dx ** 3
        ux[n - 1] = (2.5 * u[n - 1] - 9 * u[n - 2] + 12 * u[n - 3] - 7 * u[n - 4] + 1.5 * u[n - 5]) / dx ** 3
        ux[n - 2] = (2.5 * u[n - 2] - 9 * u[n - 3] + 12 * u[n - 4] - 7 * u[n - 5] + 1.5 * u[n - 6]) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u, dx, 3), dx, d - 3)


class ModelArch(torch.nn.Module):
    def __init__(self, layers):
        super(ModelArch, self).__init__()
        models = []
        for idx in range(1, len(layers) - 1):
            models.append(nn.Linear(layers[idx - 1], layers[idx], bias=True))
            models.append(nn.Tanh())
        models.append(nn.Linear(layers[len(layers) - 2], layers[len(layers) - 1], bias=True))
        self.model = nn.Sequential(*models)

    def forward(self, inputs):
        return self.model(inputs)


class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, X, data_train, X_f, X_val, data_val, lb, ub, lr, reg, model, records, lambda_w):
        super(PhysicsInformedNN, self).__init__()

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.model = model.to(device)

        # Parameters  (1, 50, 50, 12, 1)
        self.lambda_w = torch.nn.Parameter(lambda_w.to(device), requires_grad=False)

        # Training data
        self.x = torch.tensor(X[..., 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(X[..., 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(X[..., 2:3], dtype=torch.float32, requires_grad=True).to(device)
        self.data = torch.tensor(data_train, dtype=torch.float32).to(device)

        # Collocation points
        self.x_f = torch.tensor(X_f[..., 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_f = torch.tensor(X_f[..., 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[..., 2:3], dtype=torch.float32, requires_grad=True).to(device)

        # Validation data
        self.x_val = torch.tensor(X_val[..., 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_val = torch.tensor(X_val[..., 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_val = torch.tensor(X_val[..., 2:3], dtype=torch.float32, requires_grad=True).to(device)
        self.data_val = torch.tensor(data_val, dtype=torch.float32).to(device)

        # Optimizers
        self.parameter_list = list(self.model.parameters()) + [self.lambda_w]
        self.NonZeroMask_w_tf = torch.ones((total_terms, 1)).to(device)
        self.threshold = records['adam_pre_valid_loss'][-1][0] if len(records['adam_pre_valid_loss']) > 0 else 1e8
        self.learning_rate = lr
        self.loss_f_ratio = 1
        self.L1_coef = reg
        self.criterion = nn.MSELoss(reduction='mean')
        # Specify the parameters you want to optimize
        self.optimizer_adam = torch.optim.Adam(self.parameter_list, lr=self.learning_rate)

        # Recording
        self.records = records

    def net_U(self, x, y, t, bs):
        H = torch.cat([x, y, t], -1)
        H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0  # (50, 50, 2, 3)
        Y = self.model(H.view(-1, 3)).reshape(n, m, bs, num_targets)
        return Y

    def net_f(self, x, y, t, bs):
        predict_data = self.net_U(x, y, t, bs)
        uvw = [predict_data[..., i:i + 1] for i in range(num_targets)]

        derivatives, derivatives_description = [], []
        for i in range(num_targets):
            w_x = torch.autograd.grad(uvw[i], x, grad_outputs=torch.ones_like(uvw[i]), create_graph=True)[0]
            w_y = torch.autograd.grad(uvw[i], y, grad_outputs=torch.ones_like(uvw[i]), create_graph=True)[0]
            w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
            w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
            w_xy = torch.autograd.grad(w_x, y, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]

            if i == 0:
                derivatives.append(torch.ones_like(w_x))
                derivatives_description.append('')

            derivatives.extend([w_x, w_y, w_xx, w_yy, w_xy])
            derivatives_description.extend(['V{}_x'.format(i), 'V{}_y'.format(i), 'V{}_xx'.format(i),
                                            'V{}_yy'.format(i), 'V{}_xy'.format(i)])

        lib_fun, self.lib_descr = self.build_library(uvw, derivatives, derivatives_description, PolyOrder=1,
                                                     data_description=['V{}'.format(i) for i in range(num_targets)])

        w_t, f_w = [], []
        Phi = torch.cat(lib_fun, -1)
        for i in range(num_targets):
            time_deriv = torch.autograd.grad(uvw[i], t, grad_outputs=torch.ones_like(uvw[i]), create_graph=True)[0]
            w_t.append(time_deriv)
            Xi = self.spatial_kernel(self.lambda_w[i])
            f_w.append(time_deriv - torch.matmul(Phi, (Xi * self.NonZeroMask_w_tf)))

        w_t = torch.stack(w_t)
        f_w = torch.stack(f_w)

        return f_w, Phi, w_t

    def spatial_kernel(self, est_coefs):
        def kernel_smoothing(grid):
            nw, mw, g, _ = grid.shape  # window shape
            nc, mc = int(nw / 2), int(mw / 2)  # must be even
            update_coefs = torch.zeros((g, 1))
            kernel = np.zeros((nw, mw))

            for k in range(g):
                for i in range(nw):
                    for j in range(mw):
                        dis_sq = (i - nc) ** 2 + (j - mc) ** 2
                        kernel[i, j] = np.exp(-gamma * dis_sq)
                        update_coefs[k, 0] += kernel[i, j] * grid[i, j, k, 0]
                update_coefs[k, 0] /= np.sum(kernel)
            return update_coefs

        updated_coefs = torch.zeros(est_coefs.shape)
        for i in range(n - win + 1):
            for j in range(m - win + 1):
                updated_coefs[i + int(win / 2), j + int(win / 2)] = \
                    kernel_smoothing(est_coefs[i:i + win, j:j + win])

        return updated_coefs

    def u_plot(self):
        H = torch.cat([self.x, self.y, self.t], -1)
        H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0  # (50, 50, 24, 3)
        Y = self.model(H.view(-1, 3))  # .reshape(n, m, -1, num_targets)
        L = self.data.view(-1, 1)

        xlim = np.linspace(0, len(L) - 1, len(L))
        plt.title('Known Equation Error')
        plt.plot(xlim, to_npy(Y), color='blue', label='u prediction')
        plt.plot(xlim, to_npy(L), color='red', label='u ground truth')
        plt.legend(loc=1)
        plt.savefig('u_plot.png')
        # plt.show()
        plt.close()
        pdb.set_trace()

    def build_library(self, data, derivatives, derivatives_description, PolyOrder=2, data_description=None):
        ## polynomial terms
        P = PolyOrder
        lib_poly = [torch.ones_like(data[0])]
        lib_poly_descr = ['']  # it denotes '1'
        for i in range(len(data)):  # polynomial terms of univariable
            for j in range(1, P + 1):
                lib_poly.append(data[i] ** j)
                if j == 1:
                    lib_poly_descr.append(data_description[i])
                else:
                    lib_poly_descr.append(data_description[i] + "**" + str(j))

        # lib_poly.append(data[0] * data[1])
        # lib_poly_descr.append(data_description[0] + data_description[1])
        # lib_poly.append(data[0] * data[2])
        # lib_poly_descr.append(data_description[0] + data_description[2])
        # lib_poly.append(data[1] * data[2])
        # lib_poly_descr.append(data_description[1] + data_description[2])

        ## derivative terms
        lib_deri = derivatives
        lib_deri_descr = derivatives_description

        ## Multiplication of derivatives and polynomials (including the multiplication with '1')
        lib_poly_deri = []
        lib_poly_deri_descr = []
        for i in range(len(lib_poly)):
            for j in range(len(lib_deri)):
                lib_poly_deri.append(lib_poly[i] * lib_deri[j])
                lib_poly_deri_descr.append(lib_poly_descr[i] + lib_deri_descr[j])

        return lib_poly_deri, lib_poly_deri_descr

    def Adam_Training(self, num_epochs, batch_size=args.batch_size, f_batch_size=args.batch_size):
        train_record, valid_record = [], []

        # Create list of shuffled indices
        indices = torch.randperm(self.x.size(2))
        f_indices = torch.randperm(self.x_f.size(2))

        # Divide data into batches
        x_batches = [self.x[:, :, indices[i * batch_size:(i + 1) * batch_size]] for i in
                     range(int(np.ceil(len(indices) / batch_size)))]
        y_batches = [self.y[:, :, indices[i * batch_size:(i + 1) * batch_size]] for i in
                     range(int(np.ceil(len(indices) / batch_size)))]
        t_batches = [self.t[:, :, indices[i * batch_size:(i + 1) * batch_size]] for i in
                     range(int(np.ceil(len(indices) / batch_size)))]
        data_batches = [self.data[:, :, :, indices[i * batch_size:(i + 1) * batch_size]] for i in
                        range(int(np.ceil(len(indices) / batch_size)))]

        x_f_batches = [self.x_f[:, :, f_indices[i * batch_size:(i + 1) * batch_size]] for i in
                       range(int(np.ceil(len(f_indices) / batch_size)))]
        y_f_batches = [self.y_f[:, :, f_indices[i * batch_size:(i + 1) * batch_size]] for i in
                       range(int(np.ceil(len(f_indices) / batch_size)))]
        t_f_batches = [self.t_f[:, :, f_indices[i * batch_size:(i + 1) * batch_size]] for i in
                       range(int(np.ceil(len(f_indices) / batch_size)))]

        for it in tqdm(range(num_epochs)):
            # print('Epoch {}.'.format(it + 1))
            self.model.train()

            for b in range(len(x_batches)):
                x_batch = x_batches[b]  # (50, 50, 2, 1)
                y_batch = y_batches[b]
                t_batch = t_batches[b]
                data_batch = data_batches[b]

                x_f_batch = x_f_batches[b % len(x_f_batches)]
                y_f_batch = y_f_batches[b % len(y_f_batches)]
                t_f_batch = t_f_batches[b % len(t_f_batches)]

                predict_data = self.net_U(x_batch, y_batch, t_batch, batch_size)
                f_w_pred, _, _ = self.net_f(x_f_batch, y_f_batch, t_f_batch, batch_size)

                loss_unit, loss_f_w, loss_lambda_w = [], [], []
                for i in range(num_targets):
                    loss_unit.append(self.criterion(data_batch[i], predict_data[..., i:i + 1]))
                    loss_f_w.append(self.loss_f_ratio * torch.mean(torch.square(f_w_pred[i])))
                    loss_lambda_w.append(self.L1_coef * torch.norm(self.lambda_w[i], p=1))

                losses = sum(loss_unit) + sum(loss_f_w) + sum(loss_lambda_w)
                loss = losses  # torch.log(losses)

                loss.backward(retain_graph=True)
                self.optimizer_adam.step()
                self.optimizer_adam.zero_grad()

                loss_unit_list = [to_npy(loss_unit[j]).item() for j in range(num_targets)]
                loss_f_w_list = [to_npy(loss_f_w[j]).item() for j in range(num_targets)]
                loss_lambda_w = [to_npy(loss_lambda_w[j]).item() for j in range(num_targets)]
                train_record.append([to_npy(losses).item()] + loss_unit_list + loss_f_w_list + loss_lambda_w)

                print('The training loss: {}, data loss: {}, equation loss:{}, regularization loss:{}.'.
                      format(to_npy(losses).item(), loss_unit_list, loss_f_w_list, loss_lambda_w))

            # validation
            self.model.eval()

            predict_val = self.net_U(self.x_val, self.y_val, self.t_val, N_valid)

            loss_unit_val = []
            for i in range(num_targets):
                loss_unit_val.append(self.criterion(self.data_val[i], predict_val[..., i:i + 1]))

            losses_val = sum(loss_unit_val)
            # loss = torch.log(losses)

            loss_unit_val_list = [to_npy(loss_unit_val[j]).item() for j in range(num_targets)]
            valid_record.append([to_npy(losses_val).item()] + loss_unit_val_list)

            print('The validation loss: {}, data loss: {}.'.
                  format(to_npy(losses_val).item(), loss_unit_val_list))

            # print(self.lib_descr, self.lambda_w.reshape(n, m, total_terms))

            if self.threshold > losses_val:
                self.threshold = losses_val
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'lambda_w': self.lambda_w,
                            'records': self.records},
                           'checkpoints/{}.pth'.format(checkpoint_name))

            if it % 100 == 0:
                lambda_w_pred = to_npy(self.lambda_w).reshape(-1, total_terms)
                lambda_w_true = correct_coefs.reshape(-1, total_terms)
                cosine_similarity_w = 1 - distance.cosine(lambda_w_true.reshape(-1), lambda_w_pred.reshape(-1))
                print('Cosine similarity of lambda_w: %.2f \n' % (cosine_similarity_w))
                print('w predict mean:{}, real mean:{} \n'.format(abs(lambda_w_pred).mean(), abs(lambda_w_true).mean()))
                error_lambda_w = np.linalg.norm(lambda_w_true - lambda_w_pred, 2) / np.linalg.norm(lambda_w_true, 2)
                print('lambda_w Error: %.2f \n' % (error_lambda_w))
                nonzero_ind_w = np.nonzero(lambda_w_true)
                lambda_w_error_vector = np.absolute(
                    (lambda_w_true[nonzero_ind_w] - lambda_w_pred[nonzero_ind_w]) / lambda_w_true[nonzero_ind_w])
                error_lambda_w_mean = np.mean(lambda_w_error_vector)
                error_lambda_w_std = np.std(lambda_w_error_vector)
                print('lambda_w Mean Error: %.4f \n' % (error_lambda_w_mean))
                print('lambda_w Std Error: %.4f \n' % (error_lambda_w_std))

                disc_eq_temp = []
                for i_lib in range(len(self.lib_descr)):
                    if lambda_w_pred[..., i_lib].sum() != 0:
                        disc_eq_temp.append(str(lambda_w_pred[..., i_lib].mean()) + self.lib_descr[i_lib])
                disc_eq = '+'.join(disc_eq_temp)
                print('The discovered equation: w_t = ' + disc_eq + '\n')

                ######################## Plots for lambdas #################
                fig = plt.figure()
                plt.plot(lambda_w_pred, 'ro-', label='pred')
                plt.plot(lambda_w_true, label='true')
                plt.legend()
                plt.title('lambda_w')
                plt.savefig('19.png')
                plt.close(fig)

        return train_record, valid_record

    def Filter_Terms(self, num_epochs):
        self.loss_f_ratio = 2.
        self.L1_coef = 0.
        self.optimizer_adam = torch.optim.Adam(self.parameter_list, lr=1e-4)

        print('STRidge starts')
        _, Phi, w_t_pred = self.net_f(self.x_f, self.y_f, self.t_f, args.batch_size)
        # lambda_w2 = np.zeros(shape=(num_targets, n, m, total_terms, 1))

        for k in range(num_targets):
            W_best, MSE_best, AIC_best = self.One_Iter_Out(Phi.detach().cpu().numpy(), w_t_pred[k].detach().cpu().numpy())
            self.lambda_w[k][..., 0] = torch.tensor(W_best)

        # self.lambda_w = torch.nn.Parameter(lambda_w2.to(device), requires_grad=True)

        lambda_w = self.lambda_w.detach().cpu().numpy()
        NonZeroInd_w = np.nonzero(lambda_w)
        NonZeroMask_w = np.zeros_like(lambda_w)
        NonZeroMask_w[NonZeroInd_w] = 1
        self.NonZeroMask_w_tf = torch.tensor(NonZeroMask_w).to(device)

    def joint_training(self, adam_epo=5000, in_epo=1000, pt_adam_epo=5000):
        # Adam optimizer pre-training
        self.records['adam_pre_train_loss'], self.records['adam_pre_valid_loss'] = self.Adam_Training(adam_epo)

        self.Filter_Terms(in_epo)

        # Adam optimizer post-training
        self.records['adam_post_train_loss'], self.records['adam_post_valid_loss'] = self.Adam_Training(pt_adam_epo)

    def predict(self, X_star):
        x_star = torch.tensor(X_star[:, 0:1], requires_grad=True).to(device)
        y_star = torch.tensor(X_star[:, 1:2], requires_grad=True).to(device)
        t_star = torch.tensor(X_star[:, 2:3], requires_grad=True).to(device)

        u_star = self.u_pred(x_star, y_star, t_star)
        v_star = self.v_pred(x_star, y_star, t_star)
        w_star = self.w_pred(x_star, y_star, t_star)

        return u_star.cpu().data.numpy(), v_star.cpu().data.numpy(), w_star.cpu().data.numpy()

    def One_Iter_Out(self, theta, ut, normalize=False, strict_num_term=True):
        ut = ut[:, :, :, 0]
        # Xs--theta (n,m,h,g), ys--ut (n,m,h)
        n1, m1, h1, g1 = theta.shape
        rhs = np.array(self.lib_descr)
        iter_record = []

        true_coefs = copy.deepcopy(correct_coefs)[0, ..., 0]

        # Normalize
        if normalize:
            c_norms = np.linalg.norm(theta.reshape((n1 * m1 * h1, g1)), axis=0)
            c_norms = np.repeat(c_norms[np.newaxis, :], h1, axis=0)
            c_norms = np.repeat(c_norms[np.newaxis, :, :], m1, axis=0)
            c_norms = np.repeat(c_norms[np.newaxis, :, :, :], n1, axis=0)
            theta /= c_norms
            true_coefs *= c_norms[:, :, 0, :]

        ''' with assumed terms'''
        biginds = correct_ix
        W, mse_upd, record = self.report_kstr(theta, ut, biginds, true_coefs)
        [[mse_est, mse_real, diff_upd]] = record
        print('mse_est:{}, mse_real:{}'.format(mse_est.mean(), mse_real.mean()))
        print('diff_est:{}'.format(diff_upd.mean()))
        # # coefs_norm = np.linalg.norm(W, axis=0)
        # coefs_norm = np.mean(W, 0)
        # print('coefs_norm:{}'.format(coefs_norm), file=f)

        print(rhs[biginds])
        AIC_best = len(biginds) + np.log(mse_upd).mean()
        print('AIC:{}'.format(AIC_best))
        iter_record.append([record, biginds, W, mse_upd, AIC_best])
        plot(correct_ix, x, y, true_coefs, W, rhs)

        '''all from scratch'''
        biginds = [i for i in range(g1)]
        smallinds = []
        W, mse_upd, record = self.report_kstr(theta, ut, biginds, true_coefs)
        [[mse_est, mse_real, diff_upd]] = record
        print('mse_est:{}, mse_real:{}'.format(mse_est.mean(), mse_real.mean()))
        print('diff_est:{}'.format(diff_upd.mean()))
        print(rhs[biginds])
        # coefs_norm = np.linalg.norm((np.repeat(W[:, :, np.newaxis, :], h, axis=2) * theta[win:-win, win:-win]).
        #                             reshape(((n - 2 * win) * (m - 2 * win) * h, g)), axis=0)
        # print('coefs_norm:{}'.format(coefs_norm), file=f)

        plot(correct_ix, x, y, true_coefs, W, rhs)

        W_best = np.zeros(W.shape, dtype=np.float64)  # (n, g)
        AIC_best = len(biginds) + np.log(mse_upd).mean()
        print('AIC:{}'.format(AIC_best))
        iter_record.append([record, biginds, W, mse_upd, AIC_best])
        MSE_best = mse_upd
        ix = 0

        while len(biginds) > num_terms:
            ix += 1
            Ws, mse_upds, aics = [], [], []

            for j in range(len(biginds)):
                # try on copied data
                biginds_trial = copy.deepcopy(biginds)
                smallinds_trial = copy.deepcopy(smallinds)

                smallest = biginds_trial[j]
                biginds_trial.remove(smallest)
                smallinds_trial.append(smallest)

                for i in smallinds_trial:
                    W[:, :, i] = np.zeros((n1, m1))

                W, mse_upd, record = self.report_kstr(theta, ut, biginds_trial, true_coefs)
                Ws.append(np.array(W, ))  # it has to be np.array, otherwise there will be random 0 replacing elements
                mse_upds.append(mse_upd)

                aic = len(biginds_trial) + np.log(mse_upd).mean()
                aics.append(aic)

            trial_opt_ix = np.argmin(aics)
            smallest = biginds[trial_opt_ix]
            W = Ws[trial_opt_ix]
            mse_upd = mse_upds[trial_opt_ix]
            aic = aics[trial_opt_ix]
            biginds.remove(smallest)
            smallinds.append(smallest)
            [[mse_est, mse_real, diff_upd]] = record
            print('mse_est:{}, mse_real:{}'.format(mse_est.mean(), mse_real.mean()))
            print('diff_est:{}'.format(diff_upd.mean()))
            iter_record.append([record, biginds, W, mse_upd, aic])

            # plot(correct_ix, x, y, win, true_coefs, W, rhs)

            # print('smallinds:{}'.format(smallinds), file=f)
            print(rhs[biginds])
            # coefs_norm = np.mean(W, 0)
            # print('coefs_norm:{}'.format(coefs_norm), file=f)
            print('AIC:{}'.format(aic))

            if aic < AIC_best or strict_num_term:
                AIC_best = aic
                W_best = W
                MSE_best = mse_upd

        # print the discovered PDE structure
        print('The discovered dynamical system:{}'.format(rhs[biginds]))

        plot(correct_ix, x, y, true_coefs, W, rhs)

        return W_best, MSE_best, AIC_best

    def compute_kstr(self, Xs, ut, normalize=False):
        normalize = False
        # theta (n,m,h,g), ut (n,m,h)
        theta = copy.deepcopy(Xs)  # avoid overwrite
        n, m, h, g = theta.shape

        est_coefs = np.zeros((n, m, g))  # (n, m, g)

        # store denom for recovery
        denom = np.ones((n, m, g))

        if normalize:
            for i in range(n):
                for j in range(m):
                    local_denom = norm(theta[i, j], axis=0)
                    if np.any(local_denom == 0):
                        ix = np.argwhere(local_denom == 0)
                        local_denom[ix] = 1  # avoid /0
                    denom[i, j] = local_denom
                    local_denom_2d = np.repeat(local_denom[np.newaxis, :], h, axis=0)
                    theta[i, j] /= local_denom_2d

        for i in range(n):
            for j in range(m):
                est_coefs[i, j] = Ridge(theta[i, j], ut[i, j])

        # rescale the coefficients to match k
        if normalize:
            est_back_coefs = est_coefs / denom
        else:
            est_back_coefs = est_coefs

        return est_back_coefs  # (n, m, g)

    def report_kstr(self, theta, ut, biginds, true_coefs, nr=False):
        divide1 = int(0.5 * theta.shape[2])
        divide2 = int(0.75 * theta.shape[2])
        n1, m1, h1, g1 = theta.shape

        upd_coefs = np.zeros(true_coefs.shape, np.float64)
        upd_coefs[:, :, biginds] = self.compute_kstr(theta[:, :, 0:divide1, biginds], ut[:, :, 0:divide1], nr)
        # upd_coefs[:, :, correct_ix] = compute_kstr(theta[:, :, :, correct_ix], ut, nr)

        # compare mse
        upd_coefs_ed = np.repeat(upd_coefs[:, :, np.newaxis, :], divide1, axis=2)
        cor_coefs_ed = np.repeat(true_coefs[:, :, np.newaxis, :], divide1, axis=2)
        mse_upd_train = abs(ut[:, :, 0:divide1] - np.sum(upd_coefs_ed * theta[:, :, 0:divide1], axis=3))
        mse_real_train = abs(ut[:, :, 0:divide1] - np.sum(cor_coefs_ed * theta[:, :, 0:divide1], axis=3))
        print('mae_est_train:{}, mae_real_train:{}'.
              format(mse_upd_train.mean(), mse_real_train.mean()))  # , file=f)

        upd_coefs_ed = np.repeat(upd_coefs[:, :, np.newaxis, :], divide2 - divide1, axis=2)
        cor_coefs_ed = np.repeat(true_coefs[:, :, np.newaxis, :], divide2 - divide1, axis=2)
        mse_upd_dev = abs(ut[:, :, divide1:divide2] - np.sum(upd_coefs_ed * theta[:, :, divide1:divide2], axis=3))
        mse_real_dev = abs(ut[:, :, divide1:divide2] - np.sum(cor_coefs_ed * theta[:, :, divide1:divide2], axis=3))
        print('mae_est_dev:{}, mae_real_dev:{}'.
              format(mse_upd_dev.mean(), mse_real_dev.mean()))  # , file=f)

        upd_coefs_ed = np.repeat(upd_coefs[:, :, np.newaxis, :], h1 - divide2, axis=2)
        cor_coefs_ed = np.repeat(true_coefs[:, :, np.newaxis, :], h1 - divide2, axis=2)
        mse_upd_test = abs(ut[:, :, divide2:] - np.sum(upd_coefs_ed * theta[:, :, divide2:], axis=3))
        mse_real_test = abs(ut[:, :, divide2:] - np.sum(cor_coefs_ed * theta[:, :, divide2:], axis=3))
        print('mae_est_test:{}, mae_real_test:{}'.
              format(mse_upd_test.mean(), mse_real_test.mean()))  # , file=f)

        diff_upd = abs(upd_coefs - true_coefs)[:, :, correct_ix]
        print('diff_upd_test:{}'.format(diff_upd.mean()))
        print(len(biginds))

        return upd_coefs, mse_upd_test, [[mse_upd_test, mse_real_test, diff_upd]]


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # load model arch
    num_targets = 1
    total_terms = 12
    num_terms = 4
    layers = [3] + 8 * [args.hidden_size] + [num_targets]
    apriori = [3, 4]

    num = '5'
    num_var = 1
    correct_ix = [1, 2, 3, 4]

    s = 1  # discard a few boundary values
    ex = 2  # discard unstable first few steps
    # Set size of grid
    n = 51 - s  # x=50
    m = 51 - s  # y=50
    steps = 50 - ex  # t=48

    # Set up grid
    x = np.arange(1, 52 - s, 1)
    x = x * 20
    y = np.arange(1, 52 - s, 1)
    y = y * 20
    dx = 20
    dy = 20
    t = np.linspace(0.2, 10, 50)[:-ex]
    dt = 0.2
    # these can all be scaled back once the PDE is discovered
    # dx = dy = dt = 1  # for standardization

    # Initial condition
    # [time, channel, spacex, spacey]
    u = np.reshape(np.fromfile('data/h_' + num + '.bin'), (50, 51, 51))[np.newaxis, ex:, s:, s:]  # (1, 48, 50, 50)
    u_data = np.array(u, dtype=np.float32)
    u_data = np.transpose(u_data, (0, 2, 3, 1))  # (1, 50, 50, 48), i.e. (x, y, t)
    k = np.reshape(np.fromfile('data/k_' + num + '.bin'), (51, 51))[s:, s:]  # (50, 50)
    k_coef = np.array(k, dtype=np.float32)
    # here k is only a function w.r.t. x and y, not t.
    k = np.expand_dims(k, 2).repeat(steps, axis=2)  # (50, 50, 48)

    # coefficient field processing
    kx = np.zeros((n, m, steps), dtype=u.dtype)
    for i in range(m):
        for j in range(steps):
            kx[:, i, j] = FiniteDiff(k[:, i, j], dx, 1)
    kx = np.reshape(kx, (n * m * steps, 1)) * 1e4

    ky = np.zeros((n, m, steps), dtype=u.dtype)
    for i in range(n):
        for j in range(steps):
            ky[i, :, j] = FiniteDiff(k[i, :, j], dy, 1)
    ky = np.reshape(ky, (n * m * steps, 1)) * 1e4

    k = np.reshape(k, (n * m * steps, 1)) * 1e4

    kx = np.reshape(kx, (num_targets, n, m, steps))
    ky = np.reshape(ky, (num_targets, n, m, steps))
    k1 = np.reshape(k, (num_targets, n, m, steps))
    k2 = np.reshape(k, (num_targets, n, m, steps))

    correct_coefs = np.zeros((num_targets, n, m, total_terms), np.float32)
    correct_coefs[..., correct_ix[0]] = kx[..., 0]
    correct_coefs[..., correct_ix[1]] = k1[..., 0]
    correct_coefs[..., correct_ix[2]] = ky[..., 0]
    correct_coefs[..., correct_ix[3]] = k2[..., 0]  # (1, 50, 50, 12)
    correct_coefs = correct_coefs[..., np.newaxis]

    # data processing
    mean = np.mean(u_data)
    std = np.std(u_data)
    print(mean, std)
    u_normalized = (u_data - mean)  # / std

    # Preprocess data #1(First dimensiton is space and the second dimension is time.)
    st_data = u_normalized.reshape((num_var, n * m, steps))

    t_data = np.arange(steps).reshape((1, 1, -1)) * dt
    t_data = np.tile(t_data, (n, m, 1))

    # This part reset the coordinates
    x_data = np.arange(n).reshape((-1, 1, 1)) * dx
    x_data = np.tile(x_data, (1, m, steps))

    y_data = np.arange(m).reshape((1, -1, 1)) * dy
    y_data = np.tile(y_data, (n, 1, steps))

    # Preprocess data #2(compatible with NN format)
    data_star = np.reshape(st_data, (num_var, n, m, steps, 1))  # (1, 50, 50, 48, 1)

    X_star = np.stack((x_data, y_data, t_data), axis=-1)  # (50, 50, 48, 3)

    # Divide the dataset into training and validation/test sets
    N_valid_test = int(steps * 0.5)  # The last few timestamps for validation and test sets
    N_train = steps - N_valid_test  # The rest for training set

    # Training set
    X_train = X_star[:, :, :N_train]
    data_train = data_star[:, :, :, :N_train]

    # Validation and test sets
    X_valid_test = X_star[:, :, N_train:]
    data_valid_test = data_star[:, :, :, N_train:]

    # Randomly sample validation and test sets from the same set
    N_valid = int(X_valid_test.shape[2] * 0.5)
    idx_valid = np.random.choice(X_valid_test.shape[2], N_valid, replace=False)
    X_valid = X_valid_test[:, :, idx_valid]
    data_valid = data_valid_test[:, :, :, idx_valid]

    # Test set, which are the rest of validation/test set
    idx_test = np.setdiff1d(np.arange(X_valid_test.shape[2]), idx_valid, assume_unique=True)
    X_test = X_valid_test[:, :, idx_test]
    data_test = data_valid_test[:, :, :, idx_test]

    # # Bounds
    lb = np.min(X_star, (0, 1, 2))
    ub = np.max(X_star, (0, 1, 2))
    #
    # # Collocation points
    # N_f = n * m * N_train * 2
    # X_f = lb + (ub - lb) * lhs(3, N_f)
    # X_f = np.vstack((X_f, X_train.reshape(-1, 3)))  # (180000, 3)

    X_f = X_train  # (50, 50, 24, 3)

    # add noise
    noise = 0.05
    data_train = data_train + noise * np.std(data_train) * np.random.randn(num_var, n, m, N_train, 1)
    data_valid = data_valid + noise * np.std(data_valid) * np.random.randn(num_var, n, m, data_valid.shape[3], 1)
    data_test = data_test + noise * np.std(data_test) * np.random.randn(num_var, n, m, data_test.shape[3], 1)

    # =============================================================================
    # model training
    # =============================================================================
    model = ModelArch(layers)
    load = False
    checkpoint_name = 'kdd'
    if load:
        checkpoint = torch.load('checkpoints/{}.pth'.format(checkpoint_name))
        model.load_state_dict(checkpoint['state_dict'])
        lambda_w = checkpoint['lambda_w']
        records = checkpoint['records']
    else:
        records = {'adam_pre_train_loss': [], 'adam_pre_valid_loss': [],
                   'str_eq_progress': [], 'nonzero_mask': [],
                   'adam_post_train_loss': [], 'adam_post_valid_loss': []}
        lambda_w = torch.zeros([num_targets, n, m, total_terms, 1], dtype=torch.float32)

    Trainer = PhysicsInformedNN(X_train, data_train, X_f, X_valid, data_valid, lb, ub,
                                args.lr, args.reg, model, records, lambda_w)

    Trainer.joint_training()
    # Trainer.u_plot()
