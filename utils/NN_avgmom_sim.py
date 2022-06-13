# Neural Net Average Derivative Monte Carlo Simulations

# 0. Importing modules
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump
import scipy
import scipy.stats
import scipy.special
import torch
import torch.nn as nn
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from .riesznet import RieszNet

device = torch.cuda.current_device() if torch.cuda.is_available() else None

def _combinations(n_features, degree, interaction_only):
        comb = (combinations if interaction_only else combinations_w_r)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(0, degree + 1))

class Learner(nn.Module):

    def __init__(self, n_t, n_hidden, p, degree, interaction_only=False):
        super().__init__()
        n_common = 200
        self.monomials = list(_combinations(n_t, degree, interaction_only))
        self.common = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU())
        self.riesz_nn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))
        self.reg_nn0 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))


    def forward(self, x):
        poly = torch.cat([torch.prod(x[:, t], dim=1, keepdim=True)
                          for t in self.monomials], dim=1)
        feats = self.common(x)
        riesz = self.riesz_nn(feats) + self.riesz_poly(poly)
        reg = self.reg_nn0(feats) * (1 - x[:, [0]]) + self.reg_nn1(feats) * x[:, [0]] + self.reg_poly(poly)
        return torch.cat([reg, riesz], dim=1)

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def rmse_fn(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

methods = ['dr', 'direct', 'ips']

# 1. Estimation
def est_avgmom_NN(X, y, moment_fn, n_hidden, drop_prob, true_reg, true_rr, scale_y = True, fast_train_opt = {}, train_opt = {}):

    # Scale y
    if scale_y:
        scaler = StandardScaler().fit(y.reshape(-1, 1))
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
        scale, offset = scaler.scale_[0], scaler.mean_[0]
    else:
        y_scaled, scale, offset = y, 1.0, 0
    
    # Split in train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size = 0.2)

    # Train
    torch.cuda.empty_cache()
    learner = Learner(X_train.shape[1], n_hidden, drop_prob, 0)
    agmm = RieszNet(learner, moment_fn)
    # Fast training
    agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
            **fast_train_opt,
            model_dir=str(Path.home()), device=device, verbose=0)
    # Fine tune
    agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
            **train_opt,
            warm_start=True,
            model_dir=str(Path.home()), device=device, verbose=0)

    reg_hat, rr_hat = agmm.predict(X, model = 'earlystop')[:, 0].flatten(), agmm.predict(X, model = 'earlystop')[:, 1].flatten()

    rmse_reg = rmse_fn(reg_hat * scale + offset, true_reg(X))
    r2_reg = 1 - (rmse_reg ** 2) / np.var(true_reg(X))
    rmse_rr = rmse_fn(rr_hat, true_rr(X))
    r2_rr = 1 - (rmse_rr ** 2) / np.var(true_rr(X))
    ipsbias = np.mean((rr_hat - true_rr(X)) * true_reg(X))
    drbias = np.mean((rr_hat - true_rr(X)) * (true_reg(X) - reg_hat * scale + offset))

    # Return average moment and CI for all methods
    final_params = tuple(x * scale for method in methods
                         for x in agmm.predict_avg_moment(X, y_scaled, model = 'earlystop', method = method))
    nuisance_metrics = (rmse_reg, r2_reg, rmse_rr, r2_rr, ipsbias, drbias)
    return final_params + nuisance_metrics

# 2. Simulations
def get_est(W, *, moment_fn, n_hidden, drop_prob, true_reg, true_rr, gen_y, gen_T, sim = 1, scale_y = True, 
            fast_train_opt = {}, train_opt = {}, seed = 1234):

    np.random.seed(seed + sim)
    X = np.hstack((gen_T(W), W))
    y = gen_y(X)
    truth = np.mean(moment_fn(X, true_reg, device = None))

    return est_avgmom_NN(X, y, moment_fn, n_hidden, drop_prob, true_reg = true_reg, true_rr = true_rr, scale_y = scale_y, 
                         fast_train_opt = fast_train_opt, train_opt = train_opt) + (truth,)


def sim_fun(W, *, moment_fn, n_hidden, drop_prob, true_reg, true_rr, gen_y, gen_T, N_sim = 100, scale_y = True, 
            fast_train_opt = {}, train_opt = {}, seed = 1234, verbose = 1, save = '', plot = True, saveplot = ''):
    
    res = Parallel(n_jobs = -1, verbose = verbose)(delayed(get_est)(W, moment_fn = moment_fn, n_hidden = n_hidden, drop_prob = drop_prob, 
                                                                    true_reg = true_reg, true_rr = true_rr,
                                                                    gen_y = gen_y, gen_T = gen_T, sim = sim, scale_y = scale_y, 
                                                                    fast_train_opt = fast_train_opt, train_opt = train_opt,
                                                                    seed = seed) for sim in range(N_sim))

    res = tuple(np.array(x) for x in zip(*res))
    rmse_reg, r2_reg, rmse_rr, r2_rr, ipsbias, drbias, truth = res[-7:]
    res_dict = {}
    for it, method in enumerate(methods):
        point, lb, ub = res[it * 3: (it + 1)*3]
        res_dict[method] = {'point': point, 'lb': lb, 'ub': ub,
            'cov': np.mean(np.logical_and(truth >= lb, truth <= ub)),
            'bias': np.mean(point - truth),
            'rmse': rmse_fn(point, truth)
        }

    if save != '':
        to_save = [res_dict, rmse_reg, r2_reg, rmse_rr, r2_rr, ipsbias, drbias, truth]
        dump(to_save, save)

    if plot:
        #nuisance_str = ("reg RMSE: {:.3f}, R2: {:.3f}, rr RMSE: {:.3f}, R2: {:.3f}\n"
        #                "IPS orthogonality: {:.3f}, DR orthogonality: {:.3f}").format(np.mean(rmse_reg), np.mean(r2_reg),
        #                                                                       np.mean(rmse_rr), np.mean(r2_rr),
        #                                                                       np.mean(ipsbias), np.mean(drbias))
        method_strs = ["{}. Bias: {:.3f}, RMSE: {:.3f}, Coverage: {:.3f}".format(method, d['bias'], d['rmse'], d['cov'])
                       for method, d in res_dict.items()]
        #plt.title("\n".join([nuisance_str] + method_strs))
        plt.title("\n".join(method_strs))
        plt.axvline(x = np.mean(truth), label='true', color='red')
        for method, d in res_dict.items():
            plt.hist(np.array(d['point']), alpha=.5, label=method)
        plt.xlabel("estimates")
        plt.ylabel("frequency")
        plt.legend()
        if saveplot != '':
            plt.savefig(saveplot, bbox_inches='tight')
        plt.show()