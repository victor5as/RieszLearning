# Random Forest Average Derivative Monte Carlo Simulations

# 0. Importing modules
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump
import scipy
import scipy.stats
import scipy.special
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from .forestriesz import ForestRiesz, RFreg, RFrr, poly_feature_fns
from sklearn.model_selection import KFold
import statsmodels.api as sm

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def rmse_fn(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

methods = ['dr', 'reg', 'ips', 'tmle', 'plugin']

# 1. Estimation
def predict_avg_mom(y, X, reg, rr, mom_reg, mom_rr, method = 'dr'):
    y = y.flatten()
    reg = reg.flatten()
    rr = rr.flatten()
    mom_reg = mom_reg.flatten()
    mom_rr = mom_rr.flatten()

    if method == 'reg':
        return mean_ci(mom_reg)
    elif method == 'ips':
        return mean_ci(rr * y)
    elif method == 'dr':
        return mean_ci(mom_reg + rr * (y - reg))
    elif method == 'tmle':
        res = y - reg
        tmle = sm.OLS(res, rr).fit()
        return mean_ci(mom_reg + tmle.predict(mom_rr)
                       + rr * (y - reg - tmle.predict(rr)))
    elif method == 'plugin':
        mu_T = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 50, random_state = 123)
        mu_T.fit(X[:, 1:], X[:, 0])
        sigma2_T = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 50, max_depth = 5, random_state = 123)
        e_T = X[:, 0] - cross_val_predict(mu_T, X[:, 1:], X[:, 0])
        sigma2_T.fit(X[:, 1:], e_T ** 2)
        rr = ((X[:, 0] - mu_T.predict(X[:, 1:]))/(sigma2_T.predict(X[:, 1:]))).flatten()
        return mean_ci(mom_reg + rr * (y - reg))
    else:
        raise AttributeError('not implemented')

class OracleReg:

    def __init__(self, true_reg, scale):
        self.true_reg = true_reg
        self.scale = scale

    def fit(self, X, T, y):
        return self

    def predict_reg(self, X):
        return self.true_reg(X) / self.scale

class OracleRR:

    def __init__(self, true_rr):
        self.true_rr = true_rr

    def fit(self, X, T, y):
        return self

    def predict_riesz(self, X):
        return self.true_rr(X)

def est_avgmom_RF(X, y, moment_fn, true_reg, true_rr, scale_y = True,
                  xfit = 0, multitasking = True, oracle = '',
                  ForestRiesz_opt = {}, RFreg_opt = {}, RFrr_opt = {}):

    # Scale y
    if scale_y:
        scaler = StandardScaler().fit(y.reshape(-1, 1))
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
        scale, offset = scaler.scale_[0], scaler.mean_[0]
    else:
        y_scaled, scale, offset = y, 1.0, 0

    if oracle != '':
        multitasking = False

    gen_reg = lambda : OracleReg(true_reg, scale) if oracle == 'reg' else RFreg(**RFreg_opt)
    gen_rr = lambda : OracleRR(true_rr) if oracle == 'rr' else RFrr(**RFrr_opt)
    gen_est = lambda : ForestRiesz(**ForestRiesz_opt)

    # nuisance quantities for each sample
    reg_hat, rr_hat, mom_reg, mom_rr = (np.zeros(X.shape[0]), np.zeros(X.shape[0]),
                                        np.zeros(X.shape[0]), np.zeros(X.shape[0]))
    # No cross-fitting
    if xfit == 0:

        # Multitasking
        if multitasking:
            est = gen_est().fit(X[:, 1:], X[:, [0]], y_scaled.reshape(-1, 1))

            rr_hat, reg_hat = est.predict_riesz_and_reg(X)
            mom_reg, mom_rr = moment_fn(X, est.predict_reg), moment_fn(X, est.predict_riesz)

        # No multitasking
        else:
            reg = gen_reg().fit(X[:, 1:], X[:, [0]], y_scaled.reshape(-1, 1))
            rr = gen_rr().fit(X[:, 1:], X[:, [0]], y_scaled.reshape(-1, 1))

            reg_hat, rr_hat = reg.predict_reg(X), rr.predict_riesz(X)
            mom_reg, mom_rr = moment_fn(X, reg.predict_reg), moment_fn(X, rr.predict_riesz)

    # 5-fold cross-fitting
    else:
        if (xfit == 2 and multitasking) or (xfit > 2):
            raise AttributeError("Not implemented or available!")
        # Sample split
        for train, test in KFold(5).split(X):

            # Multitasking
            if multitasking:
                est = gen_est().fit(X[train, 1:], X[train, :1], y_scaled[train].reshape(-1, 1))

                rr_hat[test], reg_hat[test] = est.predict_riesz_and_reg(X[test])
                mom_reg[test], mom_rr[test] = moment_fn(X[test], est.predict_reg), moment_fn(X[test], est.predict_riesz)

            # No multitasking
            else:
                # Check if we are doing three way cross-fitting
                nt = len(train)
                regfold = train[nt//2:] if xfit == 2 else train
                rrfold = train[:nt//2] if xfit == 2 else train

                reg = gen_reg().fit(X[regfold, 1:], X[regfold, [0]], y_scaled[regfold].reshape(-1, 1))
                rr = gen_rr().fit(X[rrfold, 1:], X[rrfold, [0]], y_scaled[rrfold].reshape(-1, 1))

                reg_hat[test], rr_hat[test] = reg.predict_reg(X[test]), rr.predict_riesz(X[test])
                mom_reg[test], mom_rr[test] = moment_fn(X[test], reg.predict_reg), moment_fn(X[test], rr.predict_riesz)

    rmse_reg = rmse_fn(reg_hat * scale + offset, true_reg(X))
    r2_reg = 1 - (rmse_reg ** 2) / np.var(true_reg(X))
    rmse_rr = rmse_fn(rr_hat, true_rr(X))
    r2_rr = 1 - (rmse_rr ** 2) / np.var(true_rr(X))
    ipsbias = np.mean((rr_hat - true_rr(X)) * true_reg(X))
    drbias = np.mean((rr_hat - true_rr(X)) * (true_reg(X) - reg_hat * scale + offset))

    # Return average moment and CI for all methods
    final_params = tuple(x * scale for method in methods
                         for x in predict_avg_mom(y_scaled, X, reg_hat, rr_hat, mom_reg, mom_rr, method = method))
    nuisance_metrics = (rmse_reg, r2_reg, rmse_rr, r2_rr, ipsbias, drbias)
    return final_params + nuisance_metrics

# 2. Simulations
def get_est(W, *, moment_fn, true_reg, true_rr, gen_y, gen_T, sim = 1, oracle = '', scale_y = True,
            xfit = 0, multitasking = True, ForestRiesz_opt = {}, RFreg_opt = {}, RFrr_opt = {}, seed = 1234):

    np.random.seed(seed + sim)
    X = np.hstack((gen_T(W), W))
    y = gen_y(X)
    truth = np.mean(moment_fn(X, true_reg))

    return est_avgmom_RF(X, y, moment_fn, true_reg = true_reg, true_rr = true_rr, scale_y = scale_y,
                         xfit = xfit, multitasking = multitasking, oracle = oracle,
                         ForestRiesz_opt = ForestRiesz_opt, RFreg_opt = RFreg_opt, RFrr_opt = RFrr_opt) + (truth,)


def sim_fun(W, *, moment_fn, true_reg, true_rr, gen_y, gen_T, N_sim = 100, oracle = '', scale_y = True, xfit = 0,
            multitasking = True, ForestRiesz_opt = {}, RFreg_opt = {}, RFrr_opt = {}, seed = 1234, verbose = 1,
            save = '', plot = True, saveplot = ''):

    res = Parallel(n_jobs = -1, verbose = verbose)(delayed(get_est)(W, moment_fn = moment_fn, true_reg = true_reg, true_rr = true_rr,
                                                                    gen_y = gen_y, gen_T = gen_T, sim = sim, oracle = oracle,
                                                                    scale_y = scale_y, xfit = xfit, multitasking = multitasking,
                                                                    ForestRiesz_opt = ForestRiesz_opt, RFreg_opt = RFreg_opt,
                                                                    RFrr_opt = RFrr_opt, seed = seed)
                                                                    for sim in range(N_sim))

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
        plt.title(method_strs)
        plt.axvline(x = np.mean(truth), label='true', color='red')
        for method, d in res_dict.items():
            plt.hist(np.array(d['point']), alpha=.5, label=method)
        plt.xlabel("estimates")
        plt.ylabel("frequency")
        plt.legend()
        if saveplot != '':
            plt.savefig(saveplot, bbox_inches='tight')
        plt.show()
