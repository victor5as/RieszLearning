import os
import copy
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import scipy.stats
from .moments import ate_moment_fn, avg_der_moment_fn
import statsmodels.api as sm

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def L1_reg(net, l1_value, skip_list=()):
    L1_reg_loss = 0.0
    for name, param in net.named_parameters():
        if not param.requires_grad or len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            continue  # frozen weights
        else:
            L1_reg_loss += torch.sum(abs(param))
    L1_reg_loss *= l1_value
    return L1_reg_loss

class RieszArch(nn.Module):

    def __init__(self, learner):
        super(RieszArch, self).__init__()
        self.learner = learner
        # Scharfstein-Rotnitzky-Robins correction parameter
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.learner(x)
        # Scharfstein-Rotnitzky-Robins corrected output
        srr = out[:, [0]] + self.beta * out[:, [1]]
        return torch.cat([out, srr], dim=1)

class RieszNet:

    def __init__(self, learner, moment_fn):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        moment_fn : a function that takes as input a tuple (x, adversary, device) and
            evaluates the moment function at each of the x's, for a test function given by the adversary model.
            The adversary is a torch model that take as input x and return the output of the test function.
        """
        self.learner = RieszArch(learner)
        self.moment_fn = moment_fn

    def _pretrain(self, X, y, Xval, yval, *, bs,
                  warm_start, logger, model_dir, device, verbose):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.device = device

        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)
        if not torch.is_tensor(y):
            y = torch.Tensor(y).to(self.device)
        if (Xval is not None) and (not torch.is_tensor(Xval)):
            Xval = torch.Tensor(Xval).to(self.device)
        if (yval is not None) and (not torch.is_tensor(yval)):
            yval = torch.Tensor(yval).to(self.device)
        y = y.reshape(-1, 1)
        yval = yval.reshape(-1, 1)

        self.train_ds = TensorDataset(X, y)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        self.logger = logger
        if self.logger is not None:
            self.writer = SummaryWriter()

        return X, y, Xval, yval

    def _train(self, X, y, *, Xval, yval,
               earlystop_rounds, earlystop_delta,
               learner_l2, learner_l1, learner_lr,
               n_epochs, bs, target_reg, riesz_weight,
               optimizer):

        parameters = add_weight_decay(self.learner, learner_l2)
        if optimizer == 'adam':
            self.optimizerD = optim.Adam(parameters, lr=learner_lr)
        elif optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(parameters, lr=learner_lr, momentum=.9)
        elif optimizer == 'sgd':
            self.optimizerD = optim.SGD(parameters, lr=learner_lr, momentum=.9, nesterov=True)
        else:
            raise AttributeError("Not implemented")

        riesz_fn = lambda x: self.learner(x)[:, [1]]

        if Xval is not None:
            min_eval = np.inf
            time_since_last_improvement = 0
            best_learner_state_dict = copy.deepcopy(self.learner.state_dict())
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.5,
                patience=5, threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=0,
                eps=1e-08, verbose=(self.verbose>0))

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (xb, yb) in enumerate(self.train_dl):

                self.learner.train()
                output = self.learner(xb)

                L1_reg_loss = 0.0
                if learner_l1 > 0.0:
                    L1_reg_loss = L1_reg(self.learner, learner_l1)

                D_loss = torch.mean((yb - output[:, [0]]) ** 2)
                D_loss += riesz_weight * torch.mean(- 2 * self.moment_fn(
                    xb, riesz_fn, self.device) + output[:, [1]] ** 2)
                D_loss += target_reg * torch.mean((yb - output[:, [2]])**2)
                D_loss += L1_reg_loss

                self.optimizerD.zero_grad()
                D_loss.backward()
                self.optimizerD.step()
                self.learner.eval()

            if Xval is not None:  # if early stopping was enabled we check the out of sample violation
                output = self.learner(Xval)
                loss1 = np.mean(torch.mean((yval - output[:, [0]]) ** 2).cpu().detach().numpy())
                loss2 = np.mean(torch.mean(- 2 * self.moment_fn(
                    Xval, riesz_fn, self.device) + output[:, [1]] ** 2).cpu().detach().numpy())
                loss3 = np.mean(torch.mean((yval - output[:, [2]]) ** 2).cpu().detach().numpy())

                self.curr_eval = loss1 + riesz_weight * loss2 + target_reg * loss3
                lr_scheduler.step(self.curr_eval)

                if self.verbose > 0:
                    print("Validation losses:", loss1, loss2, loss3)
                if min_eval > self.curr_eval + earlystop_delta:
                    min_eval = self.curr_eval
                    time_since_last_improvement = 0
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())
                else:
                    time_since_last_improvement += 1
                    if time_since_last_improvement > earlystop_rounds:
                        break

            if self.logger is not None:
                self.logger(self, self.learner, epoch, self.writer)

        torch.save(self.learner, os.path.join(
            self.model_dir, "epoch{}".format(epoch)))

        self.n_epochs = epoch + 1
        if Xval is not None:
            self.learner.load_state_dict(best_learner_state_dict)
            torch.save(self.learner, os.path.join(
                self.model_dir, "earlystop"))

        return self

    def fit(self, X, y, Xval=None, yval=None, *,
            earlystop_rounds=20, earlystop_delta=0,
            learner_l2=1e-3, learner_l1=0, learner_lr=0.001,
            n_epochs=100, bs=100, target_reg=.1, riesz_weight=1.0, optimizer='adam',
            warm_start=False, logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        X : features of shape (n_samples, n_features)
        y : label of shape (n_samples, 1)
        Xval : validation set, if not None, then earlystopping is enabled based on out of sample moment violation
        yval : validation labels
        earlystop_rounds : how many epochs to wait for an out of sample improvement
        earlystop_delta : min increment for improvement for early stopping
        learner_l2 : l2_regularization of parameters of learner
        learner_l1 : l1_regularization of parameters of learner
        learner_lr : learning rate of the Adam optimizer for learner
        n_epochs : how many passes over the data
        bs : batch size
        target_reg : float in [0, 1]. weight on targeted regularization vs mse loss
        optimizer : one of {'adam', 'rmsprop', 'sgd'}. default='adam'
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        device : name of device on which to perform all computation
        verbose : whether to print messages related to progress of training
        """

        X, y, Xval, yval = self._pretrain(X, y, Xval, yval, bs=bs, warm_start=warm_start,
                                 logger=logger, model_dir=model_dir,
                                 device=device, verbose=verbose)

        self._train(X, y, Xval=Xval, yval=yval,
                    earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
                    learner_l2=learner_l2, learner_l1=learner_l1,
                    learner_lr=learner_lr, n_epochs=n_epochs, bs=bs,
                    target_reg=target_reg, riesz_weight=riesz_weight,
                    optimizer=optimizer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def get_model(self, model):
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))
        if model == 'earlystop':
            return torch.load(os.path.join(self.model_dir,
                                           "earlystop"))

        raise AttributeError("Not implemented")

    def predict(self, X, model='final'):
        """
        Parameters
        ----------
        X : (n, p) matrix of features
        model : one of ('final', 'earlystop'), whether to use an average of models or the final
        Returns
        -------
        ypred, apred : (n, 2) matrix of learned regression and riesz representers g(X), a(X)
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)

        return self.get_model(model)(X).cpu().data.numpy()

    def _postTMLE(self, Xtest, ytest, correctionmethod='residuals', model='final', postproc_riesz=False):

        if not torch.is_tensor(Xtest):
            Xtest = torch.Tensor(Xtest).to(self.device)
        if torch.is_tensor(ytest):
            ytest = ytest.cpu().data.numpy()
        
        ytest = ytest.flatten()
        pred_test = self.predict(Xtest, model=model)
        a_test = pred_test[:, 1]
        y_pred_test = pred_test[:, 0]

        if postproc_riesz:
            agmm_model = self.get_model(model)
            riesz_fn = lambda x: agmm_model(x)[:, [1]]
            d = torch.mean(self.moment_fn(Xtest, riesz_fn, self.device)).cpu().detach().numpy().flatten() / (np.mean(a_test ** 2) - np.mean(a_test) ** 2)
            c = - d * torch.mean(self.moment_fn(Xtest, riesz_fn, self.device)).cpu().detach().numpy().flatten()
            a_test = c + d * a_test

        if correctionmethod == 'residuals':
            res = (ytest - y_pred_test)
            tmle = sm.OLS(res, a_test).fit()
            return tmle

        elif correctionmethod == 'full':
            tmle = sm.OLS(ytest, np.c_[y_pred_test, a_test]).fit()
            return tmle

    def predict_avg_moment(self, Xtest, ytest, method='dr',
                           model='final', alpha=0.05, srr=True, postTMLE=False, correctionmethod='residuals',
                           postproc_riesz=False):
        """
        Parameters
        ----------
        Xtest : (n, p) matrix of features
        ytest : (n,) vector of labels
        method : one of ('dr', 'ips', 'reg') for approach
        model : one of ('final', 'earlystop'), whether to use the final or the earlystop model
        alpha : confidence level, creates (1 - alpha)*100% confidence interval
        srr : whether to apply Scharfstein-Rotnitzky-Robins correction to regressor
        Returns
        -------
        avg_moment, lb, ub: avg moment with confidence intervals
        """
        if not torch.is_tensor(Xtest):
            Xtest = torch.Tensor(Xtest).to(self.device)
        if torch.is_tensor(ytest):
            ytest = ytest.cpu().data.numpy()
        ytest = ytest.flatten()

        pred_test = self.predict(Xtest, model=model)
        a_test = pred_test[:, 1]
        # Robins-Rotnitzky-Scharfstein correction or not
        y_pred_test = pred_test[:, 2] if srr and not postTMLE else pred_test[:, 0]
        agmm_model = self.get_model(model)
        reg_fn = lambda x: agmm_model(x)[:, [2]] if srr and not postTMLE else agmm_model(x)[:, [0]]
        riesz_fn = lambda x: agmm_model(x)[:, [1]]

        if postproc_riesz:
            d = torch.mean(self.moment_fn(Xtest, riesz_fn, self.device)).cpu().detach().numpy().flatten() / (np.mean(a_test ** 2) - np.mean(a_test) ** 2)
            c = - d * torch.mean(self.moment_fn(Xtest, riesz_fn, self.device)).cpu().detach().numpy().flatten()
            a_test = c + d * a_test
            riesz_fn = lambda x: c + d * agmm_model(x)[:, [1]]

        if postTMLE:
            tmle = self._postTMLE(Xtest, ytest, correctionmethod=correctionmethod, model=model, postproc_riesz=postproc_riesz)
            if correctionmethod == 'residuals':
                adj_reg_fn = lambda x: reg_fn(x).flatten() + torch.Tensor(tmle.predict(self.predict(x, model=model)[:, [1]])).to(self.device).flatten()
            elif correctionmethod == 'full':
                adj_reg_fn = lambda x: torch.Tensor(tmle.predict(self.predict(x, model=model)[:, [0, 1]])).to(self.device).flatten()
            y_pred_test = adj_reg_fn(Xtest).cpu().data.numpy().flatten()
        else:
            adj_reg_fn = reg_fn

        if method=='direct':
            return mean_ci(self.moment_fn(Xtest, adj_reg_fn, self.device).cpu().detach().numpy().flatten(),
                           confidence=1 - alpha)
        elif method=='ips':
            return mean_ci(a_test * ytest, confidence=1 - alpha)
        elif method=='dr':
            return mean_ci((self.moment_fn(Xtest, adj_reg_fn, self.device).cpu().detach().numpy().flatten()
                            + a_test * (ytest - y_pred_test)),
                           confidence=1 - alpha)
        else:
            raise AttributeError('not implemented')

class RieszNetRR:

    def __init__(self, learner, moment_fn):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        moment_fn : a function that takes as input a tuple (x, adversary, device) and
            evaluates the moment function at each of the x's, for a test function given by the adversary model.
            The adversary is a torch model that take as input x and return the output of the test function.
        """
        self.learner = learner
        self.moment_fn = moment_fn

    def _pretrain(self, X, Xval, *, bs,
                  warm_start, logger, model_dir, device, verbose):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.device = device

        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)
        if (Xval is not None) and (not torch.is_tensor(Xval)):
            Xval = torch.Tensor(Xval).to(self.device)

        self.train_ds = TensorDataset(X)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        self.logger = logger
        if self.logger is not None:
            self.writer = SummaryWriter()

        return X, Xval

    def _train(self, X, *, Xval,
               earlystop_rounds, earlystop_delta,
               learner_l2, learner_l1, learner_lr,
               n_epochs, bs, optimizer):

        parameters = add_weight_decay(self.learner, learner_l2)
        if optimizer == 'adam':
            self.optimizerD = optim.Adam(parameters, lr=learner_lr)
        elif optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(parameters, lr=learner_lr, momentum=.9)
        elif optimizer == 'sgd':
            self.optimizerD = optim.SGD(parameters, lr=learner_lr, momentum=.9, nesterov=True)
        else:
            raise AttributeError("Not implemented")

        riesz_fn = lambda x: self.learner(x)[:, [0]]

        if Xval is not None:
            min_eval = np.inf
            time_since_last_improvement = 0
            best_learner_state_dict = copy.deepcopy(self.learner.state_dict())
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.5,
                patience=5, threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=0,
                eps=1e-08, verbose=(self.verbose>0))

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (xb,) in enumerate(self.train_dl):

                self.learner.train()
                output = self.learner(xb)[:, [0]]

                L1_reg_loss = 0.0
                if learner_l1 > 0.0:
                    L1_reg_loss = L1_reg(self.learner, learner_l1)

                D_loss = torch.mean(- 2 * self.moment_fn(xb, riesz_fn, self.device) + output ** 2)
                D_loss += L1_reg_loss

                self.optimizerD.zero_grad()
                D_loss.backward()
                self.optimizerD.step()
                self.learner.eval()

            if Xval is not None:  # if early stopping was enabled we check the out of sample violation
                output = self.learner(Xval)[:, [0]]
                loss = np.mean(torch.mean(- 2 * self.moment_fn(
                    Xval, riesz_fn, self.device) + output ** 2).cpu().detach().numpy())
                self.curr_eval = loss 
                lr_scheduler.step(self.curr_eval)

                if self.verbose > 0:
                    print("Validation losses:", loss)
                if min_eval > self.curr_eval + earlystop_delta:
                    min_eval = self.curr_eval
                    time_since_last_improvement = 0
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())
                else:
                    time_since_last_improvement += 1
                    if time_since_last_improvement > earlystop_rounds:
                        break

            if self.logger is not None:
                self.logger(self, self.learner, epoch, self.writer)

        torch.save(self.learner, os.path.join(
            self.model_dir, "epoch{}".format(epoch)))

        self.n_epochs = epoch + 1
        if Xval is not None:
            self.learner.load_state_dict(best_learner_state_dict)
            torch.save(self.learner, os.path.join(
                self.model_dir, "earlystop"))

        return self

    def fit(self, X, Xval=None, *,
            earlystop_rounds=20, earlystop_delta=0,
            learner_l2=1e-3, learner_l1=0.0, learner_lr=0.001,
            n_epochs=100, bs=100, optimizer='adam',
            warm_start=False, logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        X : features of shape (n_samples, n_features)
        Xval : validation set, if not None, then earlystopping is enabled based on out of sample moment violation
        earlystop_rounds : how many epochs to wait for an out of sample improvement
        earlystop_delta : min increment for improvement for early stopping
        learner_l2 : l2_regularization of parameters of learner
        learner_l1 : l1_regularization of parameters of learner
        learner_lr : learning rate of the Adam optimizer for learner
        n_epochs : how many passes over the data
        bs : batch size
        optimizer : one of {'adam', 'rmsprop', 'sgd'}. default='adam'
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        device : name of device on which to perform all computation
        verbose : whether to print messages related to progress of training
        """

        X, Xval = self._pretrain(X, Xval, bs=bs, warm_start=warm_start,
                                 logger=logger, model_dir=model_dir,
                                 device=device, verbose=verbose)

        self._train(X, Xval=Xval,
                    earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
                    learner_l2=learner_l2, learner_l1=learner_l1,
                    learner_lr=learner_lr, n_epochs=n_epochs, bs=bs,
                    optimizer=optimizer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def get_model(self, model):
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))
        if model == 'earlystop':
            return torch.load(os.path.join(self.model_dir,
                                           "earlystop"))

        raise AttributeError("Not implemented")

    def predict(self, X, model='final'):
        """
        Parameters
        ----------
        X : (n, p) matrix of features
        model : one of ('final', 'earlystop'), whether to use an average of models or the final
        Returns
        -------
        apred : (n, 1) rr
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)

        return self.get_model(model)(X).cpu().data.numpy()

class ATERieszNet(RieszNet):

    def __init__(self, learner):
        super().__init__(learner, ate_moment_fn)

class AvgDerivativeRieszNet(RieszNet):

    def __init__(self, learner):
        super().__init__(learner, avg_der_moment_fn)
