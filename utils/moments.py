import torch
import numpy as np

# Returns the moment for the ATE example, for each sample in x
def ate_moment_fn(x, test_fn, device):
    if torch.is_tensor(x):
        with torch.no_grad():
            t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1) - test_fn(t0)


def policy_moment_gen(policy):
    def policy_moment_fn(x, test_fn, device):
        with torch.no_grad():
            if torch.is_tensor(x):
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
                t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            else:
                t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
                t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
            p1 = policy(x)
        out1 = test_fn(t1)
        out0 = test_fn(t0)
        if len(out1.shape) > 1:
            p1 = p1.reshape(-1, 1)
        return out1 * p1 + out0 * (1 - p1)

    return policy_moment_fn


def trans_moment_gen(trans):
    def trans_moment_fn(x, test_fn, device):
        with torch.no_grad():
            if torch.is_tensor(x):
                tx = torch.cat([x[:, [0]], trans(x[:, [1]]), x[:, 2:]], dim=1)
            else:
                tx =  np.hstack([x[:, [0]], trans(x[:, [1]]), x[:, 2:]])
        return test_fn(tx) - test_fn(x)

    return trans_moment_fn


def avg_der_moment_fn(x, test_fn, device):
    if torch.is_tensor(x):
        T = torch.autograd.Variable(x[:, [0]], requires_grad=True)
        input = torch.cat([T, x[:, 1:]], dim=1)
        output = test_fn(input)
        gradients = torch.autograd.grad(outputs=output, inputs=T,
                              grad_outputs=torch.ones(output.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        raise AttributeError('Not implemented')
    return gradients

def avg_small_diff(x, test_fn, device):
    epsilon = 0.01

    if torch.is_tensor(x):
        with torch.no_grad():
            t1 = torch.cat([(x[:, [0]] + epsilon).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([(x[:, [0]] - epsilon).to(device), x[:, 1:]], dim=1)
    else:
        t1 = np.hstack([x[:, [0]] + epsilon, x[:, 1:]])
        t0 = np.hstack([x[:, [0]] - epsilon, x[:, 1:]])
    return (test_fn(t1) - test_fn(t0)) / (2*epsilon)
