import torch
import numpy as np


def check_None(f, x, multi_arg=False):
    if multi_arg:
        for element in x:
            if element is None: return None
        return f(*x)
    else:
        if x is None: return None
        return f(x)


def apply2nestLists(f, xs):
    if type(xs) == tuple:
        return [check_None(f, x, multi_arg=True) for x in zip(*xs)]
    else:
        return [check_None(f, x) for x in xs]


# def delta_step(grads, eps, need_get_grad = False):
#     if type(eps) in [list, np.ndarray, tuple]:
#         if len(eps) != len(grads):
#             raise ValueError(
#                 f'number of distortion step dosen\'t fit the number of embeddings: {len(eps)} != {len(grads)}')
#         if need_get_grad:
#             deltas = apply2nestLists(lambda x, y: x * y.grad.sign(), (eps, grads))
#         else:
#             deltas = apply2nestLists(lambda x, y: x * y.sign(), (eps, grads))
#     else:
#         if need_get_grad:
#             deltas = apply2nestLists(lambda x: eps * x.grad.sign(), grads)
#         else:
#             deltas = apply2nestLists(lambda x: eps * x.sign(), grads)
#     return deltas

def delta_step(grads, eps, need_get_grad = False):
    if need_get_grad:
        f = lambda x,y: (x*y.grad.sign()).detach()
    else:
        f = lambda x,y: (x*y.sign()).detach()
    deltas = func_detect_arg_type(f, eps, grads)
    return deltas


def clamp_step(deltas, eps):
    deltas = func_detect_arg_type(lambda x, y: y.clamp(-x, x).detach(), eps, deltas)
    return deltas


def func_detect_arg_type(f, unclear_arg, *others):
    if type(unclear_arg) in [list, np.ndarray, tuple]:
        if len(unclear_arg) != len(others[0]):
            raise ValueError(
                f'number of values dosen\'t fit: {len(unclear_arg)} != {len(others[0])}')
        return apply2nestLists(f, (unclear_arg, *others))
    else:
        return apply2nestLists(lambda x:f(unclear_arg, x), *others)


def clone_embs(x, device):
    return x.clone().detach().to(device).requires_grad_(True)


def get_grad(x, cost):
    return torch.autograd.grad(cost, x, retain_graph=True, create_graph=False)[0]


def add_nestLists(xs, ys):
    for x, y in zip(xs, ys):
        if x is not None and y is not None:
            size_x, size_y = x.size(0), y.size(0)
            break
    if size_x == size_y:
        return [check_None(lambda a, b: a + b, x, multi_arg=True) for x in zip(xs, ys)]
    else:
        if size_x > size_y:
            bigs, smalls = xs, ys
        else:
            bigs, smalls = ys, xs
        for big, small in zip(bigs, smalls):
            if big is not None and small is not None:
                big[:small.size(0)] += small
        return bigs


def cat_nestLists(xss, dim=1):
    xs = [torch.cat(xs, dim=dim) for xs in xss if len(xs) > 0]
    x = torch.cat(xs, dim=dim)
    return x


def get_rmse(deltas):
    '''
    一个batch的数据的平均root mean squared error
    deltas: list of 各个embbeing的扰动，每个embbeing shape不同
    '''
    ## 删除为空的元素
    deltas = [delta for delta in deltas if delta is not None and len(delta) > 0]
    ## 将不同shape的delta合并
    if len(deltas) <= 0:
        return 0
    # deltas_num = len(deltas[0][0])
    deltas_num = deltas[0].size(0)
    deltas = apply2nestLists(lambda x: x.view(deltas_num, -1), deltas)
    deltas = torch.cat(deltas, dim=1)

    mse = torch.mean(deltas * deltas, dim=1)
    rmse = torch.sqrt(mse).mean()
    return rmse


def trades_loss(x, x_adv):
    min_limit = torch.ones(x.shape).to(x.device) * 0.001

    def log_cust(a):
        return torch.log(torch.max(a, min_limit))

    loss = x * (log_cust(x) - log_cust(x_adv)) + (1 - x) * (log_cust(1 - x) - log_cust(1 - x_adv))
    return loss.sum()
