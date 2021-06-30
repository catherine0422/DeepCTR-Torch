import torch

def apply2nestLists(f, xss, needZip = False):
    if needZip:
        return [[f([x , y]) for (x, y) in zip(xs, ys)] for (xs, ys) in zip(*xss)]
    else:
        return [[f(x) for x in xs] for xs in xss]


def clone_embs(x, device):
    return x.clone().detach().to(device).requires_grad_(True)


def get_grad(x, cost):
    return torch.autograd.grad(cost, x, retain_graph=True, create_graph=False)[0]


def add_nestLists(xss, yss):
    return [[x + y for (x, y) in zip(xs, ys)] for (xs, ys) in zip(xss, yss)]


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
    deltas = [delta for delta in deltas if len(delta) > 0]
    ## 将不同shape的delta合并
    if len(deltas) <= 0:
        return 0
    deltas_num = len(deltas[0][0])
    deltas = apply2nestLists(lambda x: x.view(deltas_num, -1), deltas)
    deltas = cat_nestLists(deltas)

    delta_len = len(deltas[0])
    mse = torch.sum(deltas * deltas, dim=1) / delta_len
    rmse = torch.sqrt(mse).mean()
    return rmse