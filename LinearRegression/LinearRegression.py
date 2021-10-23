import numpy as np
import torch


def cal_avg_square_error(w, b, points):
    """
    loss = (y-(wx+b))^2
    :param w:
    :param b:
    :param points:
    :return:
    """
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (w*x + b - y) ** 2
    return total_error / len(points)


def step_gradient(w, b, lr, points):
    b_gradient = 0
    w_gradient = 0
    N = len(points)

    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N) * (w * x + b - y)
        w_gradient += (2/N) * x * (w * x + b - y)
    return [w - lr * b_gradient, b - lr * b_gradient]


def iter_gradient(w_start, b_start, lr, points, iter):
    b = b_start
    w = w_start
    for i in range(iter):
        w, b = step_gradient(w, b, lr, np.array(points))
    return [w, b]

w_start = 20
b_start = 10
lr = 1e-3
points = np.genfromtxt("data.csv", delimiter=",")
iter = 10000

print("start w = {0} b = {1} error = {2}".format(w_start, b_start,
                                                 cal_avg_square_error(w_start, b_start, points)))

w,b = iter_gradient(w_start, b_start, lr, points, iter)
print("after w = {0} b = {1} error = {2}".format(w, b,
                                                 cal_avg_square_error(w, b, points)))

