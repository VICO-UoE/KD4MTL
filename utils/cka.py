import math
import numpy as np
import torch

# def centering(K):
#     n = K.shape[0]
#     unit = np.ones([n, n])
#     I = np.eye(n)
#     H = I - unit / n

#     return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
#     # return np.dot(H, K)  # KH


# def rbf(X, sigma=None):
#     GX = np.dot(X, X.T)
#     KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
#     if sigma is None:
#         mdist = np.median(KX[KX != 0])
#         sigma = math.sqrt(mdist)
#     KX *= - 0.5 / (sigma * sigma)
#     KX = np.exp(KX)
#     return KX


# def kernel_HSIC(X, Y, sigma):
#     return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


# def linear_HSIC(X, Y):
#     L_X = np.dot(X, X.T)
#     L_Y = np.dot(Y, Y.T)
#     return np.sum(centering(L_X) * centering(L_Y))


# def linear_CKA(X, Y):
#     hsic = linear_HSIC(X, Y)
#     var1 = np.sqrt(linear_HSIC(X, X))
#     var2 = np.sqrt(linear_HSIC(Y, Y))

#     return hsic / (var1 * var2)


# def kernel_CKA(X, Y, sigma=None):
#     hsic = kernel_HSIC(X, Y, sigma)
#     var1 = np.sqrt(kernel_HSIC(X, X, sigma))
#     var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

#     return hsic / (var1 * var2)

def centering(K):
    n = K.size(0)
    unit = torch.ones(n,n).cuda()
    I = torch.eye(n).cuda()
    H = I - unit/n
    return torch.mm(torch.mm(H, K), H)

def linear_HSIC(X, Y):
    L_X = torch.mm(X, X.transpose(0,1) )
    L_Y = torch.mm(Y, Y.transpose(0,1) )
    return torch.sum(centering(L_X) * centering(L_Y))

def rbf(X, sigma=None):
    GX = torch.mm(X, X.transpose(0,1) )
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).transpose(0,1) 
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

