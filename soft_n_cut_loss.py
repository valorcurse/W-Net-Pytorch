# Some methods in this file ported to Pytorch from https://github.com/Ashish77IITM/W-Net/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
import numpy as np

import torch.autograd.profiler as profiler
from config import Config

config = Config()

# The weight matrix w is a measure of the weight between each pixel and
# every other pixel. so w[u][v] is a measure of
#   (a) Distance between the brightness of the two pixels.
#   (b) Distance in positon between the two pixels

# The NCut loss metric is then:
#   (a) association:    sum(weight_connection) * P(both pixels in the connection are in the class)
#   (b) disassociation: sum(weight_connection) * P(first pixel is in the class)
#   N Cut loss = disassociation / association

def soft_n_cut_loss(inputs, segmentations):
    # We don't do n_cut_loss batch wise -- split it up and do it instance wise
    loss = 0

    flat_images = torch.mean(inputs, dim=1)
    flat_images.reshape(flat_images.shape[0], flat_images.shape[1]*flat_images.shape[2])
    losses = soft_n_cut_loss_(flat_images, segmentations, config.k, config.input_size, config.input_size)
    # for i in range(inputs.shape[0]):
        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        # flatten_image = torch.mean(inputs[i], dim=0)
        # flatten_image = flatten_image.reshape(flatten_image.shape[0]**2)
        # loss += soft_n_cut_loss_(flatten_image, segmentations[i], config.k, config.input_size, config.input_size)
            # local_loss = soft_n_cut_loss_(flatten_image, segmentations[i], config.k, config.input_size, config.input_size)
            # local_loss.backwards()

        # del flatten_image
        # torch.cuda.empty_cache()
        # print(prof)
        # print("")
    loss = loss / inputs.shape[0]
    return loss

def soft_n_cut_loss_(flatten_image, prob, k, rows, cols):
    '''
    Inputs:
    prob : (rows*cols*k) tensor
    k : number of classes (integer)
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    rows : number of the rows in the original image
    cols : number of the cols in the original image
    Output :
    soft_n_cut_loss tensor for a single image
    '''

    soft_n_cut_loss = k
    weights = edge_weights(flatten_image, rows,cols)

    # for t in range(k):
    #     soft_n_cut_loss -= (numerator(prob[t,:,], weights) / denominator(prob[t,:,:], weights))


    flat_prob = prob.reshape((prob.shape[0], prob.shape[1], prob.shape[2]*prob.shape[3]))

    # Numerator
    outer = flat_prob.unsqueeze(3) * flat_prob.unsqueeze(2)
    # a = torch.mul(weights, outer)
    a = weights.unsqueeze(1) * outer
    nom = torch.sum(a)

    del outer, a
    torch.cuda.empty_cache()

    # Denominator
    # k_class_prob = k_class_prob.view(-1)
    # denom = torch.sum( torch.mul(weights, flat_prob.unsqueeze(1) * torch.ones_like(flat_prob).unsqueeze(2)))
    denom = torch.sum(weights.unsqueeze(1) * (flat_prob.unsqueeze(2) * torch.ones_like(flat_prob).unsqueeze(3)))
    del weights

    new_loss = soft_n_cut_loss - (nom / denom)

    del nom, denom
    torch.cuda.empty_cache()

    return new_loss

def edge_weights(flatten_image, rows, cols, std_intensity=3, std_position=1, radius=5):
    '''
    Inputs :
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    std_intensity : standard deviation for intensity
    std_position : standard devistion for position
    radius : the length of the around the pixel where the weights
    is non-zero
    rows : rows of the original image (unflattened image)
    cols : cols of the original image (unflattened image)
    Output :
    weights :  2d tf array edge weights in the pixel graph
    Used parameters :
    n : number of pixels
    '''
    ones = torch.ones_like(flatten_image[0], dtype=torch.float, device=torch.device('cuda:0')).reshape(96*96).unsqueeze(0)
    # if torch.cuda.is_available():
    #     ones = ones.cuda()

    A = flatten_image.reshape(flatten_image.shape[0], flatten_image.shape[1]*flatten_image.shape[2]).unsqueeze(2) * ones

    # A = outer_product(flatten_image, ones)
    # A_T = torch.t(A)
    d = torch.div((A - A.permute((0, 2, 1))), std_intensity)
    intensity_weight = torch.exp(-1*torch.mul(d, d))

    del A, d
    torch.cuda.empty_cache()

    xx, yy = torch.meshgrid(torch.arange(rows, dtype=torch.float), torch.arange(cols, dtype=torch.float))
    xx = xx.reshape(rows*cols).cuda()
    yy = yy.reshape(rows*cols).cuda()
    # if torch.cuda.is_available():
    #     xx = xx.cuda()
    #     yy = yy.cuda()
    # cuda_available = torch.cuda.is_available()
    # xx = torch.where(cuda_available, xx.cuda(), xx)
    # yy = torch.where(cuda_available, yy.cuda(), yy)

    ones_xx = torch.ones_like(xx, dtype=torch.float, device=torch.device('cuda:0'))
    ones_yy = torch.ones_like(yy, dtype=torch.float, device=torch.device('cuda:0'))
    # if torch.cuda.is_available():
    #     ones_yy = ones_yy.cuda()
    #     ones_xx = ones_xx.cuda()

    # ones_yy = torch.where(cuda_available, ones_yy.cuda(), ones_yy)
    # ones_xx = torch.where(cuda_available, ones_xx.cuda(), ones_xx)

    A_x = outer_product(xx, ones_xx)
    # A_x = xx * ones_xx
    A_y = outer_product(yy, ones_yy)
    # A_y = yy.unqueeze(2) * ones_yy.unsqueeze(1)

    xi_xj = A_x - torch.t(A_x)
    yi_yj = A_y - torch.t(A_y)

    del A_x, A_y, ones_xx, ones_yy
    torch.cuda.empty_cache()

    sq_distance_matrix = torch.mul(xi_xj, xi_xj) + torch.mul(yi_yj, yi_yj)
    del xi_xj, yi_yj
    torch.cuda.empty_cache()
    # Might have to consider casting as float32 instead of creating meshgrid as float32

    dist_weight = torch.exp(-torch.div(sq_distance_matrix,std_position**2))
    del sq_distance_matrix
    torch.cuda.empty_cache()

    weight = torch.mul(intensity_weight, dist_weight) # Element wise product


    # ele_diff = tf.reshape(ele_diff, (rows, cols))
    # w = ele_diff + distance_matrix
    return weight

def outer_product(v1,v2):
    '''
    Inputs:
    v1 : m*1 tf array
    v2 : m*1 tf array
    Output :
    v1 x v2 : m*m array
    '''
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    v1 = torch.unsqueeze(v1, dim=0)
    v2 = torch.unsqueeze(v2, dim=0)
    return torch.matmul(torch.t(v1),v2)

def numerator(k_class_prob,weights):
    '''
    Inputs :
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights n*n tensor
    '''
    k_class_prob = k_class_prob.reshape(-1)
    a = torch.mul(weights,outer_product(k_class_prob,k_class_prob))
    return torch.sum(a)

def denominator(k_class_prob,weights):
    '''
    Inputs:
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights	n*n tensor
    '''
    k_class_prob = k_class_prob.view(-1)
    return torch.sum(
        torch.mul(
            weights,
            outer_product(
                k_class_prob,
                torch.ones_like(k_class_prob)
                )
            )
        )
