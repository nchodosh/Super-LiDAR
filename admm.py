import tensorflow as tf
import numpy as np

def maxpool(x, kern, stride):
    return tf.nn.max_pool(tf.pad(x, [[0, 0], [kern//2, kern//2],
                                      [kern//2, kern//2], [0, 0]]),
                          [ 1, kern, kern, 1 ], [ 1, stride, stride, 1], 'VALID')
def count(x, kern, stride):
    kern = tf.ones([kern, kern, 1, 1])
    return tf.nn.conv2d(x, kern, [ 1, stride, stride, 1], 'SAME')

def make_admm(sdmask, sd, dmask, d, tv_loss,
              num_iters, kernels, filters, strides):
    print(sdmask.get_shape().as_list())
    print(sd.get_shape().as_list())    
    n = len(kernels)
    mask = sdmask
    in_channels = sd.get_shape().as_list()[-1]
    print(in_channels)
    w = {}
    b = {}
    m = {}
    for i, kern, filt, stride in zip(range(len(filters)), kernels, filters, strides):
        stddev = 2/(kern*kern*filt)
        w[i] = tf.get_variable('kernel{}'.format(i), [ kern, kern, in_channels, filt ],
                               dtype = tf.float32,
                               initializer = tf.random_normal_initializer(stddev =
                                                                          np.sqrt(stddev)))
        b[i] = tf.get_variable('bias{}'.format(i), (), dtype = tf.float32,
                               initializer = tf.ones_initializer())*0.001
        if i > 0:
            m[i] = tf.cast(tf.greater(count(m[i-1], kern, stride), 0), tf.float32)
            print(m[i].get_shape().as_list())
        else:
            m[i] = tf.cast(tf.greater(count(sdmask, kern, stride), 0), tf.float32)
        in_channels = filt
    def Wt(x, i):
        return tf.nn.conv2d(x, w[i], [ 1, strides[i], strides[i], 1], 'SAME')
    def W(x, i, output_shape):
        xshape = x.get_shape().as_list()
        batch_size = tf.shape(x)[0]
        return tf.nn.conv2d_transpose(x, w[i], output_shape,
                                      [ 1, strides[i], strides[i], 1], 'SAME')


    rho = tf.constant(1, dtype = tf.float32)

    
    def phi(x, b, l):
        return tf.maximum(x - (tf.abs(b)-l), 0)
    def do_iter(l, z, y, m):
        ytil = y[0] - l[0]/rho
        z[0] = 1/(1+rho)*Wt(sd - mask * W(ytil, 0, tf.shape(sd)), 0) + ytil
        if n > 1:
            y[0] = 1/(rho+1)*phi(rho*z[0] + W(z[1], 1, tf.shape(z[0])), b[0], l[0])
        else:
            y[0] = 1/rho*phi(rho*z[0], b[0], l[0])
        l[0] = l[0] + rho*(z[0] - y[0])
        for i in range(1, n):
            ytil = y[i] - l[i]/rho
            z[i] = 1/(1+rho)*Wt(m[i-1]*y[i-1] - m[i-1] * W(ytil, i, tf.shape(y[i-1])), i) + ytil
                
            if i < n-1:
                y[i] = 1/(rho+1)*phi(rho*z[i] + W(z[i+1], i+1, tf.shape(z[i])), b[i], l[i])
            else:
                y[i] = 1/rho*phi(rho*z[i], b[i], l[i])
            l[i] = l[i] + rho*(z[i] - y[i])
        return l, z, y

    dshape = sd.get_shape().as_list()
    batch_size = tf.shape(sd)[0]
    z = {}
    l = {}
    y = {}

    z[0] = Wt(sd, 0)
    l[0] = tf.zeros(tf.shape(z[0]), dtype = tf.float32)
    y[0] = 1/rho*phi(rho*z[0], b[0], l[0])
    print(z[0].get_shape().as_list())
    for i in range(1, len(filters)):
        z[i] = Wt(m[i-1]*y[i-1], i)
        l[i] = tf.zeros(tf.shape(z[i]), dtype = tf.float32)
        y[i] = 1/rho*phi(rho*z[i], b[i], l[i])
        print(z[i].get_shape().as_list())

    loss_mask = dmask

    rec_errors = [ 0 for i in range(num_iters) ]
    aux_errors = [ 0 for i in range(num_iters) ]
    pred_errors = [ 0 for i in range(num_iters) ]
    masks = [ tf.reduce_mean(m[i]) for i in range(0, n) ]
    for i in range(num_iters):
        l, z ,y = do_iter(l, z, y, m)

        cur_pred = W(z[0], 0, tf.shape(sd))
        rec_err = (tf.reduce_sum(tf.pow(mask*(sd-cur_pred),2))/tf.reduce_sum(mask),)
        aux_error = (tf.reduce_mean(tf.pow(z[0] - y[0], 2)),)
        for j in range(1, n, 3):
            rec_err = rec_err + (tf.reduce_mean(tf.pow(m[j-1]*y[j-1] - m[j-1]*W(z[j], j, tf.shape(y[j-1])), 2)),)
            aux_error = aux_error + (tf.reduce_mean(tf.pow(z[j] - y[j], 2)),)
        rec_errors[i] = rec_err
        #pred_errors[i] = tf.reduce_sum(tf.pow(loss_mask*(d-cur_pred),2))/tf.reduce_sum(loss_mask)
        aux_errors[i] = aux_error
        # errors[i] = (tf.reduce_sum(tf.pow(mask*(sd-cur_pred),2))/tf.reduce_sum(mask),
        #              tf.reduce_sum(tf.pow(loss_mask*(d-cur_pred),2))/tf.reduce_sum(loss_mask),
        #              tf.reduce_sum(tf.pow(mask*(sd-cur_pred),2)) +
        #              rho/2*tf.reduce_sum(tf.pow(z - y, 2)) + tf.reduce_sum(tf.abs(b*y)),
        #              tf.reduce_mean(tf.pow(z - y, 2)))
    
    z[-1] = sd
    pred = W(z[n-1], n-1, tf.shape(z[n-2]))
    for i in range(n-2, -1, -1):
        pred = W(pred, i, tf.shape(z[i-1]))
    
    loss = 0.5*tf.reduce_sum(tf.pow(loss_mask*(d-pred), 2), axis=[1,2,3])
    loss = tf.reduce_mean(loss)
    if tv_loss is not None:
        print('Using TV loss')
        loss = loss + tv_loss*tf.reduce_mean(tf.image.total_variation(pred))
        
    return pred, loss, { 'b' : b }, {'sdmask' : mask, 'm' : m, 'w' : w}, None
