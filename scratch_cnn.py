# scratch_cnn.py
import numpy as np

class ConvLayer:
    def __init__(self, f, n_c, stride=1, pad=0):
        self.f = f
        self.stride = stride
        self.pad = pad
        self.weights = np.random.randn(f, f, n_c) * 0.01
        self.biases = np.zeros((f, f, 1))

    def zero_pad(self, X, pad):
        return np.pad(X, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))

    def conv_single_step(self, a_slice, W, b):
        s = a_slice * W
        Z = np.sum(s)
        Z = float(Z + b)
        return Z

    def forward(self, A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        n_H = int((n_H_prev - self.f + 2 * self.pad) / self.stride) + 1
        n_W = int((n_W_prev - self.f + 2 * self.pad) / self.stride) + 1
        n_C = self.weights.shape[3]

        Z = np.zeros((m, n_H, n_W, n_C))

        A_prev_pad = self.zero_pad(A_prev, self.pad)

        for i in range(m):     
            a_prev_pad = A_prev_pad[i]
            for h in range(n_H):       
                for w in range(n_W):   
                    for c in range(n_C):   
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, self.weights[..., c], self.biases[..., c])

        assert(Z.shape == (m, n_H, n_W, n_C))

        self.cache = (A_prev, self.weights, self.biases, self.stride, self.pad)

        return Z


    def backward(self, dZ):
        (A_prev, W, b, stride, pad) = self.cache
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (m, n_H, n_W, n_C) = dZ.shape
        
        dA_prev = np.zeros_like(A_prev)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * stride
                        vert_end = vert_start + self.f
                        horiz_start = w * stride
                        horiz_end = horiz_start + self.f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, dW, db
    
    def update(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
