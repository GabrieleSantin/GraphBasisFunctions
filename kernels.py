from abc import ABC, abstractmethod
import numpy as np
from utils import graph_spectrum
import networkx as nx

# Abstract kernel on a graph
class GraphKernel(ABC):
    @abstractmethod    
    def __init__(self, G=nx.empty_graph(), par=[]):
        super().__init__()
        self.G = G
        if not isinstance(par, list):
            self.par = [par]
        else:
            self.par = par
        
    @abstractmethod
    def eval(self):
        pass

    def eval_prod(self, idx_x, idx_y, v, batch_size=100):
        N = len(idx_x)
        batch_size = np.min([N, batch_size])
        
        num_batches = int(np.ceil(N / batch_size))
        mat_vec_prod = np.zeros((N, 1)) 
        for idx in range(num_batches):
            idx_begin = int(idx * batch_size)
            idx_end = int((idx + 1) * batch_size)
            A = self.eval(np.arange(idx_begin, idx_end, dtype=int), idx_y)
            mat_vec_prod[idx_begin:idx_end] = A @ v
        return mat_vec_prod

    def diagonal(self, idx):
        diag = self.eval(idx, idx).diagonal() 
        return diag

    def __str__(self):
        return self.get_name()

    def set_params(self, par):
        if not isinstance(par, list):
            par = [par]
        self.par = par
        return self


# Abstract GBF
class GBF(GraphKernel):
    @abstractmethod    
    def __init__(self, G=nx.empty_graph(), par=[], size_threshold=3000):
        super(GBF, self).__init__(G, par)
        size_message = 'The graph is too large - Consider switching to an approximated kernel'
        assert len(G) <= size_threshold, size_message
        self.U, self.L = graph_spectrum(self.G)
        
    def eval(self, idx_x, idx_y):
        idx_x = np.atleast_1d(np.squeeze(idx_x))
        idx_y = np.atleast_1d(np.squeeze(idx_y))
        A = np.array(self.U[idx_x, :] @ np.diag(self.f) @ self.U[idx_y, :].transpose())
        return A


# Implementation of concrete GBFs
class VarSpline(GBF):
    def __init__(self, G=nx.empty_graph(), par=[1, 0]):
        super(VarSpline, self).__init__(G, par)
        self.f = (self.par[1] + self.L) ** self.par[0]
        self.f[np.abs(self.f) >= 1e12] = 0

    def get_name(self):
        name = 'Variational spline: f = (%2.2f + lambda) ** %2.2f'
        return name % (self.par[1], self.par[0])
        

class Diffusion(GBF):
    def __init__(self, G=nx.empty_graph(), par=[-10]):
        super(Diffusion, self).__init__(G)
        self.set_params(par)
        self.f = np.exp(self.par[0] * self.L)

    def get_name(self):
        name = 'Diffusion: f = exp(%2.2f * lambda)'
        return name % self.par[0]


class PolyDecay(GBF):
    def __init__(self, G=nx.empty_graph(), par=[1, 1]):
        super(PolyDecay, self).__init__(G, par)
        self.f = (1 + self.par[1] * np.arange(len(G))) ** self.par[0]

    def get_name(self):
        name = 'PolyDecay: f = (1 + %2.2f * (0, 1, ..., len(G)-1) ** %2.2f'
        return name % (self.par[1], self.par[0])
    

class BandLimited(GBF):
    def __init__(self, G=nx.empty_graph(), par=[1]):
        super(BandLimited, self).__init__(G, par)
        self.f = np.r_[np.ones(self.par[0]), np.zeros(len(G) - self.par[0])]

    def get_name(self):
        name = 'BandLimited: f = [1, ..., 1, 0, ..., 0] (%2d ones)'
        return name % self.par[0]


class Trivial(GBF):
    def __init__(self, G=nx.empty_graph(), par=[]):
        super(Trivial, self).__init__(G, par)
        self.f = np.ones(len(G))

    def get_name(self):
        name = 'Trivial: f = [1, ..., 1]'
        return name
    

#%%
def get_kernel(kernel_id, G=nx.empty_graph(), par=[]):
    kernel_id = kernel_id.lower()
    if par:
        if kernel_id == 'VarSpline'.lower():
        	kernel = VarSpline(G, par)
        if kernel_id == 'Diffusion'.lower():
        	kernel = Diffusion(G, par)
        if kernel_id == 'PolyDecay'.lower():
        	kernel = PolyDecay(G, par)
        if kernel_id == 'BandLimited'.lower():
        	kernel = BandLimited(G, par)
        if kernel_id == 'Trivial'.lower():
        	kernel = Trivial(G, par)
    else:
        if kernel_id == 'VarSpline'.lower():
        	kernel = VarSpline(G)
        if kernel_id == 'Diffusion'.lower():
        	kernel = Diffusion(G)
        if kernel_id == 'PolyDecay'.lower():
        	kernel = PolyDecay(G)
        if kernel_id == 'BandLimited'.lower():
        	kernel = BandLimited(G)
        if kernel_id == 'Trivial'.lower():
        	kernel = Trivial(G)
    return kernel

        