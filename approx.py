from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from kernels import GraphKernel, get_kernel
import networkx as nx

# Abstract approximation class
class GBFApprox(BaseEstimator):
    def __init__(self, G=nx.empty_graph(), 
                 kernel='Trivial', kernel_par=[],
                 verbose=True):
        super().__init__()
        # Define the graph
        if not G:
            print('The graph G is initialized to the default empty graph')
        self.G = G
        
        # Initialize the kernel
        self.kernel = kernel
        self.kernel_par = kernel_par
       
        # Set the verbosity on/off
        self.verbose = verbose
            
    def set_kernel(self):
        # Check if self.kernel is a GraphKernel or a string id
        if isinstance(self.kernel, GraphKernel):
            # In this case we just check the parameter
            if self.kernel_par:
                # If a nonempy parameter is passed, set it
                self.kernel.set_params(self.kernel_par)
            else:
                # Otherwise take the one passed via the kernel
                self.kernel_par = self.kernel.par
        else:
            # Here kernel is a string id
            if self.kernel_par:
                # If a kernel_par is passed, use it
                self.kernel = get_kernel(self.kernel, self.G, self.kernel_par)
            else:
                # Otherwise use the default
                self.kernel = get_kernel(self.kernel, self.G)
                # And then store it
                self.kernel_par = self.kernel.par
        
    @abstractmethod    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pred = self.kernel.eval(X, self.ctrs_) @ self.coef_
        return  np.array(pred)   

    def set_messages(self):
        begin_str =  '\n'  
        begin_str += '\nTraining ' + self.name + ' model with'
        begin_str += '\n\t\t|_ kernel              : %s' % self.kernel
        self.begin_str = begin_str
        
    def print_message(self, when):
        if self.verbose and when == 'begin':
            print(self.begin_str)

    @abstractmethod
    def eval_power_fun(self, X):
        pass

# GBF Interpolation or Regularized Least Squares
class GBFInterpolant(GBFApprox):
    def __init__(self, G=nx.empty_graph(), 
                 kernel='Trivial', kernel_par=[],
                 reg_par=0,
                 verbose=True):
        super(GBFInterpolant, self).__init__(G, kernel, kernel_par, verbose)
        self.name = 'Kernel Interpolation'
        self.reg_par = reg_par
        if self.verbose:
            self.set_messages()
     
    def fit(self, X, y):
        self.set_kernel()
        self.ctrs_ = X
        self.print_message('begin')
        # Assembly the kernel matrix
        A = self.kernel.eval(X, X)
        # Solve the linear system            
        self.coef_ = np.linalg.solve(A + self.reg_par * np.eye(len(X)), y)
        return self

    def set_messages(self):
        super(GBFInterpolant, self).set_messages()
        if self.verbose:
            reg_info = '\n\t\t|_ regularization par. : %2.2e' % self.reg_par
            self.begin_str += reg_info
        
    def eval_power_fun(self, X):
        k_X = self.kernel.eval(X, self.ctrs_)
        A = self.kernel.eval(self.ctrs_, self.ctrs_)
        p2 = self.kernel.diagonal(X)  - np.diag(k_X @ np.linalg.solve(A, k_X.transpose()))
        return np.sqrt(np.abs(p2))
    

# Interpolation or Reg Least Suqares with greedy selection of the points
class GBFGreedy(GBFApprox):    
    def __init__(self, G=nx.empty_graph(), 
                 kernel='Trivial', kernel_par=[],
                 verbose=True, n_report=10, 
                 greedy_type='p_greedy', reg_par=0, restr_par=0, 
                 tol_f=1e-10, tol_p=1e-10, max_iter=10):
        
        super(GBFGreedy, self).__init__(G, kernel, kernel_par, verbose)

        # Set the frequency of report
        self.n_report = n_report
        
        # Set the params defining the method
        self.greedy_type = greedy_type
        self.reg_par = reg_par
        self.restr_par = restr_par
        
        # Set the stopping values
        self.max_iter = max_iter
        self.tol_f = tol_f
        self.tol_p = tol_p
        
    def selection_rule(self, f, p):
        if self.restr_par > 0:
            p_ = np.max(p)
            restr_idx = np.nonzero(p >= self.restr_par * p_)[0]
        else:
            restr_idx = np.arange(len(p))

        f = np.sum(f ** 2, axis=1)
        if self.greedy_type == 'f_greedy':
            idx = np.argmax(f[restr_idx])
            idx = restr_idx[idx]
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'fp_greedy':
            idx = np.argmax(f[restr_idx] / p[restr_idx])
            idx = restr_idx[idx]
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'p_greedy':
            f_max = np.max(f)
            idx = np.argmax(p)
            p_max = p[idx]
        return idx, f_max, p_max

    def fit(self, X, y):
        self.set_kernel()

        ### TODO: Replace with np.atleast_1d(X.squeeze()
        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        if len(y.shape) == 1:
            y = np.atleast_2d(y)
        if X.shape[0] == 1:
            X = X.transpose()
        if y.shape[0] == 1:
            y = y.transpose()
        
        y = y.ravel()
        
        # Check the dataset
        X, y = check_X_y(X, y, multi_output=False)
        
        # Initialize the convergence history (cold start)
        self.train_hist = {}
        self.train_hist['n'] = []
        self.train_hist['f'] = []
        self.train_hist['p'] = []
        
        # Initialize the residual
        y = np.array(y)
        if len(y.shape) == 1:
            y = y[:, None]
        
        # Get the data dimension        
        N, q = y.shape

        self.max_iter = min(self.max_iter, N) 
        
        self.kernel.set_params(self.kernel_par)
        
        # Check compatibility of restriction
        if self.greedy_type == 'p_greedy':
            self.restr_par = 0
        if not self.reg_par == 0:
            self.restr_par = 0
        
        self.indI_  = []
        notIndI = list(range(N))
        Vx = np.zeros((N, self.max_iter))
        if q > 1:
            c = np.zeros((self.max_iter, q))
        else:
            c = np.zeros(self.max_iter)
            
            
        p = self.kernel.diagonal(X) + self.reg_par
        
        self.Cut_ = np.zeros((self.max_iter, self.max_iter))       
        
        self.print_message('begin')
        # Iterative selection of new points
        for n in range(self.max_iter):
            # prepare
            self.train_hist['n'].append(n+1)
            self.train_hist['f'].append([])
            self.train_hist['p'].append([])
            # select the current index
            idx, self.train_hist['f'][n], self.train_hist['p'][n] = self.selection_rule(y[notIndI], p[notIndI])
            # add the current index
            self.indI_ .append(notIndI[idx])
            # check if the tolerances are reacheded
            if self.train_hist['f'][n] <= self.tol_f:
                n = n - 1
                self.train_hist['p'] = self.train_hist['p'][:-1]
                self.train_hist['f'] = self.train_hist['f'][:-1]
                self.print_message('end')   
                break
            if self.train_hist['p'][n] <= self.tol_p:
                n = n - 1
                self.train_hist['p'] = self.train_hist['p'][:-1]
                self.train_hist['f'] = self.train_hist['f'][:-1]
                self.print_message('end')   
                break
            # compute the nth basis
            Vx[notIndI, n] = self.kernel.eval(X[notIndI, :], X[self.indI_[n],:])[:, 0] - \
                Vx[notIndI, :n+1] @ Vx[self.indI_ [n], 0:n+1].transpose()
            Vx[self.indI_ [n], n] += self.reg_par
            # normalize the nth basis
            Vx[notIndI, n] = Vx[notIndI, n] / np.sqrt(p[self.indI_ [n]])
            # update the change of basis
            Cut_new_row = np.ones(n + 1)
            Cut_new_row[:n] = -Vx[self.indI_ [n], :n] @ self.Cut_[:n:, :n]
            self.Cut_[n, :n+1] = Cut_new_row / Vx[self.indI_ [n], n]      
            # compute the nth coefficient
            c[n] = y[self.indI_ [n]] / np.sqrt(p[self.indI_ [n]])
            # update the power function
            p[notIndI] = p[notIndI] - Vx[notIndI, n] ** 2
            # update the residual
            y[notIndI] = y[notIndI] - Vx[notIndI, n][:, None] * c[n]
            # remove the nth index from the dictionary
            notIndI.pop(idx)
            
            # Report some data every now and then
            if n % self.n_report == 0:
                self.print_message('track')              

        else:
            self.print_message('end')              

        # Define coefficients and centers
        c = c[:n+1]
        self.Cut_ = self.Cut_[:n+1, :n+1]
        self.indI_  = self.indI_ [:n+1]
        self.coef_ = self.Cut_.transpose() @ c
        self.ctrs_ = X[self.indI_ , :]

        return self


    def predict(self, X):
        # Check is fit has been called
        check_is_fitted(self, 'coef_')

        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.transpose()
            
        # Validate the input
        X = check_array(X)
   
        # Evaluate the model
        pred = self.kernel.eval(X, self.ctrs_) @ self.coef_

        return pred

    def print_message(self, when):
        
        if self.verbose and when == 'begin':
            print('')
            print('*' * 30 + ' [VKOGA] ' + '*' * 30)
            print('Training model with')
            print('       |_ kernel              : %s' % self.kernel)
            print('       |_ regularization par. : %2.2e' % self.reg_par)
            print('       |_ restriction par.    : %2.2e' % self.restr_par)
            print('')
            
        if self.verbose and when == 'end':
            print('Training completed with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))
                        
        if self.verbose and when == 'track':
            print('Training ongoing with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))  

    def eval_power_fun(self, X):
        p = np.sqrt(np.abs(self.kernel.diagonal(X) 
                           - np.sum((self.kernel.eval(X, np.atleast_2d(self.ctrs_)) 
                                     @ self.Cut_.transpose()) ** 2, axis=1)))
        return p


# Interpolation or Reg Least Suqares with greedy selection of the points
class GBFPointSelection(GBFApprox):    
    def __init__(self, G=nx.empty_graph(), 
                 kernel='Trivial', kernel_par=[],
                 verbose=True, n_report=10, 
                 greedy_type='p_greedy', reg_par=0, restr_par=0, 
                 tol_f=1e-10, tol_p=1e-10, max_iter=10):
        
        super(GBFPointSelection, self).__init__(G, kernel, kernel_par, verbose)

        # Set the frequency of report
        self.n_report = n_report
        
        # Set the params defining the method
        self.greedy_type = greedy_type
        self.reg_par = reg_par
        self.restr_par = restr_par
        
        # Set the stopping values
        self.max_iter = max_iter
        self.tol_f = tol_f
        self.tol_p = tol_p
        
    def selection_rule(self, f, p):
        if self.restr_par > 0:
            p_ = np.max(p)
            restr_idx = np.nonzero(p >= self.restr_par * p_)[0]
        else:
            restr_idx = np.arange(len(p))

        f = np.sum(f ** 2, axis=1)
        if self.greedy_type == 'f_greedy':
            idx = np.argmax(f[restr_idx])
            idx = restr_idx[idx]
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'fp_greedy':
            idx = np.argmax(f[restr_idx] / p[restr_idx])
            idx = restr_idx[idx]
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'p_greedy':
            f_max = np.max(f)
            idx = np.argmax(p)
            p_max = p[idx]
        return idx, f_max, p_max

    def fit(self, X, y):
        self.set_kernel()

        ### TODO: Replace with np.atleast_1d(X.squeeze()
        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        if len(y.shape) == 1:
            y = np.atleast_2d(y)
        if X.shape[0] == 1:
            X = X.transpose()
        if y.shape[0] == 1:
            y = y.transpose()
        
        y = y.ravel()
        
        # Check the dataset
        X, y = check_X_y(X, y, multi_output=False)
        
        # Initialize the convergence history (cold start)
        self.train_hist = {}
        self.train_hist['n'] = []
        self.train_hist['f'] = []
        self.train_hist['p'] = []
        
        # Initialize the residual
        y = np.array(y)
        if len(y.shape) == 1:
            y = y[:, None]
        
        # Get the data dimension        
        N, q = y.shape

        self.max_iter = min(self.max_iter, N) 
        
        self.kernel.set_params(self.kernel_par)
        
        # Check compatibility of restriction
        if self.greedy_type == 'p_greedy':
            self.restr_par = 0
        if not self.reg_par == 0:
            self.restr_par = 0
        
        self.indI_  = []
        notIndI = list(range(N))
        Vx = np.zeros((N, self.max_iter))
        if q > 1:
            c = np.zeros((self.max_iter, q))
        else:
            c = np.zeros(self.max_iter)
            
            
        p = self.kernel.diagonal(X) + self.reg_par
        
        self.Cut_ = np.zeros((self.max_iter, self.max_iter))       
        
        self.print_message('begin')
        # Iterative selection of new points
        for n in range(self.max_iter):
            # prepare
            self.train_hist['n'].append(n+1)
            self.train_hist['f'].append([])
            self.train_hist['p'].append([])
            # select the current index
            idx, self.train_hist['f'][n], self.train_hist['p'][n] = self.selection_rule(y[notIndI], p[notIndI])
            # add the current index
            self.indI_ .append(notIndI[idx])
            # check if the tolerances are reacheded
            if self.train_hist['f'][n] <= self.tol_f:
                n = n - 1
                self.train_hist['p'] = self.train_hist['p'][:-1]
                self.train_hist['f'] = self.train_hist['f'][:-1]
                self.print_message('end')   
                break
            if self.train_hist['p'][n] <= self.tol_p:
                n = n - 1
                self.train_hist['p'] = self.train_hist['p'][:-1]
                self.train_hist['f'] = self.train_hist['f'][:-1]
                self.print_message('end')   
                break
            # compute the nth basis
            Vx[notIndI, n] = self.kernel.eval(X[notIndI, :], X[self.indI_[n],:])[:, 0] - \
                Vx[notIndI, :n+1] @ Vx[self.indI_ [n], 0:n+1].transpose()
            Vx[self.indI_ [n], n] += self.reg_par
            # normalize the nth basis
            Vx[notIndI, n] = Vx[notIndI, n] / np.sqrt(p[self.indI_ [n]])
            # update the change of basis
            Cut_new_row = np.ones(n + 1)
            Cut_new_row[:n] = -Vx[self.indI_ [n], :n] @ self.Cut_[:n:, :n]
            self.Cut_[n, :n+1] = Cut_new_row / Vx[self.indI_ [n], n]      
            # compute the nth coefficient
            c[n] = y[self.indI_ [n]] / np.sqrt(p[self.indI_ [n]])
            # update the power function
            p[notIndI] = p[notIndI] - Vx[notIndI, n] ** 2
            # update the residual
            y[notIndI] = y[notIndI] - Vx[notIndI, n][:, None] * c[n]
            # remove the nth index from the dictionary
            notIndI.pop(idx)
            
            # Report some data every now and then
            if n % self.n_report == 0:
                self.print_message('track')              

        else:
            self.print_message('end')              

        # Define coefficients and centers
        c = c[:n+1]
        self.Cut_ = self.Cut_[:n+1, :n+1]
        self.indI_  = self.indI_ [:n+1]
        self.coef_ = self.Cut_.transpose() @ c
        self.ctrs_ = X[self.indI_ , :]

        return self


    def predict(self, X):
        # Check is fit has been called
        check_is_fitted(self, 'coef_')

        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.transpose()
            
        # Validate the input
        X = check_array(X)
   
        # Evaluate the model
        pred = self.eval_power_fun(X)
        
        pred /= np.sqrt(self.kernel.diagonal(X))
        
        return pred

    def print_message(self, when):
        
        if self.verbose and when == 'begin':
            print('')
            print('*' * 30 + ' [VKOGA] ' + '*' * 30)
            print('Training model with')
            print('       |_ kernel              : %s' % self.kernel)
            print('       |_ regularization par. : %2.2e' % self.reg_par)
            print('       |_ restriction par.    : %2.2e' % self.restr_par)
            print('')
            
        if self.verbose and when == 'end':
            print('Training completed with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))
                        
        if self.verbose and when == 'track':
            print('Training ongoing with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))  

    def eval_power_fun(self, X):
        p = np.sqrt(np.abs(self.kernel.diagonal(X) 
                           - np.sum((self.kernel.eval(X, np.atleast_2d(self.ctrs_)) 
                                     @ self.Cut_.transpose()) ** 2, axis=1)))
        return p
