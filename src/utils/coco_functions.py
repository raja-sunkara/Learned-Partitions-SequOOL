#####-------------------------------- New set of test function from COCO--------------------- #####
import numpy as np
import scipy.stats as scistats
import pandas as pd
from HesBO.BLOSSOM.embd_functions import Hartmann6

np.seterr(divide='ignore', invalid='ignore')

class BaseObjective:
    def __init__(self, d, rng, sub_space_dim, int_opt = None, alpha = None, beta = None):
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.rng = rng

        self.lb = int_opt[0] * np.ones((1, d))
        self.ub = int_opt[1] * np.ones((1, d))
        self.sub_space_dim = sub_space_dim
        # check if optimal parameters is in interval
#        if int_opt is not None:
#            self.x_opt = rng.uniform(int_opt[0], int_opt[1], size=(1, d))
        # or based on a single value
#        elif val_opt is not None:
#            self.one_pm = np.where(rng.rand(1, d) > 0.5, 1, -1)
#            self.x_opt = val_opt * self.one_pm
#        else:
#            raise ValueError("Optimal value or interval has to be defined")
        #self.f_opt = np.round(np.clip(scistats.cauchy.rvs(loc=0, scale=100, size=1)[0], -1000, 1000), decimals=2)
        
        self.i = np.arange(self.d)
        self._lambda_alpha = None
        self._q = None
        self._r = None
        
        
        
        self.f_opt = np.round(np.clip(self.rng.standard_cauchy() * 100, -1000, 1000), decimals=2)



        self.x_opt = self._generate_valid_x_opt()



    def _generate_valid_x_opt(self):
        while True:
            x_opt = self.rng.uniform(self.lb, self.ub)
            projection_matrix = self.r[:self.sub_space_dim].T @ self.r[:self.sub_space_dim]
            projected_x_opt =projection_matrix @ x_opt.T
            if np.all(projected_x_opt >= self.lb.T) and np.all(projected_x_opt <= self.ub.T):
                return x_opt
        
    def __call__(self, x):
        return self.evaluate_full(x)

    def evaluate_full(self, x):
        raise NotImplementedError("Subclasses should implement this!")
    
        # TODO: property probably unnecessary
    @property
    def q(self):
        if self._q is None:
            a = self.rng.standard_normal((self.d, self.d))
            
            # TODO: correct way of doing Gram Schmidt ortho-normalization?
            q, _ = np.linalg.qr(a)
            #q = np.eye(self.d)
            self._q = q
        return self._q

    @property
    def r(self):
        if self._r is None:
            a = self.rng.standard_normal((self.d, self.d))
            # TODO: correct way of doing Gram Schmidt ortho-normalization?
            r, _ = np.linalg.qr(a)
            #r = np.eye(self.d)
            self._r = r
        return self._r
    
    @r.setter
    def r(self, value):
        self._r = value

    @property
    def lambda_alpha(self):
        if self._lambda_alpha is None:
            if isinstance(self.alpha, int):
                lambda_ii = np.power(self.alpha,  1/2 * self.i / (self.d - 1))
                self._lambda_alpha = np.diag(lambda_ii)
            else:
                lambda_ii = np.power(self.alpha[:, None],  1/2 * self.i[None, :] / (self.d - 1))
                self._lambda_alpha = np.stack([np.diag(l_ii) for l_ii in lambda_ii])
        return self._lambda_alpha

    @staticmethod
    def f_pen(x):
        return np.sum(np.maximum(0, np.abs(x) - 5), axis=1)

    def t_asy_beta(self, x):
        # this is a map from \mathrm{R}^d to \mathrm{R}^d. when there is 
        # low subspace, need to use subspace dimension
        #exp = np.power(x, 1 + self.beta * self.i[:, None] / (self.d - 1) * np.sqrt(x))
        indices = np.arange(self.sub_space_dim)
        exp = np.power(x, 1 + self.beta * indices[:, None] / (self.sub_space_dim- 1) * np.sqrt(x))
        return np.where(x > 0, exp, x)

    def t_osz(self, x):
        x_hat = np.where(x != 0, np.log(np.abs(x)), 0)
        c_1 = np.where(x > 0, 10, 5.5)
        c_2 = np.where(x > 0, 7.9, 3.1)
        return np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c_1 * x_hat) + np.sin(c_2 * x_hat)))




class Custom_f1(BaseObjective):
    def __init__(self, d, rng, int_opt = (-5., 5.),sub_space_dim=5):
        super(Custom_f1, self).__init__(d, rng,int_opt=int_opt,sub_space_dim=sub_space_dim)
        self.x_opt = 1 * np.ones((1, d))
        self.f_opt = 0.0
        self.r = np.vstack((np.eye(d)[0], np.eye(d)[-1]))
        
    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d
        
        z = (x[:,0]-1)**2 + (x[:,-self.sub_space_dim]-1)**4
        
        return z
    
class SphereRotated(BaseObjective):
    def __init__(self, d,rng, int_opt = (-5., 5.),sub_space_dim=5):
        super(SphereRotated, self).__init__(d, rng,int_opt=int_opt, sub_space_dim=sub_space_dim)

    
    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d
        
        z = (self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        return np.linalg.norm(z, axis=1)**2 + self.f_opt
 

# this function is not coming from the pdf, so it is only for the 2d case. 
# TODO will fix it later  
class BraninRotated(BaseObjective):
    def __init__(self, d, rng, int_opt=(-5., 15.),sub_space_dim=5):
        super(BraninRotated, self).__init__(d, rng, sub_space_dim=sub_space_dim,int_opt=int_opt)
        self.alpha_opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.x_opt = self.r[:self.sub_space_dim].T @ self.alpha_opt.T
        
        if (self.x_opt <= self.ub.T).all().all() and (self.x_opt >= self.lb.T).all().all():
            self.f_opt = self.evaluate_full(self.x_opt.T)[0, 0]  # Global optimum
        self.f_opt = self.evaluate_full(self.x_opt.T)[0, 0]
        #self.f_opt = 0.39788735772973816
        self.x_opt = self.x_opt.T
    def evaluate_full(self, x):
        projected_point = self.r[:self.sub_space_dim] @ x.T
        
        a = 1.0
        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
    
        x1, x2 = projected_point[0], projected_point[1]
        return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s).reshape(-1,1)
  
  
class EllipsoidRotated(BaseObjective):
    def __init__(self, d, rng, int_opt = (-5., 5.),sub_space_dim=5):
        super(EllipsoidRotated, self).__init__(d, rng, sub_space_dim=sub_space_dim, int_opt=int_opt)
        indices = np.arange(self.sub_space_dim)
        self.c = np.power(1e2, indices / (self.sub_space_dim - 1))
        
    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d
        
        z = self.t_osz(self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        #z = (self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        #z = (self.r[2:2+self.sub_space_dim] @ (x - self.x_opt).T).T
        return np.sum(self.c * z**2, axis=1) + self.f_opt
    

class RastriginRotated(BaseObjective):
    def __init__(self, d, rng, int_opt=(-1., 1.), alpha=10, beta=0.2,sub_space_dim=5):
        super(RastriginRotated, self).__init__(d, rng, sub_space_dim = sub_space_dim, int_opt = int_opt, alpha=alpha, beta=beta)
       
        self.mat_fac = (self.r @ self.lambda_alpha @ self.q)[:self.sub_space_dim,:self.sub_space_dim]

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        # TODO: maybe flip data dimensions?
        #z = (self.mat_fac @ self.t_asy_beta(self.t_osz(self.r[:self.sub_space_dim] @ (x - self.x_opt).T))).T
        z = (self.t_asy_beta(self.t_osz(self.r[:self.sub_space_dim] @ (x - self.x_opt).T))).T

        #out = 10 * (self.d - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1)**2 + self.f_opt
        
        out = 10 * (self.sub_space_dim - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1)**2 + self.f_opt

        return out
    
class StyblinskiTang(BaseObjective):
    """
    The Styblinski-Tang Function. See https://www.sfu.ca/~ssurjano/stybtang.html for details.
    """
    def __init__(self,d,rng,int_opt = (-5.,5.),sub_space_dim=5):
        super(StyblinskiTang, self).__init__(d, rng, sub_space_dim=sub_space_dim, int_opt=int_opt)
        
        self.alpha_star = np.array([[-2.903534 for _ in range(sub_space_dim)]]).T  # Global minimiser
        
        self.x_opt = self.r[:self.sub_space_dim].T @ self.alpha_star   # (d,1) Global minimiser in original space
        # check if this point is in the interval:
        if (self.x_opt <= self.ub).all().all() and (self.x_opt >= self.lb).all().all():
            self.f_opt = self.evaluate_full(self.x_opt.T)[0, 0]  # Global optimum
            
        self.x_opt = self.x_opt.T
    def evaluate_full(self, x) -> np.ndarray:
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d
        #assert (x <= self.ub).all().all()
        #assert (x >= self.lb).all().all()

        #x = x.to_numpy().astype(float)
        
        
        z = (self.r[:self.sub_space_dim] @ x.T).T  # (num_points,M)

        return (z ** 4 - 16 * z ** 2 + 5 * z).sum(axis=-1).reshape(-1, 1) / 2
    
# Hartmann 6D function
class Hartmann6(BaseObjective):
    def __init__(self,d,rng,int_opt = (0.,1.),sub_space_dim=5):
        super(Hartmann6, self).__init__(d, rng, sub_space_dim=sub_space_dim, int_opt=int_opt)
        assert sub_space_dim == 6, "Hartmann6 is a 6-dimensional function, choose sub_space_dim=6"
        self.alpha_star = np.array([[0.20169, 0.150011, 0.476874, 
                                     0.275332, 0.311652, 0.6573]]).T  # Global minimiser
        
        self.x_opt = self.r[:self.sub_space_dim].T @ self.alpha_star   # (d,1) Global minimiser in original space
        # check if this point is in the interval:
        
            
            
        self.A = np.array(
                [[10, 3, 17, 3.50, 1.7, 8], 
                 [0.05, 10, 17, 0.1, 8, 14], 
                 [3, 3.5, 1.7, 10, 17, 8], 
                 [17, 8, 0.05, 10, 0.1, 14]])

        self.P = (
            np.array(
        [[1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],])/ 10000.0)

        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        
        if (self.x_opt <= self.ub).all().all() and (self.x_opt >= self.lb).all().all():
            self.f_opt = self.evaluate_full(self.x_opt.T)[0, 0]  # Global optimum
        
        self.x_opt = self.x_opt.T 
    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d
        
        z = (self.r[:self.sub_space_dim] @ x.T).T  # (num_points,6)
        
        
        # Compute the inner sum for all points simultaneously
        #inner_sum = np.sum(self.A[:, np.newaxis, :] * (z[:, np.newaxis, :] - self.P[:, np.newaxis, :])**2, axis=2)
        
        #inner_sum = np.diagonal(((z[:, np.newaxis, :] - self.P) @ self.A.T),axis1=1,axis2=2)
        
        #inner_sum = np.sum(self.A[:, np.newaxis, :] * (z[:, np.newaxis, :] - self.P)**2, axis=2)
        
        inner_sum = np.sum(self.A * (z[:, np.newaxis, :] - self.P)**2, axis=2)
        
        # Compute the final result for all points
        result = -np.sum(self.alpha * np.exp(-inner_sum), axis=1)
        
        return result.reshape(-1, 1)
        
        
        
        
    
    
    
    
    
class Rosenbrock(BaseObjective):
    def __init__(self, d, rng, int_opt=(-1., 1.),sub_space_dim = 5):
        super(Rosenbrock, self).__init__(d, rng, sub_space_dim=sub_space_dim,
                                                int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.d) / 8)


    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        #z = (self.c * self.r @ x.T + 1/2).T
        
        
        z = self.c*(self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T + 1
        a = z[:, :-1]**2 - z[:, 1:]
        b = z[:, :-1] - 1

        out = np.sum(100 * a**2 + b**2, axis=1) + self.f_opt

        return out
    
class SharpRidge(BaseObjective):
    def __init__(self, d,rng, int_opt=(-5., 5.), alpha=10,sub_space_dim = 5):
        super(SharpRidge, self).__init__(d, rng, int_opt=int_opt, sub_space_dim=sub_space_dim, alpha=alpha)
       

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        #z = (self.q[:,:self.sub_space_dim] @ self.lambda_alpha[:self.sub_space_dim,:self.sub_space_dim] @ self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        z = (self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        out = z[:, 0] ** 2 + 10 * np.sqrt(np.sum(z[:, 1:] ** 2, axis=1)) + self.f_opt
        return out   

    
class LinearSlope(BaseObjective):
    def __init__(self, d, val_opt=5):
        super(LinearSlope, self).__init__(d, val_opt=val_opt)
        self.c = np.power(10, self.i / (self.d - 1))



    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = np.where(self.x_opt * x < 25, x, self.x_opt)
        s = np.sign(self.x_opt) * self.c
        # TODO: check implementation for boundary errors
        return np.sum(5 * np.abs(s) - s * z, axis=1) + self.f_opt
    
           
# Functions with low or moderate conditioning    
class AttractiveSector(BaseObjective):
    def __init__(self, d, rng, int_opt=(-5., 5.), alpha=10, sub_space_dim=5):
        super(AttractiveSector, self).__init__(d,rng, sub_space_dim=sub_space_dim, int_opt=int_opt, alpha=alpha)
        #self.mat_fac = self.q @ self.lambda_alpha @ self.r



    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        s = 100 #np.where(z * self.x_opt > 0, 100, 1)
        out = self.t_osz(np.sum((s * z)**2, axis=1))**0.9 + self.f_opt
        return out
    
class StepEllipsoid(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10):
        super(StepEllipsoid, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.mat_fac = self.lambda_alpha @ self.r


    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z_hat = (self.mat_fac @ (x - self.x_opt).T)
        z_tilde = np.where(np.abs(z_hat) > 0.5, np.floor(0.5 + z_hat), np.floor(0.5 + 10 * z_hat) / 10)
        z = (self.q @ z_tilde).T

        out = 0.1 * np.maximum(np.abs(z_hat[0, :]) / 1e4,
                               np.sum(np.power(100, self.i / (self.d - 1)) * z**2, axis=1)
                               ) + self.f_pen(x) + self.f_opt
        return out
    

    
#3 Functions with high conditioning and unimodal
class Discus(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(Discus, self).__init__(d, int_opt)



    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = self.t_osz(self.r @ (x - self.x_opt).T).T

        return 1.e6 * z[:, 0]**2 + np.sum(z[:, 1:]**2, axis=1) + self.f_opt
    
class BentCigar(BaseObjective):
    def __init__(self, d, rng, int_opt=(-5., 5.), sub_space_dim=5,beta=0.5):
        super(BentCigar, self).__init__(d, rng,sub_space_dim=sub_space_dim,int_opt=int_opt, beta=beta)


    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        # FIXME: is this correct?
        z = (self.r @ self.t_asy_beta(self.r @ (x - self.x_opt).T)).T

        return z[:, 0]**2 + 1e6 * np.sum(z[:, 1:]**2, axis=1) + self.f_opt
    

    
class DifferentPowers(BaseObjective):
    def __init__(self, d, rng, int_opt=(-5., 5.),sub_space_dim=5):
        super(DifferentPowers, self).__init__(d, rng, sub_space_dim = sub_space_dim,int_opt=int_opt)
        self.indices = np.arange(self.sub_space_dim)



    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        #R = np.eye(self.d)[:self.sub_space_dim]
        z = (self.r[:self.sub_space_dim] @ (x - self.x_opt).T).T
        #z = (R @(x - self.x_opt).T).T        
        out = np.sqrt(np.sum(np.power(np.abs(z), 2 + 4 * self.indices / (self.sub_space_dim - 1)), axis=1)) + self.f_opt
        
        #out = np.sqrt(np.sum(np.power(np.abs(z), 1 + 2 * self.indices / (self.sub_space_dim - 1)), axis=1)) + self.f_opt
        return out
    
# 4: Multi-modal functions with adequate global structure
# class RastriginRotated(BaseObjective):
#     def __init__(self, d, int_opt=(-5., 5.), alpha=10, beta=0.2):
#         super(RastriginRotated, self).__init__(d, int_opt, alpha=alpha, beta=beta)
#         self.mat_fac = self.r @ self.lambda_alpha @ self.q

#     def evaluate_full(self, x):
#         x = np.atleast_2d(x)
#         assert x.shape[1] == self.d

#         # TODO: maybe flip data dimensions?
#         z = (self.mat_fac @ self.t_asy_beta(self.t_osz(self.r @ (x - self.x_opt).T))).T

#         out = 10 * (self.d - np.sum(np.cos(2 * np.pi * z), axis=1)) + np.linalg.norm(z, axis=1)**2 + self.f_opt

#         return out

class Weierstrass(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=0.01):
        super(Weierstrass, self).__init__(d, int_opt=int_opt, alpha=alpha)
        self.mat_fac = self.r @ self.lambda_alpha @ self.q
        self.k = np.arange(12)
        self.f_0 = np.sum(1 / (2 ** self.k) * np.cos(2 * np.pi * 3**self.k * 1/2))  # maybe remove 2 * 1/2

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.mat_fac @ self.t_osz(self.r @ (x - self.x_opt).T)).T

        sum_k = np.zeros_like(z)
        for k in self.k:
            sum_k += 1 / 2 ** k * np.cos(2 * np.pi * 3 ** k * (z + 1 / 2))

        out = 10 * (1 / self.d * np.sum(sum_k, axis=1) - self.f_0) ** 3 + 10/self.d * self.f_pen(x) + self.f_opt
        return out
    
class Schaffers(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.), alpha=10, beta=0.5):
        super(Schaffers, self).__init__(d, int_opt=int_opt, alpha=alpha, beta=beta)
        self.mat_fac = self.lambda_alpha @ self.q



    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.mat_fac @ self.t_asy_beta(self.r @ (x - self.x_opt).T)).T

        s = np.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)

        out = (1 / (self.d - 1) * np.sum(np.sqrt(s) + np.sqrt(s) * np.sin(50 * s ** 0.2) ** 2, axis=1)) ** 2 \
            + 10 * self.f_pen(x) + self.f_opt
        return out

class GriewankRosenbrock(BaseObjective):
    def __init__(self, d, int_opt=(-5., 5.)):
        super(GriewankRosenbrock, self).__init__(d, int_opt=int_opt)
        self.c = np.maximum(1, np.sqrt(self.d) / 8)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        z = (self.c * self.r @ x.T + 1/2).T
        a = z[:, :-1]**2 - z[:, 1:]
        b = z[:, :-1] - 1

        s = 100 * a**2 + b**2

        out = 10 / (self.d - 1) * np.sum(s / 4000 - np.cos(s), axis=1) + 10 + self.f_opt

        return out
    
# 5 Multi-modal functions with weak global structure
class Schwefel(BaseObjective):
    def __init__(self, d, val_opt=4.2096874633 / 2, alpha=10):
        super(Schwefel, self).__init__(d, val_opt=val_opt, alpha=alpha)

    def evaluate_full(self, x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d

        x_hat = 2 * self.one_pm * x

        z_hat = np.zeros_like(x)
        z_hat[:, 0] = x_hat[:, 0]
        z_hat[:, 1:] = x_hat[:, 1:] + 0.25 * (x_hat[:, :-1] - 2 * np.abs(self.x_opt[:, :-1]))

        z = 100 * (self.lambda_alpha @ (z_hat - 2 * np.abs(self.x_opt)).T + 2 * np.abs(self.x_opt).T).T

        return -1 / (100 * self.d) * np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=1) + 4.189828872724339 \
            + 100 * self.f_pen(z / 100) + self.f_opt
            
    