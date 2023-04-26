import numpy as np

# class AdaDelta:
#     def __init__(self, gamma=0.99):
#         self.gamma = gamma
#         self.E_g = None
#         self.E_p = None
#         self.delta_p = None

#     def update(self, params, grads):
#         if self.E_g == None:
#             self.E_g = []
#             self.E_p = []
#             self.delta_p = []
#             for val in params:
#                 self.E_g.append(np.zeros_like(val))
#                 self.E_p.append(np.zeros_like(val))
#                 self.delta_p.append(np.zeros_like(val))

#         for i in range(len(params)):
#             self.E_g[i] = self.gamma*self.E_g[i] + (1-self.gamma)*(grads[i]*grads[i])
#             self.E_p[i] = self.gamma*self.E_p[i] + (1-self.gamma)*(self.delta_p[i]**2)
#             self.delta_p[i] = -np.sqrt(self.E_p[i]+1e-7)/np.sqrt(self.E_g[i]+1e-7) * grads[i]
#             params[i] += self.delta_p[i]

class AdaDelta:
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.E_g = None
        self.E_p = None
        self.delta_p = None

    def update(self, params, grads):
        if self.E_g == None:
            self.E_g = []
            self.E_p = []
            self.delta_p = []
            for val in params:
                self.E_g.append(np.zeros_like(val))
                self.E_p.append(np.zeros_like(val))
                self.delta_p.append(np.zeros_like(val))

        for i in range(len(params)):
            self.E_g[i] = self.gamma*self.E_g[i] + (1-self.gamma)*(grads[i]*grads[i])
            self.delta_p[i] = -np.sqrt(self.E_p[i]+1e-7)/np.sqrt(self.E_g[i]+1e-7) * grads[i]
            self.E_p[i] = self.gamma*self.E_p[i] + (1-self.gamma)*(self.delta_p[i]**2)
            params[i] += self.delta_p[i]