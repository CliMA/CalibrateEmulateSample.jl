def eks_update(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
  """
  Ensemble update based on the continuous time limit of the EKS.
  """
  self.update_rule = 'eks_update'

  # For ensemble update
  E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
  R = Geval - y_obs[:,np.newaxis]
  D = (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

  hk = self.timestep_method(D,  Geval, y_obs, Gamma, np.linalg.cholesky(Gamma), **kwargs)
  if kwargs.get('time_step', None) in ['adaptive', 'constant']:
    Cpp = np.cov(Geval, bias = True)
    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(hk * Cpp + Gamma, R))

  Umean = U0.mean(axis = 1)[:, np.newaxis]
  Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

  Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
    U0 - hk * np.matmul(U0 - Umean, D)  + \
    hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)))
  Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
    np.random.normal(0, 1, [self.p, self.J])))

  return Uk
