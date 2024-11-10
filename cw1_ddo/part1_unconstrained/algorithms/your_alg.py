from scipy.optimize import minimize
import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from scipy.optimize import minimize

#########################
# --- Random search --- #
#########################

def Random_search(f, n_p, bounds_rs, iter_rs):
    '''
    This function is a naive optimization routine that randomly samples the 
    allowed space and returns the best value.

    n_p: dimensions
    iter_rs: number of points to create
    '''

    # arrays to store sampled points
    localx   = np.zeros((n_p,iter_rs))  # points sampled
    localval = np.zeros((iter_rs))        # function values sampled
    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias # sampling
        localx[:,sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b, x_b


def your_alg(f, x_dim, bounds, f_eval_, has_x0=False):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    bounds: should be a list of tuples, where each tuple contains the lower and upper bounds for each dimension
    '''
    if has_x0:
        iter_ = f_eval_ - 1
        x_best = f.x0[0].flatten()
    else:
        n_rs = int(max(x_dim + 1, f_eval_ * 0.05))
        iter_ = f_eval_ - n_rs
        f_best, x_best = Random_search(f, x_dim, bounds, n_rs)

    # Now, bounds are passed as a list of tuples
    # Bounds are now compatible with minimize (in COBYQA style)
    
    # Create and run the Bayesian Optimization
    myBopt = GPyOpt.methods.BayesianOptimization(
        f=f.fun_test,
        domain=[{'name': 'var_' + str(i + 1), 'type': 'continuous', 'domain': bounds[i]} for i in range(len(bounds))],
        X=x_best.reshape(1, -1) if has_x0 else None,
        initial_design_numdata=n_rs if not has_x0 else 0,
        exact_feval=True
    )
    myBopt.run_optimization(max_iter=iter_)

    # Extract the optimal solution and its corresponding function value
    x_opt = myBopt.x_opt
    f_opt = myBopt.fx_opt

    # Add the required team_names and cids
    team_names = ['7', '8']
    cids = ['01234567']

    return x_opt, f_opt, team_names, cids
