import numpy as np
from scipy import optimize
from scipy.sparse import eye
from sklearn.utils.extmath import safe_sparse_dot

def two_loops(grad_x, m, s_list, y_list, mu_list, B0):
    '''
    Parameters
    ----------
    grad_x : ndarray, shape (n,)
        gradient at the current point

    m : int
        memory size

    s_list : list of length m
        the past m values of s

    y_list : list of length m
        the past m values of y

    mu_list : list of length m
        the past m values of mu

    B0 : ndarray, shape (n, n)
        Initial inverse Hessian guess

    Returns
    -------
    r :  ndarray, shape (n,)
        the L-BFGS direction
    '''
    q = grad_x.copy()
    alpha_list = []
    for s, y, mu in zip(reversed(s_list), reversed(y_list), reversed(mu_list)):
        alpha = mu * safe_sparse_dot(s, q)
        alpha_list.append(alpha)
        q -= alpha * y
    r = safe_sparse_dot(B0, q)
    for s, y, mu, alpha in zip(s_list, y_list, mu_list, reversed(alpha_list)):
        beta = mu * safe_sparse_dot(y, r)
        r += (alpha - beta) * s
    return -r

def lbfgs(
        x0,
        f,
        f_grad,
        f_hessian,
        max_iter=100,
        m=2,
        tol=1e-6,
        tol_norm=None,
        maxls=10,
        inverse_direction=None,
):
    default_step = 0.01
    c1 = 0.0001
    c2 = 0.0009
    if tol_norm is None:
        tol_norm = lambda x: np.max(np.abs(x))


    # This variable is used to indicate whether or not we want to print
    # monitoring information (iteration counter, function value and norm of the gradient)
    verbose = False

    all_x_k, all_f_k = list(), list()
    x = x0

    all_x_k.append(x.copy())
    all_f_k.append(f(x))

    B0 = eye(len(x))  # Hessian approximation

    grad_x = f_grad(x)

    y_list, s_list, mu_list = [], [], []
    for k in range(1, max_iter + 1):

        # Compute the search direction
        d = two_loops(grad_x, m, s_list, y_list, mu_list, B0)

        # Compute a step size using a line_search to satisfy the
        # strong Wolfe conditions
        step, _, _, new_f, _, new_grad = optimize.line_search(f, f_grad, x,
                                                              d, grad_x,
                                                              c1=c1, c2=c2,
                                                              maxiter=maxls)

        if step is None or new_grad is None:
            print("Line search did not converge at iteration %s" % k)
            step = default_step

        # Compute the new value of x
        s = step * d
        x += s
        if new_grad is None:
            new_grad = f_grad(x)
        y = new_grad - grad_x
        mu = 1 / safe_sparse_dot(y, s)
        ##################################################################
        # Update the memory
        y_list.append(y.copy())
        s_list.append(s.copy())
        mu_list.append(mu)
        if inverse_direction is not None:
            # update the memory with the extra secant condition for inverse
            e = two_loops(inverse_direction, m, s_list, y_list, mu_list, B0)
            y_tilde = f_grad(x + e) - new_grad
            mu = 1 / safe_sparse_dot(y_tilde, e)
            y_list.append(y_tilde.copy())
            s_list.append(e.copy())
            mu_list.append(mu)
        if len(y_list) > m:
            y_list.pop(0)
            s_list.pop(0)
            mu_list.pop(0)
            if inverse_direction is not None:
                y_list.pop(0)
                s_list.pop(0)
                mu_list.pop(0)
        ##################################################################

        all_x_k.append(x.copy())
        all_f_k.append(new_f)

        l_inf_norm_grad = tol_norm(new_grad)

        if verbose:
            print('iter: %d, f: %.6g, l_inf_norm(grad): %.6g' %
                  (k, new_f, l_inf_norm_grad))

        if l_inf_norm_grad < tol:
            break

        grad_x = new_grad

    return np.array(all_x_k), np.array(all_f_k)
