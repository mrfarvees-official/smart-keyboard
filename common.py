"""
common.py : shared toolkit for the nn-from-scratch course.

Conventions
-----------
* All arrays are float64 (gradient checking needs the precision).
* Parameters live in a plain: {"W0": array, "b0": array, ...}.
* Gradients live in a dict with the SAME keys and shapes as the param
* Batches are row-major: X has shape (batch, features)
"""
import numpy as np

#-------------
# Activations
#-------------
def sigmoid(x):
    """Numerical stable sigmoid (never exponentiates a large positive number)."""
    x = np.asanyarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def softmax(x, axis=-1):
    """Numerical stable softmax: subtract the max before exponentiating."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

#-----------
# Optimizer
#-----------
class Adam:
    """Adam optimizer a dict of Numpy parameters (updated in place).
    
    m_t = b1*m + (1-b1)*g                           (first moment)
    v_t = b2*v + (1-b2)*g^2                         (second moment)
    m_hat = m_t/(1-b1^t), v_hat = v_t/(1-b2^t)      (bias correction)
    p -= lr * m_hat / (sqrt(v_hat) * eps)
    """
    def __init__(self, params: dict, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        
    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g**2
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    @staticmethod        
    def clip_grads(grads: dict, max_norm):
        """Rescale all gradients in place so their GLOBAL L2 norm <= max_norm.
        Returns the norm before clipping (useful to log)."""
        total = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if total > max_norm:
            scale = max_norm / (total + 1e-12)
            for k in grads:
                grads[k] *= scale
        return total

#-------------------
# Gradient checking
#-------------------
def gradcheck(loss_fn, params: dict, grads: dict, eps=1e-5, n_checks=6, seed=0, tol=1e-5, verbose=True):
    """Compare analytic gradients with centered finite differences.
    
        numeric = (L(theta + eps) - L(theta - eps)) / (2 * eps)
    
    Parameters
    ----------
    loss_fn : callable with no arguments. It must run a fresh FORWARD pass with
              the current contents of 'params' and return the scaler loss.
              (It must NOT call backward and must not cache stale values.)
    params  : dict of arrays; entries are perturbed IN PLACE, then restored.
    grads   : dict with the analytic gradients (same keys/shapes as params),
              computed BEFORE calling gradcheck, at the current params.
              
    Returns the worst relative error. Rule of thumb: below 1e-5 is correct,
    above 1e-3 means a bug.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for name, p in params.items():
        flat = p.reshape(-1) # a view: writing to flat writes into p
        gflat = grads[name].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_checks, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_fn()
            flat[i] = orig - eps
            lm = loss_fn()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(ana - num) / max(abs(ana), abs(num), 1e-8)
            worst = max(worst, rel)
            if verbose and rel > tol:
                print(f"    MISMATCH {name}[{i}]: analytic={ana:.3e} numeric={num:.3e} rel={rel:.2e}")
    if verbose:
        status = "PASS" if worst < tol else "FAIL"
        print(f"gradcheck {status}: worst relative error = {worst:.2e}")
    return worst