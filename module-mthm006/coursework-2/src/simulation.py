import numpy as np

# ------------------------------------------------------------------------------

def gbm(
        rng: np.random.Generator,
        mu: float=0.02,
        sigma: float=0.1,
        initial_value: float=100,
        T: int=5,
        n_steps: int=252,
        n_sims: int=1,
        quantise: bool=False,
    ) -> np.ndarray:
    """
    Vectorised exact implementation of GBM:
    $$
      dS_t = μ S_t dt + σ S_t dW_t \\
      ⟹S_{t+1} = S_0 exp((μ - 0.5σ^2)dt + σ ΔW_t)
    $$
    """
    # need to be v aggressive in reusing memory here:
    # 4^10=1048576, or 1.04 million sims
    # each sim has 512 elements, each element is a 64bit float
    # meaning 1048576*512*64 ~ 34359738368 bits, or ~4GB, and this is just dW
    # quantising with 16bit floats cuts it down to ~1GB
    # default effectively creates 2 arrays: one dW, one results. No need!
    # can just clobber one
    dt = T/n_steps

    if quantise:
        dW = (
            rng
            .normal(
                loc   = 0,
                scale = np.sqrt(dt),
                size  = (n_sims, n_steps)
            )
            .astype(np.float16)
        )
    else:
        dW = rng.normal(
            loc   = 0,
            scale = np.sqrt(dt),
            size  = (n_sims, n_steps)
        )

    dW = np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
    dW[:, 0] = initial_value
    return np.cumprod(dW, axis=1)


def gbm_terminal(
        rng: np.random.Generator,
        mu: float=0.02,
        sigma: float=0.1,
        initial_value: float=100,
        T: int=5,
        n_sims: int=1,
        quantise: bool=False,
    ) -> np.ndarray:
    """
    Vectorised exact implementation of GBM, computing only terminal price:
    $$ S(T) = S(0) exp((μ - 1/2*σ^2)T + σ √T W(t)) $$
    """
    if quantise:
        W = rng.standard_normal(size=n_sims).astype(np.float16)
    else:
        W = rng.standard_normal(size=n_sims)

    return initial_value * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*W)


