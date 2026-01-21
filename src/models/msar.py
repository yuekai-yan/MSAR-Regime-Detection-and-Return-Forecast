import numpy as np
import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


def msar(
    T: np.ndarray,
    k_regimes: int = 2,
    order: int = 1,
    switching_ar: bool = True,         # AR coefficient φ switches by regime
    switching_variance: bool = True,   # Variance switches by regime
    trend: str = "c",                  # Each regime has its own constant μ_j
    init_probs: np.ndarray | None = None,
    exog_tvtp: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Fit Markov-Switching AR models for each company in each industry and return a single `info` dict.

    Parameters:
    T : np.ndarray, shape (k, n, m)
        3D tensor:
          - k: number of categories
          - n: number of dates
          - m: number of stocks per category
        Entry T[c, t, i] is (close - open) for stock i in category c at time t.
    k_regimes : int, default 2
        Number of Markov regimes.
    order : int, default 1
        AR order
    switching_ar : bool, default True
        Whether AR coefficients are regime-specific.
    switching_variance : bool, default True
        Whether the innovation variance is regime-specific.
    trend : {"n", "c"}, default "c"
        Per-regime trend component (e.g., constant).
    init_probs: shape (K, ) or (k, K), predicted init_probs, if it is not None else "steady-state".

    Returns:
    info : Dict[str, Any]
        A single dictionary containing all outputs for downstream analysis:

        info["meta"]
            Basic metadata about the run:
              - "k_categories": int, number of categories k
              - "n_time": int, number of time points n
              - "m_stocks": int, number of stocks per category m
              - "k_regimes": int, number of regimes
              - "order": int, AR order
              - "switching_ar": bool, regime-specific AR?
              - "switching_variance": bool, regime-specific variance?
              - "trend": str, trend specification
              - "random_state": int, recorded seed

        info["standardization"]
            Per (c, i) standardization parameters used to z-score each series:
            info["standardization"][str(c)][str(i)] = {"mean": float, "std": float}

        info["fits"]
            Per (c, i) fitted results and key quantities needed for analysis:
            info["fits"][str(c)][str(i)] is a dict with:
              - "diagnostics":
                  * "converged": bool, optimizer convergence flag if available
                  * "llf": float, log-likelihood
                  * "aic": float, Akaike information criterion
                  * "bic": float, Bayesian information criterion
                  * "nobs": int, number of observations used
                  * "method": str, optimizer name ("powell")
              - "params":
                  * "names": list[str], parameter names in `res.params`
                  * "values": list[float], parameter values aligned with names
                  * "mu_by_regime": list[float], per-regime constants μ_j
                  * "phi_by_regime": list[float], per-regime (or shared) AR(1) φ_j
                  * "sigma2_by_regime": list[float], per-regime (or shared) innovation variances σ_j^2
              - "probabilities":
                  * "p_t_t_last": list[float], filtered p(s_t | Y_{1:t}) at the last time t
                  * "p_tp1_t": list[float], one-step-ahead p(s_{t+1} | Y_{1:t})
              - "transition":
                  * "P_last": list[list[float]], last transition matrix P where
                    P[i][j] = Pr(next regime i | current regime j)
              - "last_observation":
                  * "r_t_raw": float, final raw observation r_t
                  * "r_t_std": float, final standardized observation
              - "conditional_means":
                  * "cond_mean_by_regime_std": list[float], μ_j + φ_j * r_t (standardized scale)
                  * "forecast_std": float, state-weighted one-step forecast (standardized scale)
                  * "forecast_raw": float, inverse-transformed one-step forecast (original units)
              - "standardization": {"mean": float, "std": float} duplicated for convenience
              - If fitting failed, includes:
                  * "diagnostics": {"converged": False, "method": "powell"}
                  * "error": str with the exception message
                  * "standardization": {"mean": float, "std": float}

        info["fail_counts"]
            Per-category count of stocks that failed to fit:
            info["fail_counts"][str(c)] = int

        info["summary"]
            Aggregated, ready-to-use arrays (stored as lists for JSON friendliness):
              - "exp_next": (m, k) one-step expected return in original units
              - "p_next_all": (k_regimes, m, k) one-step predicted state probabilities

    Notes
    -----
    Model per stock (standardized series r_t):
      r_t = μ_{s_t} + φ_{s_t} * r_{t-1} + σ_{s_t} * ε_t,
    with regime transition described by matrix P where P[i, j] = Pr(s_{t+1}=i | s_t=j).
    The one-step state probabilities are p_{t+1|t} = P @ p_{t|t},
    and the one-step forecast (std. scale) is E[r_{t+1}|Y_{1:t}] = sum_j p_{t+1|t}(j) * (μ_j + φ_j * r_t).
    """

    assert T.ndim == 3, "T must have shape (k, n, m)"
    k, n, m = T.shape
    if n < 3:
        raise ValueError("Need at least 3 time points to fit AR and forecast the next period.")

    exp_next = np.zeros((m, k), dtype=float)
    p_next_all = np.zeros((k_regimes, m, k), dtype=float)

    info: Dict[str, Any] = {
        "meta": {
            "k_categories": k,
            "n_time": n,
            "m_stocks": m,
            "k_regimes": k_regimes,
            "order": order,
            "switching_ar": switching_ar,
            "switching_variance": switching_variance,
            "trend": trend,
            "has_tvtp": exog_tvtp is not None, 
        },
        "standardization": {str(c): {str(i): None for i in range(m)} for c in range(k)},
        "fits": {str(c): {str(i): None for i in range(m)} for c in range(k)},
        "fail_counts": {str(c): 0 for c in range(k)},
        "summary": {
            "exp_next": None,     # (m, k)
            "p_next_all": None,   # (k_regimes, m, k)
        },
    }

    def _extract_regime_id(name: str) -> int:
        import re
        m_ = re.search(r"\[(\d+)\]", name)
        return int(m_.group(1)) if m_ else 0
    
    def _get_exog_tvtp_for_category(c: int, n_time: int) -> np.ndarray | None:
        """
        Return exog_tvtp design matrix of shape (n_time, q) for category c,
        or None if TVTP is not used.
        """
        if exog_tvtp is None:
            return None

        X = np.asarray(exog_tvtp)
        if X.ndim == 2:
            # exog_tvtp has shape (n, q): use same TVTP for all categories/stocks
            if X.shape[0] != n_time:
                raise ValueError("exog_tvtp with shape (n, q) must have n == T.shape[1].")
            return X
        elif X.ndim == 3:
            # exog_tvtp has shape (k, n, q): category-specific TVTP
            if X.shape[0] != T.shape[0] or X.shape[1] != n_time:
                raise ValueError("exog_tvtp with shape (k, n, q) must match (k, n) of T.")
            return X[c, :, :]
        else:
            raise ValueError("exog_tvtp must have shape (n, q) or (k, n, q).")

    for c in range(k):
        fails = 0
        X_tvtp_c = _get_exog_tvtp_for_category(c, n) 
        for i in range(m):
            r0 = pd.Series(T[c, :, i], dtype=float)
            if not np.isfinite(r0).all() or r0.std() < 1e-4:
                exp_next[i, c] = r0.mean()
                continue

            # Standardize series
            mu0 = float(r0.mean())
            sd0 = float(r0.std()) + 1e-8
            r = (r0 - mu0) / sd0
            info["standardization"][str(c)][str(i)] = {"mean": mu0, "std": sd0}

            try:
                # Fit Markov-Switching AR
                mod = MarkovAutoregression(
                    endog=r,
                    k_regimes=k_regimes,
                    order=order,
                    switching_ar=switching_ar,
                    switching_variance=switching_variance,
                    trend=trend,
                    exog_tvtp=X_tvtp_c,
                )
                if init_probs is not None:
                    init_probs = np.asarray(init_probs, dtype=float)

                    if init_probs.ndim == 1:
                        # Shape (K,): use the same initial distribution for all (c, i)
                        probs = init_probs

                    elif init_probs.ndim == 2:
                        # Shape (k, K): different initial distributions per category,
                        # but identical within the same category
                        if init_probs.shape[0] != k:
                            raise ValueError("init_probs shape (k, K) must have k equal to number of categories.")
                        probs = init_probs[c, :]

                    else:
                        raise ValueError("init_probs must have shape (K,) or (k, K).")

                    if probs.size != k_regimes:
                        raise ValueError("init_probs last dimension must equal k_regimes (K).")

                    mod.initialize_known(probabilities=probs)
                
                # Fit the model
                res = mod.fit(method="powell", disp=False, maxiter=3000)

                # Filtered probabilities p(s_t | Y_{1:t}) at the last time t
                p_t_t = np.asarray(res.filtered_marginal_probabilities.iloc[-1], dtype=float)

                # Transition matrix P[i, j] = Pr(next = i | current = j) (last time slice)
                if X_tvtp_c is not None:
                    P_full = res.model.regime_transition_matrix(res.params, exog_tvtp=X_tvtp_c)
                else:
                    P_full = res.model.regime_transition_matrix(res.params)
                P = np.asarray(P_full[..., -1], dtype=float)

                # One-step ahead probabilities: p_{t+1|t} = P @ p_{t|t}
                p_tp1_t = P @ p_t_t
                p_next_all[:, i, c] = p_tp1_t

                # Parameter extraction
                names = list(res.params.index)

                # Per-regime constants μ_j
                const_idx = [idx for idx, nm in enumerate(names) if "const" in nm]
                const_idx_sorted = sorted(const_idx, key=lambda t_: _extract_regime_id(names[t_]))
                mu = np.array([float(res.params[names[j]]) for j in const_idx_sorted]) if const_idx_sorted else np.zeros(k_regimes)

                # AR(1) coefficients φ_j (shared or regime-specific)
                if order >= 1:
                    ar1_idxs = [idx for idx, nm in enumerate(names) if "ar.L1" in nm]
                    if len(ar1_idxs) == k_regimes:
                        ar1_sorted = sorted(ar1_idxs, key=lambda t_: _extract_regime_id(names[t_]))
                        phi_vec = np.array([float(res.params[names[j]]) for j in ar1_sorted])
                    elif len(ar1_idxs) >= 1:
                        phi_vec = np.repeat(float(res.params[names[ar1_idxs[0]]]), k_regimes)
                    else:
                        phi_vec = np.zeros(k_regimes)
                else:
                    phi_vec = np.zeros(k_regimes)

                # Innovation variances σ_j^2 (shared or regime-specific)
                sigma_idxs = [idx for idx, nm in enumerate(names) if "sigma2" in nm]
                if len(sigma_idxs) == k_regimes:
                    sigma_sorted = sorted(sigma_idxs, key=lambda t_: _extract_regime_id(names[t_]))
                    sigma2 = np.array([float(res.params[names[j]]) for j in sigma_sorted])
                elif len(sigma_idxs) >= 1:
                    sigma2 = np.repeat(float(res.params[names[sigma_idxs[0]]]), k_regimes)
                else:
                    sigma2 = np.full(k_regimes, np.nan)

                # Conditional means on standardized scale and one-step forecast
                rt = float(r.iloc[-1])
                cond_mean = mu + phi_vec * rt
                exp_std = float(np.dot(p_tp1_t, cond_mean))

                # Inverse transform to original units
                exp_next[i, c] = exp_std * sd0 + mu0

                # Store per-stock results
                info["fits"][str(c)][str(i)] = {
                    "diagnostics": {
                        "converged": bool(getattr(res, "mle_retvals", {}).get("converged", True)),
                        "llf": float(getattr(res, "llf", np.nan)),
                        "aic": float(getattr(res, "aic", np.nan)),
                        "bic": float(getattr(res, "bic", np.nan)),
                        "nobs": int(getattr(res, "nobs", len(r))),
                        "method": "powell",
                    },
                    "params": {
                        "names": names,
                        "values": [float(res.params[nm]) for nm in names],
                        "mu_by_regime": mu.tolist(),
                        "phi_by_regime": phi_vec.tolist(),
                        "sigma2_by_regime": sigma2.tolist(),
                    },
                    "probabilities": {
                        "p_t_t_last": p_t_t.tolist(),
                        "p_tp1_t": p_tp1_t.tolist(),
                    },
                    "transition": {
                        "P_last": P.tolist(),
                    },
                    "last_observation": {
                        "r_t_raw": float(r0.iloc[-1]),
                        "r_t_std": rt,
                    },
                    "conditional_means": {
                        "cond_mean_by_regime_std": cond_mean.tolist(),
                        "forecast_std": exp_std,
                        "forecast_raw": float(exp_next[i, c]),
                    },
                    "standardization": {"mean": mu0, "std": sd0},
                }

            except Exception as e:
                # On failure, fall back to mean as forecast and record error
                fails += 1
                exp_next[i, c] = mu0
                info["fits"][str(c)][str(i)] = {
                    "diagnostics": {"converged": False, "method": "powell"},
                    "error": f"{type(e).__name__}: {e}",
                    "standardization": {"mean": mu0, "std": sd0},
                }

        info["fail_counts"][str(c)] = int(fails)

    # Populate summary section with arrays (as lists for JSON compatibility)
    info["summary"]["exp_next"] = exp_next.tolist()
    info["summary"]["p_next_all"] = p_next_all.tolist()
    return info



def analyze_info(info, c: int, i: int, w=(0.4, 0.2, 0.2, 0.2)):
    """
    Summarize regime probabilities and economic interpretation (Bull vs Bear).

    Parameters
    ----------
    info : dict
        Output from msar()
    c : int
        Category index
    i : int
        Stock index
    w : tuple[float, float, float, float], optional
        Weights for the composite regime score in the order:
          (w_mean, w_var, w_persist, w_sharpe).
        - w_mean:   weight on long-run/conditional mean (higher is better)
        - w_var:    weight on variance (applied to -variance; lower is better)
        - w_persist:weight on persistence p(j→j)
        - w_sharpe: weight on Sharpe ratio
        Default is (0.4, 0.2, 0.2, 0.2).
        Note: length must be 4.
    """
    
    if len(w) != 4:
        raise ValueError("w must be a length-4 tuple/list: (w_mean, w_var, w_persist, w_sharpe)")


    fit = info["fits"][str(c)][str(i)]
    if fit is None or "probabilities" not in fit:
        print(f"[Warning] No valid fit for category {c}, asset {i}.")
        return

    # --- extract key data ---
    p_last = np.array(fit["probabilities"]["p_t_t_last"])
    p_next = np.array(fit["probabilities"]["p_tp1_t"])
    
    # --- extract one-step forecast quantities  ---
    cm = fit.get("conditional_means", {})                 # forecast container
    forecast_std = cm.get("forecast_std", None)           # E[r_{t+1} | Y_{1:t}] on standardized scale
    forecast_raw = cm.get("forecast_raw", None)           # E[r_{t+1} | Y_{1:t}] on original scale
    cond_mean_by_regime_std = cm.get("cond_mean_by_regime_std", None)  # [ADD] mu_j + phi_j * r_t (std)

    # standardization params for optional regime-wise raw means
    std_params = fit.get("standardization", {})
    mean0 = std_params.get("mean", None)                  # original mean used for z-score
    std0  = std_params.get("std", None)                   # original std used for z-score

    mu = np.array(fit["params"]["mu_by_regime"])
    sigma2 = np.array(fit["params"]["sigma2_by_regime"])
    phi = np.array(fit["params"]["phi_by_regime"])

    # --- classify regimes (bull vs bear) ---
    # Composite scoring using mean/variance, persistence, and Sharpe
    phi = np.asarray(fit["params"]["phi_by_regime"], float)
    P   = np.asarray(fit["transition"]["P_last"], float)
    # Persistence (stay probability) and simple Sharpe
    p_stay = np.diag(P)
    sharpe = mu / np.sqrt(np.maximum(sigma2, 1e-12))
    # Weighted score: higher mean, lower variance, higher persistence, higher Sharpe
    score = w[0] * mu + w[1] * -sigma2 + w[2] * p_stay + w[3] * sharpe

    bull_id = int(np.argmax(score))
    bear_id = 1 - bull_id


    # --- identify current & next likely regimes ---
    curr_regime = int(np.argmax(p_last))
    next_regime = int(np.argmax(p_next))

    def regime_name(idx):
        if idx == bull_id:
            return f"Bull  (Regime {idx})"
        elif idx == bear_id:
            return f"Bear  (Regime {idx})"
        else:
            return f"Regime {idx}"

    print("One-step-ahead regime probabilities")
    print("-" * 45)
    # --- print summary report ---
    print(f"μ by regime     : {mu}")
    print(f"σ² by regime    : {sigma2}")
    print(f"φ by regime     : {phi}")
    print(f"Current state probabilities p(s_t | Y₁:ₜ): {p_last}")
    print(f"Predicted next  probabilities p(sₜ₊₁ | Y₁:ₜ): {p_next}")
    print(f"→ Current regime  : {regime_name(curr_regime)}  (p={p_last[curr_regime]:.3f})")
    print(f"→ Next predicted  : {regime_name(next_regime)}  (p={p_next[next_regime]:.3f})")
    
    # --- short textual interpretation ---
    if curr_regime == bear_id and next_regime == bull_id:
        print("📈 Model suggests a *Bull transition* — market may shift upward.")
    elif curr_regime == bull_id and next_regime == bear_id:
        print("📉 Model suggests a *Bear transition* — potential downturn.")
    elif next_regime == bull_id:
        print("✅ Market likely remains in *Bull regime*.")
    else:
        print("⚠️ Market likely remains in *Bear regime*.")      
    print()
    
    #  --- report one-step return forecast E[r_{t+1} | Y_{1:t}] ---
    print("One-step-ahead expected return")
    print("-" * 45)

    # regime-wise conditional means (standardized), and optionally raw
    if cond_mean_by_regime_std is not None:
        cm_reg = np.asarray(cond_mean_by_regime_std, dtype=float)  
        print(f"Regime-wise E[rₜ₊₁ | sₜ₊₁=j, Y₁:ₜ] (std): {cm_reg}")  

        # optional: convert regime-wise conditional means back to raw scale
        if mean0 is not None and std0 is not None:
            cm_reg_raw = cm_reg * float(std0) + float(mean0)       
            print(f"Regime-wise E[rₜ₊₁ | sₜ₊₁=j, Y₁:ₜ] (raw): {cm_reg_raw}")  

    # print the aggregated forecast
    if forecast_std is not None:
        print(f"E[rₜ₊₁ | Y₁:ₜ] (std) = {float(forecast_std):.6g}")  
    if forecast_raw is not None:
        print(f"E[rₜ₊₁ | Y₁:ₜ] (raw) = {float(forecast_raw):.6g}") 
    print()                                                      