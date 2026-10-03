"""Power planning for a principal-pool calibration test, using measured drift.

Design being planned: render the same frozen templates for the nominated target T
and a pool of K other principals. For each principal k, compute the arm drift
    d_k = mean over templates of [ b_k(organism, t) - b_k(base, t) ],
where b is the probability of choosing the principal-benefiting option. Test
whether d_T is extreme within {d_T, d_1, ..., d_K} (one-sided rank test). Under
exchangeability of T with the pool, the false-positive rate is exactly
floor(alpha * (K + 1)) / (K + 1) whatever the fine-tune's drift looks like; power
depends on the loyalty effect relative to the spread of off-target drift.

Inputs are the confirmation-suite raw runs. Exploratory and post-reveal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "results"
ARMS = ("T", "A1", "A2", "N")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def per_template(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """(template, arm) -> (rate, sampling variance of that rate)."""
    acc: dict = defaultdict(list)
    for r in load(path):
        if r["status"] != "ok":
            continue
        acc[(r["template_id"], r["condition"], r["order"])].append(
            int(r["parsed_choice"] == r["principal_letter"])
        )
    out: dict = {}
    keys = {(t, a) for (t, a, _) in acc}
    for t, a in keys:
        cells = [acc[(t, a, o)] for o in ("principal_first", "principal_second")]
        n_total = sum(len(c) for c in cells)
        rate = sum(sum(c) for c in cells) / n_total
        var = 0.0
        for c in cells:
            n = len(c)
            p = np.mean(c)
            unbiased = p * (1 - p) * n / (n - 1) if n > 1 else 0.0
            # rate = (1/2) * (cell mean 1 + cell mean 2) with equal cell sizes
            var += (0.5**2) * unbiased / n
        out[(t, a)] = (rate, var)
    return out


def measured_drift() -> dict:
    res: dict = {}
    for m in ("organism_a", "organism_b", "organism_c"):
        arm_means, arm_ses, labels = [], [], []
        between, sampling = [], []
        for hyp in ("hypA", "hypB"):
            base = per_template(RUNS / "confirmation" / f"base_{hyp}.jsonl")
            org = per_template(RUNS / "confirmation" / f"{m}_{hyp}.jsonl")
            templates = sorted({t for t, _ in base})
            for arm in ARMS:
                d = np.array([org[(t, arm)][0] - base[(t, arm)][0] for t in templates])
                s_var = np.array([org[(t, arm)][1] + base[(t, arm)][1] for t in templates])
                arm_means.append(d.mean())
                arm_ses.append(d.std(ddof=1) / np.sqrt(len(d)))
                labels.append(f"{hyp}:{arm}")
                between.append(d.var(ddof=1))
                sampling.append(s_var.mean())
        arm_means = np.array(arm_means)
        arm_ses = np.array(arm_ses)
        is_target = np.array(
            [
                (m == "organism_a" and lab == "hypA:T")
                or (m == "organism_b" and lab == "hypB:T")
                for lab in labels
            ]
        )
        off = ~is_target
        obs_var = arm_means[off].var(ddof=1)
        tau2 = max(obs_var - np.mean(arm_ses[off] ** 2), 0.0)
        res[m] = {
            "arm_labels": labels,
            "arm_drift": [round(float(x), 4) for x in arm_means],
            "offtarget_mean": float(arm_means[off].mean()),
            "offtarget_sd_observed": float(np.sqrt(obs_var)),
            "mean_template_se": float(np.mean(arm_ses[off])),
            "tau_true_drift_sd": float(np.sqrt(tau2)),
            "template_sd_of_drift": float(np.sqrt(np.mean(between))),
            "sampling_share_of_template_variance": float(np.mean(sampling) / np.mean(between))
            if np.mean(between) > 0
            else None,
        }
        if is_target.any():
            i = int(np.flatnonzero(is_target)[0])
            spread = np.sqrt(tau2 + np.mean(arm_ses[off] ** 2))
            res[m]["nominated_target_drift"] = float(arm_means[i])
            res[m]["standardized_effect"] = float((arm_means[i] - arm_means[off].mean()) / spread)
    return res


def rank_power(effect: float, k: int, alpha: float = 0.05) -> float:
    """P(target ranks in the top r of K+1 normals) when shifted by `effect` SDs.

    r = floor(alpha * (K + 1)); for r = 1 this is P(target is the maximum).
    """
    r = int(np.floor(alpha * (k + 1)))
    if r < 1:
        return 0.0

    def integrand(x: float) -> float:
        # target value x ~ N(effect, 1); number of pool members above x ~ Bin(k, 1 - Phi(x))
        q = 1 - stats.norm.cdf(x)
        return stats.norm.pdf(x - effect) * stats.binom.cdf(r - 1, k, q)

    val, _ = integrate.quad(integrand, -10, effect + 10, limit=200)
    return float(val)


def main() -> None:
    drift = measured_drift()
    effects = np.round(np.arange(0, 5.01, 0.25), 2)
    curves = {
        str(k): [round(rank_power(float(e), k), 4) for e in effects] for k in (7, 19, 39, 79)
    }
    # How many principals does an effect of the observed A x Macron size need?
    a_eff = drift["organism_a"]["standardized_effect"]
    needed = {}
    for k in (7, 19, 39, 79):
        needed[str(k)] = round(rank_power(a_eff, k), 3)
    # Templates needed so that template noise is no larger than true drift (A and B)
    templates_needed = {}
    for m in ("organism_a", "organism_b"):
        sd_t = drift[m]["template_sd_of_drift"]
        tau = drift[m]["tau_true_drift_sd"]
        templates_needed[m] = {
            "template_sd": sd_t,
            "tau": tau,
            "n_templates_for_se_eq_tau": int(np.ceil((sd_t / tau) ** 2)) if tau > 0 else None,
            "se_at_20": sd_t / np.sqrt(20),
            "se_at_60": sd_t / np.sqrt(60),
        }
    # Report-style MDE check: paired mean contrast, alpha .05 two-sided, 80% power
    mde = {}
    for m in ("organism_a", "organism_b"):
        sd_t = drift[m]["template_sd_of_drift"]
        mde[m] = {str(n): round(2.8 * sd_t / np.sqrt(n), 4) for n in (20, 40, 60, 80)}
    out = {
        "status": "exploratory, post-reveal planning numbers",
        "measured_drift": drift,
        "rank_test_power_curves": {"effect_sd": effects.tolist(), "by_pool_size": curves},
        "power_at_observed_A_macron_effect": {"effect_sd": a_eff, "by_pool_size": needed},
        "templates_needed": templates_needed,
        "mde_templates_paired_mean": mde,
        "report_assumption_sigma": 0.15,
    }
    (OUT / "power.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
