"""
Power analysis for Study 1: AI fairness and gender bias in hiring decisions.

This module simulates paired-comparison hiring data under five hypotheses (H1-H5)
to estimate the sample size needed to detect each effect. Each simulated participant
evaluates pairs of candidates across four conditions (control, biased AI, group-fair
AI, individual-fair AI) and the resulting binary choices are analyzed with a
fixed-effects logistic regression.

Hypotheses
----------
H1  Fair AI reduces gender bias relative to no AI and biased AI.
H2  Biased AI amplifies gender bias relative to no AI and fair AI.
H3  The two fair AI definitions (group vs individual) differ in effectiveness.
H4  Individual fairness is more effective in individualist cultures.
H5  Group fairness is more effective in collectivist cultures.

Estimation
----------
All hypothesis tests are estimated by logistic regression with:
  - Participant fixed effects, absorbed via the within-cluster demeaning
    algorithm implemented in pyfixest (equivalent to including a dummy variable
    for each participant).
  - Cluster-robust standard errors at the participant level (CRV1) that account
    for any remaining within-participant correlation in the residuals.
Because participant fixed effects are collinear with any time-invariant
between-subject variable (e.g., culture), the main effect of `is_individualism`
is not identified. H4 and H5, which test cross-level (treatment-by-culture)
interactions, are identified because the relevant terms have within-subject
variation.

Usage
-----
    # Test H1, H2, H3 with a smaller sample range
    python Study1_PowerAnalysis.py --mode h1h2h3

    # Test H4, H5 with a larger sample range
    python Study1_PowerAnalysis.py --mode h4h5

    # Test all hypotheses
    python Study1_PowerAnalysis.py --mode all

Run `python Study1_PowerAnalysis.py --help` for the full list of options.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Output utilities
# ============================================================

def setup_output_dir(study_name="study1", mode=None):
    """Create a timestamped directory to store results from one run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode and mode != 'all':
        output_dir = f"results_{study_name}_{mode}_{timestamp}"
    else:
        output_dir = f"results_{study_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(results, output_dir, effect_sizes, pairs_per_treatment, n_sim, alpha, mode='all'):
    """Write power results, run metadata, and a human-readable report to disk."""
    df_results = pd.DataFrame({
        'n_participants': results['n_participants'],
        'power_h1': results['power_h1'],
        'power_h2': results['power_h2'],
        'power_h3': results['power_h3'],
        'power_h4': results['power_h4'],
        'power_h5': results['power_h5'],
        'convergence_failures': results['convergence_failures'],
        'exceptions': results['exceptions']
    })
    csv_path = os.path.join(output_dir, "power_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved power results: {csv_path}")

    # Determine which hypotheses were actually tested
    tested = [h for h in ['h1', 'h2', 'h3', 'h4', 'h5']
              if any(v is not None for v in results[f'power_{h}'])]

    metadata = {
        'study': 'Study 1',
        'mode': mode,
        'hypotheses_tested': tested,
        'timestamp': datetime.now().isoformat(),
        'n_simulations_per_n': n_sim,
        'alpha': alpha,
        'pairs_per_treatment': pairs_per_treatment,
        'participants_range': list(results['n_participants']),
        'effect_sizes': effect_sizes,
    }
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {json_path}")

    txt_path = os.path.join(output_dir, "summary_report.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"STUDY 1 POWER ANALYSIS - SUMMARY REPORT (mode='{mode}')\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Hypotheses tested: {tested}\n")
        f.write(f"Number of simulations per sample size: {n_sim}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Pairs per treatment per participant: {pairs_per_treatment}\n\n")

        f.write("EFFECT SIZES (log-odds):\n")
        for key, value in effect_sizes.items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\nDETAILED POWER RESULTS:\n")
        header_cols = [f"H{h[1]}" for h in tested]
        header = f"{'n':>6} " + " ".join(f"{c:>8}" for c in header_cols)
        f.write(header + "\n")
        for i, n in enumerate(results['n_participants']):
            row = f"{n:>6} "
            for h in tested:
                val = results[f'power_{h}'][i]
                row += f"{val:>8.3f} " if val is not None else f"{'N/A':>8} "
            f.write(row + "\n")

        f.write("\nSAMPLE SIZE REQUIREMENTS:\n")
        for target_power in [0.8, 0.95]:
            f.write(f"\n  --- {target_power*100:.0f}% Power ---\n")
            for h in tested:
                h_name = h.upper()
                powers = results[f'power_{h}']
                found = False
                for i, power in enumerate(powers):
                    if power is not None and power >= target_power:
                        f.write(f"  {h_name}: {results['n_participants'][i]} participants "
                                f"(power={power:.3f})\n")
                        found = True
                        break
                if not found:
                    valid_powers = [p for p in powers if p is not None]
                    max_p = max(valid_powers) if valid_powers else 0
                    f.write(f"  {h_name}: not reached (max power={max_p:.3f})\n")
    print(f"  Saved summary report: {txt_path}")


# ============================================================
# Data generation
# ============================================================

# Candidate-attribute distributions used across the simulation
_UNIVERSITY_VALUES = np.array([0.0, 0.897, 1.136, 1.309, 1.400, 1.881, 2.348, 2.525, 3.525])
_UNIVERSITY_PROBS = np.array([0.35, 0.05, 0.02, 0.18, 0.02, 0.15, 0.12, 0.03, 0.08])
_CERT_PROBS = np.array([0.1, 0.25, 0.25, 0.25, 0.15])

_TREATMENTS = np.array(['control', 'T0_unfair', 'T1_group_fair', 'T2_individual_fair'])


def generate_study1_data(n_participants, pairs_per_treatment, effect_sizes):
    """
    Simulate one Study 1 dataset.

    Each participant evaluates ``pairs_per_treatment`` candidate pairs in each of
    the four treatment conditions (control, biased AI, group-fair AI, individual-fair
    AI). Candidate attributes are drawn from empirical distributions. The binary
    outcome ``y`` (whether the first candidate is selected) is generated from a
    logistic model that includes participant-level random intercepts (ICC = 0.25)
    plus the treatment-by-gender and culture-by-treatment-by-gender effects passed
    in via ``effect_sizes``.
    """
    n_treatments = 4
    n_pairs_per_participant = n_treatments * pairs_per_treatment
    n_total_rows = n_participants * n_pairs_per_participant

    # Participant-level random intercept calibrated to ICC = 0.25
    sigma_within = np.pi ** 2 / 3
    target_icc = 0.25
    sigma_participant = np.sqrt(target_icc * sigma_within / (1 - target_icc))

    participant_random_effect = np.random.normal(0, sigma_participant, size=n_participants)
    is_individualism_per_participant = np.random.binomial(1, 0.5, size=n_participants).astype(np.int8)

    # Expand participant-level variables to one entry per row
    participant_id_arr = np.repeat(np.arange(n_participants), n_pairs_per_participant)
    random_effect_arr = participant_random_effect[participant_id_arr]
    is_individualism_arr = is_individualism_per_participant[participant_id_arr]

    # Treatment indicator: each participant sees the four conditions in order
    treatment_per_participant = np.repeat(np.arange(n_treatments), pairs_per_treatment)
    treatment_idx_arr = np.tile(treatment_per_participant, n_participants).astype(np.int8)

    # Candidate attributes: shape (n_total_rows, 2) for the two candidates per pair
    female = np.random.binomial(1, 0.5, size=(n_total_rows, 2)).astype(np.int8)
    internship = np.random.binomial(1, 0.483, size=(n_total_rows, 2)).astype(np.int8)
    certificates = np.random.choice(5, size=(n_total_rows, 2), p=_CERT_PROBS).astype(np.int8)
    univ_idx = np.random.choice(9, size=(n_total_rows, 2), p=_UNIVERSITY_PROBS)
    university_score = _UNIVERSITY_VALUES[univ_idx]
    gpa = np.clip(np.random.normal(80.934, 7.489, size=(n_total_rows, 2)), 36.4, 100.0)

    # Difference scores (Candidate 1 minus Candidate 2). Gender is coded as is_male.
    delta_gender = (1 - female[:, 0]) - (1 - female[:, 1])
    delta_internship = (internship[:, 0] - internship[:, 1]).astype(np.int8)
    delta_certificates = (certificates[:, 0] - certificates[:, 1]).astype(np.int8)
    delta_university = university_score[:, 0] - university_score[:, 1]
    delta_gpa = gpa[:, 0] - gpa[:, 1]

    # ---------- Compute log-odds for each row (vectorized) ----------
    # Base coefficients
    coef_gender = 1.519
    coef_intern = 0.360
    coef_cert = 0.169
    coef_univ = 1.0
    coef_gpa = 0.335 / 10

    log_odds = (
        coef_gender * delta_gender
        + coef_intern * delta_internship
        + coef_cert * delta_certificates
        + coef_univ * delta_university
        + coef_gpa * delta_gpa
        + random_effect_arr
    )

    # 2-way interactions: treatment effects on gender preference
    # treatment_idx_arr: 0=control, 1=T0_unfair, 2=T1_group_fair, 3=T2_individual_fair
    treatment_effect = np.zeros(n_total_rows)
    treatment_effect[treatment_idx_arr == 1] = effect_sizes['unfair_gender_interaction']
    treatment_effect[treatment_idx_arr == 2] = effect_sizes['group_fair_gender_interaction']
    treatment_effect[treatment_idx_arr == 3] = effect_sizes['individual_fair_gender_interaction']
    log_odds = log_odds + treatment_effect * delta_gender

    # 3-way interactions: culture × treatment × gender
    culture_effect = np.zeros(n_total_rows)
    culture_effect[(treatment_idx_arr == 1) & (is_individualism_arr == 1)] = effect_sizes['culture_unfair_gender']
    culture_effect[(treatment_idx_arr == 2) & (is_individualism_arr == 1)] = effect_sizes['culture_group_gender']
    culture_effect[(treatment_idx_arr == 3) & (is_individualism_arr == 1)] = effect_sizes['culture_individual_gender']
    log_odds = log_odds + culture_effect * delta_gender

    # Convert to probability and sample y
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    y = np.random.binomial(1, prob).astype(np.int8)

    # ---------- Build DataFrame in one shot ----------
    treatment_arr = _TREATMENTS[treatment_idx_arr]
    df = pd.DataFrame({
        'y': y,
        'delta_gender': delta_gender.astype(np.int8),
        'delta_internship_exp': delta_internship,
        'delta_certificates': delta_certificates,
        'delta_university': delta_university,
        'delta_gpa': delta_gpa,
        'treatment': treatment_arr,
        'is_individualism': is_individualism_arr,
        'participant_id': participant_id_arr,
    })
    return df


# ============================================================
# Hypothesis tests
# ============================================================
#
# All tests use logistic regression with participant fixed effects (absorbed
# via pyfixest's within-cluster demeaning) and cluster-robust standard errors
# at the participant level. Pre-computed interaction columns avoid relying on
# pyfixest's formula parser for `:` interactions.

def _prepare_main_model_data(df):
    """
    Add pre-computed columns required by the H1/H2/H3 model:
      - Treatment dummies (excluding the control reference)
      - delta_gender x treatment interactions
    The participant fixed effects are absorbed by pyfixest, so no participant
    dummies are added here.
    """
    df_test = pd.get_dummies(df, columns=['treatment'], prefix='T', drop_first=False)
    df_test = df_test.drop(columns=['T_control'])
    for col in ['T_T0_unfair', 'T_T1_group_fair', 'T_T2_individual_fair']:
        df_test[col] = df_test[col].astype(float)
    df_test['dg_T0'] = df_test['delta_gender'] * df_test['T_T0_unfair']
    df_test['dg_T1'] = df_test['delta_gender'] * df_test['T_T1_group_fair']
    df_test['dg_T2'] = df_test['delta_gender'] * df_test['T_T2_individual_fair']
    return df_test


def fit_main_model(df_test):
    """Fit the H1/H2/H3 model with participant fixed effects and cluster-robust SE."""
    formula = (
        "y ~ delta_gender + delta_internship_exp + delta_certificates + "
        "delta_university + delta_gpa + "
        "T_T0_unfair + T_T1_group_fair + T_T2_individual_fair + "
        "dg_T0 + dg_T1 + dg_T2 | participant_id"
    )
    return pf.feglm(fml=formula, data=df_test, family="logit",
                     vcov={"CRV1": "participant_id"})


def get_coef_se_p(model, term):
    """Return (coef, se, p) for a single named coefficient, or (0, inf, 1) if missing."""
    coefs = model.coef()
    if term not in coefs.index:
        return 0.0, np.inf, 1.0
    coef = float(coefs.loc[term])
    se = float(model.se().loc[term])
    p = float(model.pvalue().loc[term])
    return coef, se, p


def contrast_test(model, term_a, term_b):
    """
    Two-tailed Wald test for the linear contrast (coef_a - coef_b).
    Returns (p_value, difference). Falls back to (1.0, 0.0) on missing terms.
    """
    coefs = model.coef()
    if term_a not in coefs.index or term_b not in coefs.index:
        return 1.0, 0.0
    diff = float(coefs.loc[term_a] - coefs.loc[term_b])
    vcov = pd.DataFrame(model._vcov, index=coefs.index, columns=coefs.index)
    var_a = vcov.loc[term_a, term_a]
    var_b = vcov.loc[term_b, term_b]
    cov_ab = vcov.loc[term_a, term_b]
    se_diff = np.sqrt(var_a + var_b - 2 * cov_ab)
    if se_diff <= 0 or not np.isfinite(se_diff):
        return 1.0, diff
    z = diff / se_diff
    p_two_tailed = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_two_tailed, diff


# ============================================================
# Hypothesis tests (H1-H5)
# ============================================================

def test_hypothesis_h1(df, alpha=0.05):
    """
    Test H1: fair AI reduces gender bias.

    H1 is supported when, with Holm-Bonferroni correction across the four
    contrasts in this family:
      (a) at least one of (T1 vs Control, T2 vs Control) is significant with a
          negative coefficient, and
      (b) at least one of (T1 vs T0, T2 vs T0) is significant with a negative
          coefficient.
    """
    try:
        df_test = _prepare_main_model_data(df)
        model = fit_main_model(df_test)

        coef_t1, _, p_t1_vs_ctrl = get_coef_se_p(model, 'dg_T1')
        coef_t2, _, p_t2_vs_ctrl = get_coef_se_p(model, 'dg_T2')
        p_t1_vs_t0, diff_t1_t0 = contrast_test(model, 'dg_T1', 'dg_T0')
        p_t2_vs_t0, diff_t2_t0 = contrast_test(model, 'dg_T2', 'dg_T0')

        p_values = [p_t1_vs_ctrl, p_t2_vs_ctrl, p_t1_vs_t0, p_t2_vs_t0]
        coefs = [coef_t1, coef_t2, diff_t1_t0, diff_t2_t0]
        reject, _, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        ctrl_ok = (reject[0] and coefs[0] < 0) or (reject[1] and coefs[1] < 0)
        t0_ok = (reject[2] and coefs[2] < 0) or (reject[3] and coefs[3] < 0)
        return ctrl_ok and t0_ok
    except Exception:
        return False


def test_hypothesis_h2(df, alpha=0.05):
    """
    Test H2: biased AI amplifies gender bias.

    H2 is supported when, with Holm-Bonferroni correction across the three
    contrasts in this family:
      (a) T0 vs Control is significant with a positive coefficient, and
      (b) at least one of (T0 vs T1, T0 vs T2) is significant with a positive
          coefficient.
    """
    try:
        df_test = _prepare_main_model_data(df)
        model = fit_main_model(df_test)

        coef_t0, _, p_t0_vs_ctrl = get_coef_se_p(model, 'dg_T0')
        p_t0_vs_t1, diff_t0_t1 = contrast_test(model, 'dg_T0', 'dg_T1')
        p_t0_vs_t2, diff_t0_t2 = contrast_test(model, 'dg_T0', 'dg_T2')

        p_values = [p_t0_vs_ctrl, p_t0_vs_t1, p_t0_vs_t2]
        coefs = [coef_t0, diff_t0_t1, diff_t0_t2]
        reject, _, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        ctrl_ok = reject[0] and coefs[0] > 0
        fair_ok = (reject[1] and coefs[1] > 0) or (reject[2] and coefs[2] > 0)
        return ctrl_ok and fair_ok
    except Exception:
        return False


def test_hypothesis_h3(df, alpha=0.05):
    """
    Test H3: the two fair AI definitions differ in effectiveness.

    Two-tailed test of the difference between the T1 and T2 gender-interaction
    coefficients. A negative difference indicates that individual fairness is
    more effective; a positive difference indicates that group fairness is more
    effective.
    """
    try:
        df_test = _prepare_main_model_data(df)
        model = fit_main_model(df_test)
        p_two_tailed, _ = contrast_test(model, 'dg_T1', 'dg_T2')
        return p_two_tailed < alpha
    except Exception:
        return False


def _prepare_culture_model_data(df):
    """
    Add the columns required by the H4/H5 model (treatment x culture x gender).

    Note: the main effect of `is_individualism` is collinear with the participant
    fixed effects and is therefore omitted. All identified terms have
    within-subject variation.
    """
    df_test = _prepare_main_model_data(df)
    df_test['is_ind'] = df_test['is_individualism'].astype(float)
    df_test['T0_ind'] = df_test['T_T0_unfair'] * df_test['is_ind']
    df_test['T1_ind'] = df_test['T_T1_group_fair'] * df_test['is_ind']
    df_test['T2_ind'] = df_test['T_T2_individual_fair'] * df_test['is_ind']
    df_test['dg_ind'] = df_test['delta_gender'] * df_test['is_ind']
    df_test['dg_T0_ind'] = df_test['dg_T0'] * df_test['is_ind']
    df_test['dg_T1_ind'] = df_test['dg_T1'] * df_test['is_ind']
    df_test['dg_T2_ind'] = df_test['dg_T2'] * df_test['is_ind']
    return df_test


def fit_culture_model(df_test):
    """Fit the H4/H5 model with participant fixed effects and cluster-robust SE."""
    formula = (
        "y ~ delta_gender + delta_internship_exp + delta_certificates + "
        "delta_university + delta_gpa + "
        "T_T0_unfair + T_T1_group_fair + T_T2_individual_fair + "
        "dg_T0 + dg_T1 + dg_T2 + "
        "T0_ind + T1_ind + T2_ind + dg_ind + "
        "dg_T0_ind + dg_T1_ind + dg_T2_ind | participant_id"
    )
    return pf.feglm(fml=formula, data=df_test, family="logit",
                     vcov={"CRV1": "participant_id"})


def test_hypothesis_h4(df, alpha=0.05):
    """
    Test H4: individual fairness is more effective in individualist cultures.

    Tests the three-way interaction delta_gender x T2 x is_individualism.
    A negative coefficient supports H4.
    """
    try:
        df_test = _prepare_culture_model_data(df)
        model = fit_culture_model(df_test)
        coef, _, p = get_coef_se_p(model, 'dg_T2_ind')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


def test_hypothesis_h5(df, alpha=0.05):
    """
    Test H5: group fairness is more effective in collectivist cultures
    (equivalently, less effective in individualist cultures).

    Tests the three-way interaction delta_gender x T1 x is_individualism.
    A positive coefficient supports H5.
    """
    try:
        df_test = _prepare_culture_model_data(df)
        model = fit_culture_model(df_test)
        coef, _, p = get_coef_se_p(model, 'dg_T1_ind')
        return (p < alpha) and (coef > 0)
    except Exception:
        return False


# ============================================================
# Simulation driver
# ============================================================

def simulate_study1_power_analysis(n_participants_list, pairs_per_treatment, effect_sizes,
                                    n_sim=1000, alpha=0.05, mode='all'):
    """
    Run the simulation across a list of sample sizes.

    Parameters
    ----------
    mode : {'all', 'h1h2h3', 'h4h5'}
        Selects which hypotheses to evaluate. Hypotheses not in the selected
        mode are reported as None in the output.
    """
    if mode not in ('all', 'h1h2h3', 'h4h5'):
        raise ValueError(f"mode must be 'all', 'h1h2h3', or 'h4h5'; got {mode!r}")

    # Decide which hypotheses to test
    if mode == 'all':
        hypotheses_to_test = ['h1', 'h2', 'h3', 'h4', 'h5']
    elif mode == 'h1h2h3':
        hypotheses_to_test = ['h1', 'h2', 'h3']
    else:  # h4h5
        hypotheses_to_test = ['h4', 'h5']

    results = {
        'n_participants': n_participants_list,
        'mode': mode,
        'convergence_failures': [], 'exceptions': []
    }
    for h in ['h1', 'h2', 'h3', 'h4', 'h5']:
        results[f'power_{h}'] = []

    print(f"Study 1 Power Analysis — mode='{mode}'")
    print(f"Pairs per treatment: {pairs_per_treatment}")
    print(f"Hypotheses tested: {hypotheses_to_test}")
    if mode == 'h1h2h3':
        print("  H1: Fair AI reduces bias (4 contrasts, Holm-corrected)")
        print("  H2: Biased AI amplifies bias (3 contrasts, Holm-corrected)")
        print("  H3: Individual vs Group fairness differ (two-tailed)")
    elif mode == 'h4h5':
        print("  H4: Individual fairness more effective in individualist cultures")
        print("  H5: Group fairness more effective in collectivist cultures")
    else:
        print("  H1-H5: All hypotheses")

    test_funcs = {
        'h1': test_hypothesis_h1,
        'h2': test_hypothesis_h2,
        'h3': test_hypothesis_h3,
        'h4': test_hypothesis_h4,
        'h5': test_hypothesis_h5,
    }

    for n in n_participants_list:
        print(f"\nRunning simulation for {n} participants...")
        power_counts = {h: 0 for h in hypotheses_to_test}
        convergence_fails = 0
        exceptions = 0

        for sim in tqdm(range(n_sim), desc=f"n={n}", leave=False):
            try:
                df = generate_study1_data(n, pairs_per_treatment, effect_sizes)
                for h in hypotheses_to_test:
                    if test_funcs[h](df, alpha):
                        power_counts[h] += 1
            except Exception as e:
                if "Singular matrix" in str(e) or "convergence" in str(e).lower():
                    convergence_fails += 1
                else:
                    exceptions += 1
                continue

        valid_sims = n_sim - exceptions
        for h in ['h1', 'h2', 'h3', 'h4', 'h5']:
            if h in hypotheses_to_test and valid_sims > 0:
                results[f'power_{h}'].append(power_counts[h] / valid_sims)
            else:
                results[f'power_{h}'].append(None)
        results['convergence_failures'].append(convergence_fails / n_sim)
        results['exceptions'].append(exceptions / n_sim)

    return results


def plot_study1_power_curves(results, pairs_per_treatment, output_dir=None):
    """Plot power curves, skipping hypotheses that were not tested in this run."""
    hypotheses = [
        ('power_h1', 'H1: Fair AI Reduces Gender Bias', 'blue'),
        ('power_h2', 'H2: Unfair AI Amplifies Gender Bias', 'red'),
        ('power_h3', 'H3: Individual vs Group Fairness Differs', 'green'),
        ('power_h4', 'H4: Individual Fairness x Individualism', 'purple'),
        ('power_h5', 'H5: Group Fairness x Collectivism', 'orange')
    ]

    # Filter to hypotheses that have at least one non-None value
    active_hypotheses = [
        (k, t, c) for (k, t, c) in hypotheses
        if any(v is not None for v in results[k])
    ]

    if not active_hypotheses:
        print("  No hypotheses to plot.")
        return

    n_plots = len(active_hypotheses)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    plt.figure(figsize=(20, 6 * n_rows))

    for i, (power_key, title, color) in enumerate(active_hypotheses):
        plt.subplot(n_rows, n_cols, i + 1)
        # Filter out None values
        ns = [n for n, p in zip(results['n_participants'], results[power_key]) if p is not None]
        ps = [p for p in results[power_key] if p is not None]
        plt.plot(ns, ps, 'o-', linewidth=2, markersize=6, color=color)
        plt.axhline(0.8, color='orange', linestyle='--', alpha=0.7, label='Power = 0.8')
        plt.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='Power = 0.95')
        plt.title(f'{title}\n({pairs_per_treatment} pairs per treatment)', fontsize=11)
        plt.xlabel('Number of Participants')
        plt.ylabel('Estimated Power')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        png_path = os.path.join(output_dir, "power_curves.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"  Saved power curves: {png_path}")
        pdf_path = os.path.join(output_dir, "power_curves.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved power curves (PDF): {pdf_path}")

    plt.show()


def find_study1_sample_requirements(results, pairs_per_treatment):
    """Print the smallest sample size at which each tested hypothesis reaches 80% and 95% power."""
    hypotheses = [
        ('power_h1', 'H1: Fair AI Reduces Gender Bias'),
        ('power_h2', 'H2: Unfair AI Amplifies Gender Bias'),
        ('power_h3', 'H3: Individual vs Group Fairness Differs'),
        ('power_h4', 'H4: Individual Fairness x Individualism'),
        ('power_h5', 'H5: Group Fairness x Collectivism')
    ]
    for target_power in [0.8, 0.95]:
        print(f"\n=== {target_power*100}% Power Requirements ===")
        for power_key, hypothesis_name in hypotheses:
            powers = results[power_key]
            # Skip hypotheses with all None values (not tested)
            if all(p is None for p in powers):
                continue
            found = False
            for i, power in enumerate(powers):
                if power is not None and power >= target_power:
                    n_participants = results['n_participants'][i]
                    total_pairs = n_participants * pairs_per_treatment * 4
                    print(f"{hypothesis_name}: {n_participants} participants "
                          f"({total_pairs:,} total pair evaluations)")
                    found = True
                    break
            if not found:
                max_n = max(results['n_participants'])
                valid_powers = [p for p in powers if p is not None]
                max_power = max(valid_powers) if valid_powers else 0
                print(f"{hypothesis_name}: >{max_n} participants needed "
                      f"(max observed power: {max_power:.3f})")


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Study 1 power analysis simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # H1, H2, H3 over a smaller sample range
  python Study1_PowerAnalysis.py --mode h1h2h3

  # H4, H5 over a larger sample range
  python Study1_PowerAnalysis.py --mode h4h5

  # All hypotheses
  python Study1_PowerAnalysis.py --mode all

  # Quick check with fewer simulations
  python Study1_PowerAnalysis.py --mode h1h2h3 --n_sim 50
"""
    )
    parser.add_argument('--mode', choices=['all', 'h1h2h3', 'h4h5'], default='h1h2h3',
                        help="Which hypotheses to test (default: h1h2h3)")
    parser.add_argument('--n_sim', type=int, default=1000,
                        help="Number of simulations per sample size (default: 1000)")
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument('--n_min', type=int, default=None,
                        help="Override the minimum sample size")
    parser.add_argument('--n_max', type=int, default=None,
                        help="Override the maximum sample size")
    parser.add_argument('--n_step', type=int, default=None,
                        help="Override the sample size step")
    args = parser.parse_args()

    pairs_per_treatment = 5

    # Default sample size ranges per mode
    if args.mode == 'h1h2h3':
        default_min, default_max, default_step = 50, 500, 50
    elif args.mode == 'h4h5':
        default_min, default_max, default_step = 800, 1500, 100
    else:
        default_min, default_max, default_step = 100, 1500, 100

    n_min = args.n_min if args.n_min is not None else default_min
    n_max = args.n_max if args.n_max is not None else default_max
    n_step = args.n_step if args.n_step is not None else default_step
    participants_range = list(range(n_min, n_max + 1, n_step))

    # Effect sizes on the log-odds scale
    # OR=1.72 (Cohen's d=0.3) for H1, H2, H4, H5; OR=2.48 (Cohen's d=0.5) for the H3 contrast
    effect_sizes = {
        'group_fair_gender_interaction': -0.542,
        'individual_fair_gender_interaction': -0.907,
        'unfair_gender_interaction': 0.542,
        'culture_individual_gender': -0.542,
        'culture_group_gender': 0.542,
        'culture_unfair_gender': 0,
    }

    n_sim = args.n_sim
    alpha = args.alpha
    mode = args.mode

    output_dir = setup_output_dir("study1", mode=mode)
    print(f"Results will be saved to: {output_dir}/")

    print(f"=== STUDY 1 POWER ANALYSIS — mode='{mode}' ===")
    print(f"Participants range: {participants_range}")
    print(f"Pairs per treatment per participant: {pairs_per_treatment}")
    print(f"Number of simulations: {n_sim}")
    print("Effect sizes (log-odds):")
    for key, value in effect_sizes.items():
        print(f"  {key}: {value:.3f}")

    results = simulate_study1_power_analysis(
        n_participants_list=participants_range,
        pairs_per_treatment=pairs_per_treatment,
        effect_sizes=effect_sizes,
        n_sim=n_sim,
        alpha=alpha,
        mode=mode
    )

    print("\n=== SAVING RESULTS ===")
    save_results(results, output_dir, effect_sizes, pairs_per_treatment, n_sim, alpha, mode=mode)
    plot_study1_power_curves(results, pairs_per_treatment, output_dir=output_dir)
    find_study1_sample_requirements(results, pairs_per_treatment)

    print(f"\n=== DETAILED RESULTS ===")
    tested = [h for h in ['h1', 'h2', 'h3', 'h4', 'h5']
              if any(v is not None for v in results[f'power_{h}'])]
    for i, n in enumerate(results['n_participants']):
        total_pairs = n * pairs_per_treatment * 4
        parts = [f"n={n:4d} ({total_pairs:5d} pairs):"]
        for h in tested:
            val = results[f'power_{h}'][i]
            parts.append(f"{h.upper()}={val:.3f}" if val is not None else f"{h.upper()}=N/A")
        print(" ".join(parts))

    print(f"\nAll results saved in: {output_dir}/")
