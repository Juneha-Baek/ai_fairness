"""
Power analysis for Study 2: AI fairness, team decision-making, and gender bias
in hiring decisions.

This module simulates a mixed-factorial design in which each participant first
makes individual hiring decisions and then participates in a three-person group
that makes additional decisions. AI presence (between-subjects, 2 levels) and
decision-making format (within-subjects: individual vs group) are crossed with
group composition (homogeneous vs mixed gender) and a binary indicator of
shared responsibility.

Hypotheses
----------
H6   Teams exhibit lower gender bias than individuals.
H7   Teams with fair AI exhibit lower bias than teams without AI.
H8   The bias-reducing effect of fair AI is stronger in team than in individual
     decisions (three-way interaction).
H9   Mixed-gender teams exhibit lower bias than homogeneous teams.
H10  Mixed-gender teams benefit more from fair AI than homogeneous teams
     (three-way interaction).
H11  Shared responsibility (median split) moderates the bias-reducing effect of
     fair AI (three-way interaction).

Unit of analysis
----------------
Each row in the simulated data corresponds to one decision made by one
"evaluation team":
  - Individual phase: an evaluation team of size 1 (the participant alone).
  - Group phase: an evaluation team of size 3 (a three-person group). One
    decision per pair, not replicated across members.

Each row carries a unique `team_id` that identifies the evaluation team that
produced the decision. Individual-phase teams have `team_id` equal to the
participant's index; group-phase teams have `team_id` offset by the number of
participants so that the two namespaces do not collide.

Estimation
----------
All hypothesis tests are estimated by logistic regression with team
(decision-maker) fixed effects absorbed via pyfixest's within-cluster
demeaning, and cluster-robust standard errors at the team level. The focal
interactions for all hypotheses involve `delta_gender`, which has within-
team variation, and are therefore identified after absorbing team fixed
effects. Main effects of team-level variables (e.g., `is_group`, `has_ai`,
`is_mixed`, `high_responsibility`) are absorbed but are not the focal terms.

Usage
-----
    # Test H6, H7, H8, H9 with a smaller sample range
    python Study2_PowerAnalysis.py --mode h6789

    # Test H10, H11 with a larger sample range
    python Study2_PowerAnalysis.py --mode h1011

    # Test all hypotheses
    python Study2_PowerAnalysis.py --mode all

Run `python Study2_PowerAnalysis.py --help` for the full list of options.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Candidate-attribute distributions
# ============================================================

_UNIVERSITY_VALUES = np.array([0.0, 0.897, 1.136, 1.309, 1.400, 1.881, 2.348, 2.525, 3.525])
_UNIVERSITY_PROBS = np.array([0.35, 0.05, 0.02, 0.18, 0.02, 0.15, 0.12, 0.03, 0.08])
_CERT_PROBS = np.array([0.1, 0.25, 0.25, 0.25, 0.15])


# ============================================================
# Output utilities
# ============================================================

def setup_output_dir(study_name="study2", mode=None):
    """Create a timestamped directory to store results from one run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode and mode != 'all':
        output_dir = f"results_{study_name}_{mode}_{timestamp}"
    else:
        output_dir = f"results_{study_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(results, output_dir, effect_sizes, pairs_indiv, pairs_group, n_sim, alpha, mode='all'):
    """Write power results, run metadata, and a human-readable report to disk."""
    df_results = pd.DataFrame({
        'n_participants': results['n_participants'],
        'power_h6': results['power_h6'],
        'power_h7': results['power_h7'],
        'power_h8': results['power_h8'],
        'power_h9': results['power_h9'],
        'power_h10': results['power_h10'],
        'power_h11': results['power_h11'],
        'convergence_failures': results['convergence_failures'],
        'exceptions': results['exceptions']
    })
    csv_path = os.path.join(output_dir, "power_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved power results: {csv_path}")

    tested = [h for h in ['h6', 'h7', 'h8', 'h9', 'h10', 'h11']
              if any(v is not None for v in results[f'power_{h}'])]

    metadata = {
        'study': 'Study 2',
        'mode': mode,
        'hypotheses_tested': tested,
        'timestamp': datetime.now().isoformat(),
        'n_simulations_per_n': n_sim,
        'alpha': alpha,
        'pairs_per_participant_indiv': pairs_indiv,
        'pairs_per_participant_group': pairs_group,
        'participants_range': list(results['n_participants']),
        'effect_sizes': effect_sizes,
    }
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {json_path}")

    txt_path = os.path.join(output_dir, "summary_report.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"STUDY 2 POWER ANALYSIS - SUMMARY REPORT (mode='{mode}')\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Hypotheses tested: {tested}\n")
        f.write(f"Number of simulations per sample size: {n_sim}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Individual pairs per participant: {pairs_indiv}\n")
        f.write(f"Group pairs per group: {pairs_group}\n\n")

        f.write("EFFECT SIZES (log-odds):\n")
        for key, value in effect_sizes.items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\nDETAILED POWER RESULTS:\n")
        col_label_map = {'h6': 'H6', 'h7': 'H7', 'h8': 'H8', 'h9': 'H9',
                         'h10': 'H10', 'h11': 'H11'}
        header_cols = [col_label_map[h] for h in tested]
        header = f"{'n':>5} " + " ".join(f"{c:>8}" for c in header_cols)
        f.write(header + "\n")
        for i, n in enumerate(results['n_participants']):
            row = f"{n:>5} "
            for h in tested:
                val = results[f'power_{h}'][i]
                row += f"{val:>8.3f} " if val is not None else f"{'N/A':>8} "
            f.write(row + "\n")

        f.write("\nSAMPLE SIZE REQUIREMENTS:\n")
        for target_power in [0.8, 0.95]:
            f.write(f"\n  --- {target_power*100:.0f}% Power ---\n")
            for h in tested:
                h_name = col_label_map[h]
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

def create_balanced_groups_with_responsibility(n_participants):
    """
    Build a balanced 2 x 2 (AI x responsibility) assignment of three-person groups.

    Group composition is balanced across MMM, MMF, MFF, and FFF in equal
    proportions. Within each composition, the four (AI x responsibility) cells
    are assigned in rotation.
    """
    if n_participants % 12 != 0:
        raise ValueError("Number of participants must be multiple of 12 for balanced design")

    n_groups = n_participants // 3
    if n_groups % 4 != 0:
        raise ValueError("Number of groups must be multiple of 4 for balanced 2x2 design")

    groups_per_condition = n_groups // 4

    participants = []
    for i in range(n_participants):
        participants.append({
            'participant_id': i,
            'gender': i % 2,
        })

    males = [p for p in participants if p['gender'] == 0]
    females = [p for p in participants if p['gender'] == 1]
    np.random.shuffle(males)
    np.random.shuffle(females)

    compositions = [
        ('MMM', 3, 0),
        ('MMF', 2, 1),
        ('MFF', 1, 2),
        ('FFF', 0, 3)
    ]

    groups = []
    male_idx = 0
    female_idx = 0
    group_id = 0

    conditions = [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ]

    for comp_name, n_males, n_females in compositions:
        groups_per_comp = groups_per_condition
        condition_idx = 0
        for _ in range(groups_per_comp):
            group_members = []
            for _ in range(n_males):
                if male_idx < len(males):
                    group_members.append(males[male_idx])
                    male_idx += 1
            for _ in range(n_females):
                if female_idx < len(females):
                    group_members.append(females[female_idx])
                    female_idx += 1

            group_has_ai, group_high_responsibility = conditions[condition_idx]
            condition_idx = (condition_idx + 1) % 4

            is_mixed = comp_name in ['MMF', 'MFF']
            composition_type = 'mixed' if is_mixed else 'homogeneous'

            groups.append({
                'group_id': group_id,
                'members': group_members,
                'composition': composition_type,
                'composition_detailed': comp_name,
                'has_ai': group_has_ai,
                'high_responsibility': group_high_responsibility
            })
            group_id += 1

    np.random.shuffle(groups)
    return groups


def generate_study2_data(n_participants, pairs_per_participant_indiv,
                          pairs_per_participant_group, effect_sizes):
    """
    Simulate one Study 2 dataset.

    Each participant first evaluates ``pairs_per_participant_indiv`` candidate
    pairs individually, then joins a three-person group that evaluates
    ``pairs_per_participant_group`` additional pairs collectively. Group
    composition (homogeneous vs mixed gender), AI presence, and shared-
    responsibility level are crossed in a balanced 2 x 2 design at the group
    level. Outcomes are generated from a logistic model with participant- and
    group-level random intercepts (ICCs of 0.25 and 0.10 respectively).
    """
    if n_participants % 12 != 0:
        raise ValueError("Number of participants must be a multiple of 12")

    n_groups = n_participants // 3
    if n_groups % 4 != 0:
        raise ValueError("Number of groups must be a multiple of 4 for the balanced 2x2 design")

    groups = create_balanced_groups_with_responsibility(n_participants)

    # Participant-level lookup arrays
    participant_gender = np.zeros(n_participants, dtype=np.int8)
    participant_group_id = np.zeros(n_participants, dtype=np.int32)
    participant_has_ai = np.zeros(n_participants, dtype=bool)
    participant_high_resp = np.zeros(n_participants, dtype=bool)
    participant_is_mixed = np.zeros(n_participants, dtype=np.int8)

    for group in groups:
        gid = group['group_id']
        is_mixed = 1 if group['composition'] == 'mixed' else 0
        for member in group['members']:
            pid = member['participant_id']
            participant_gender[pid] = member['gender']
            participant_group_id[pid] = gid
            participant_has_ai[pid] = group['has_ai']
            participant_high_resp[pid] = group['high_responsibility']
            participant_is_mixed[pid] = is_mixed

    # Group-level lookup arrays
    group_has_ai = np.zeros(n_groups, dtype=bool)
    group_high_resp = np.zeros(n_groups, dtype=bool)
    group_is_mixed = np.zeros(n_groups, dtype=np.int8)
    for group in groups:
        gid = group['group_id']
        group_has_ai[gid] = group['has_ai']
        group_high_resp[gid] = group['high_responsibility']
        group_is_mixed[gid] = 1 if group['composition'] == 'mixed' else 0

    # Random intercepts calibrated to the target ICCs
    sigma_within = np.pi ** 2 / 3
    target_icc_indiv = 0.25
    target_icc_group = 0.1
    sigma_participant_indiv = np.sqrt(target_icc_indiv * sigma_within / (1 - target_icc_indiv))
    sigma_group = np.sqrt(target_icc_group * sigma_within / (1 - target_icc_group))

    participant_random_effect = np.random.normal(0, sigma_participant_indiv, size=n_participants)
    group_random_effect = np.random.normal(0, sigma_group, size=n_groups)

    def _gen_pairs(n_pairs):
        """Generate n_pairs candidate pairs and return their delta-score arrays."""
        female = np.random.binomial(1, 0.5, size=(n_pairs, 2)).astype(np.int8)
        internship = np.random.binomial(1, 0.483, size=(n_pairs, 2)).astype(np.int8)
        certificates = np.random.choice(5, size=(n_pairs, 2), p=_CERT_PROBS).astype(np.int8)
        univ_idx = np.random.choice(9, size=(n_pairs, 2), p=_UNIVERSITY_PROBS)
        university_score = _UNIVERSITY_VALUES[univ_idx]
        gpa = np.clip(np.random.normal(80.934, 7.489, size=(n_pairs, 2)), 36.4, 100.0)
        d_gender = ((1 - female[:, 0]) - (1 - female[:, 1])).astype(np.int8)
        d_intern = (internship[:, 0] - internship[:, 1]).astype(np.int8)
        d_cert = (certificates[:, 0] - certificates[:, 1]).astype(np.int8)
        d_univ = university_score[:, 0] - university_score[:, 1]
        d_gpa = gpa[:, 0] - gpa[:, 1]
        return d_gender, d_intern, d_cert, d_univ, d_gpa

    ind_dg, ind_di, ind_dc, ind_du, ind_dgpa = _gen_pairs(pairs_per_participant_indiv)
    grp_dg, grp_di, grp_dc, grp_du, grp_dgpa = _gen_pairs(pairs_per_participant_group)

    # Base coefficients
    coef_gender = 1.519
    coef_intern = 0.360
    coef_cert = 0.169
    coef_univ = 1.0
    coef_gpa = 0.335 / 10

    # ============================================================
    # INDIVIDUAL PHASE: n_participants × pairs_per_participant_indiv rows
    # ============================================================
    n_ind_rows = n_participants * pairs_per_participant_indiv

    # Tile pair attributes: each participant evaluates the same pairs
    ind_pid = np.repeat(np.arange(n_participants), pairs_per_participant_indiv)
    ind_dg_tiled = np.tile(ind_dg, n_participants)
    ind_di_tiled = np.tile(ind_di, n_participants)
    ind_dc_tiled = np.tile(ind_dc, n_participants)
    ind_du_tiled = np.tile(ind_du, n_participants)
    ind_dgpa_tiled = np.tile(ind_dgpa, n_participants)

    # Per-participant attributes
    ind_has_ai = participant_has_ai[ind_pid]
    ind_random = participant_random_effect[ind_pid]
    ind_gender = participant_gender[ind_pid]
    ind_group_id = participant_group_id[ind_pid]

    # Individual-phase log-odds: no group-level effects apply here
    ind_log_odds = (
        coef_gender * ind_dg_tiled
        + coef_intern * ind_di_tiled
        + coef_cert * ind_dc_tiled
        + coef_univ * ind_du_tiled
        + coef_gpa * ind_dgpa_tiled
        + ind_random
    )
    ind_prob = np.clip(1.0 / (1.0 + np.exp(-ind_log_odds)), 1e-10, 1 - 1e-10)
    ind_y = np.random.binomial(1, ind_prob).astype(np.int8)

    # ----------------------------------------------------------------
    # Group phase: each group makes one decision per pair, then that
    # decision is replicated to all three members for the long-format
    # data frame.
    # ----------------------------------------------------------------
    n_group_decisions = n_groups * pairs_per_participant_group

    grp_gid = np.repeat(np.arange(n_groups), pairs_per_participant_group)
    grp_dg_tiled = np.tile(grp_dg, n_groups)
    grp_di_tiled = np.tile(grp_di, n_groups)
    grp_dc_tiled = np.tile(grp_dc, n_groups)
    grp_du_tiled = np.tile(grp_du, n_groups)
    grp_dgpa_tiled = np.tile(grp_dgpa, n_groups)

    grp_has_ai_arr = group_has_ai[grp_gid]
    grp_high_resp_arr = group_high_resp[grp_gid]
    grp_is_mixed_arr = group_is_mixed[grp_gid]
    grp_random = group_random_effect[grp_gid]

    # Centered-coding effects: each factor contributes +eff if "on" and -eff if "off"
    # so that the average across cells equals zero. This matches a standard ANOVA-
    # style effect coding for the gender-bias modifier.
    base_group = effect_sizes['group_reduces_bias']
    ai_eff = effect_sizes['ai_additional_effect']
    mixed_eff = effect_sizes['mixed_additional_effect']
    resp_eff = effect_sizes['responsibility_additional_effect']
    mixed_ai_syn = effect_sizes['mixed_ai_synergy']
    ai_resp_syn = effect_sizes['ai_responsibility_synergy']

    centered_ai = np.where(grp_has_ai_arr, ai_eff, -ai_eff)
    centered_mixed = np.where(grp_is_mixed_arr == 1, mixed_eff, -mixed_eff)
    centered_resp = np.where(grp_high_resp_arr, resp_eff, -resp_eff)
    centered_mixed_ai = np.where(grp_has_ai_arr & (grp_is_mixed_arr == 1), mixed_ai_syn, -mixed_ai_syn)
    centered_ai_resp = np.where(grp_has_ai_arr & grp_high_resp_arr, ai_resp_syn, -ai_resp_syn)

    total_gender_modifier = (base_group + centered_ai + centered_mixed + centered_resp
                             + centered_mixed_ai + centered_ai_resp)

    grp_log_odds = (
        coef_gender * grp_dg_tiled
        + coef_intern * grp_di_tiled
        + coef_cert * grp_dc_tiled
        + coef_univ * grp_du_tiled
        + coef_gpa * grp_dgpa_tiled
        + grp_random
        + total_gender_modifier * grp_dg_tiled
    )
    grp_prob = np.clip(1.0 / (1.0 + np.exp(-grp_log_odds)), 1e-10, 1 - 1e-10)
    grp_y = np.random.binomial(1, grp_prob).astype(np.int8)

    # Combine the two phases into one long-format data frame.
    # Each row is one decision by one evaluation team:
    #   - Individual phase: team of size 1 (the participant alone).
    #   - Group phase: team of size 3 (the three-person group); one row per
    #     pair, not replicated across members.
    # Team IDs are kept globally unique by offsetting group teams by the
    # number of participants.
    ind_team_id = ind_pid                      # one team per participant
    grp_team_id = grp_gid + n_participants     # one team per group, offset
    composition_str_grp = np.where(grp_is_mixed_arr == 1, 'mixed', 'homogeneous')

    df = pd.DataFrame({
        'team_id': np.concatenate([ind_team_id, grp_team_id]),
        'participant_id': np.concatenate([ind_pid, np.full(n_group_decisions, -1, dtype=np.int32)]),
        'y': np.concatenate([ind_y, grp_y]),
        'delta_gender': np.concatenate([ind_dg_tiled, grp_dg_tiled]).astype(np.int8),
        'delta_internship_exp': np.concatenate([ind_di_tiled, grp_di_tiled]).astype(np.int8),
        'delta_certificates': np.concatenate([ind_dc_tiled, grp_dc_tiled]).astype(np.int8),
        'delta_university': np.concatenate([ind_du_tiled, grp_du_tiled]),
        'delta_gpa': np.concatenate([ind_dgpa_tiled, grp_dgpa_tiled]),
        'has_ai': np.concatenate([ind_has_ai, grp_has_ai_arr]),
        'is_group': np.concatenate([np.zeros(n_ind_rows, dtype=np.int8),
                                     np.ones(n_group_decisions, dtype=np.int8)]),
        'group_composition': np.concatenate([
            np.array(['individual'] * n_ind_rows),
            composition_str_grp
        ]),
        'high_responsibility': np.concatenate([
            np.array([None] * n_ind_rows, dtype=object),
            grp_high_resp_arr.astype(object)
        ]),
        'group_id': np.concatenate([ind_group_id, grp_gid]),
    })
    return df


# ============================================================
# Hypothesis tests (H6-H11)
# ============================================================
#
# All tests use logistic regression with team-level fixed effects (absorbed
# via pyfixest's within-cluster demeaning) and cluster-robust standard
# errors at the team level. The focal interactions for every hypothesis
# involve `delta_gender`, which varies within team, so they are identified
# after absorbing team fixed effects. Pre-computed interaction columns avoid
# relying on pyfixest's `:` operator.

def _get_coef_p(model, term):
    """Return (coef, p) for a named coefficient, or (0, 1) if missing."""
    coefs = model.coef()
    if term not in coefs.index:
        return 0.0, 1.0
    return float(coefs.loc[term]), float(model.pvalue().loc[term])


def test_hypothesis_h6(df, alpha=0.05):
    """
    Test H6: teams exhibit lower gender bias than individuals.

    A negative coefficient on delta_gender x is_group supports H6. The
    interaction is identified from within-decision-maker variation in
    delta_gender; the main effect of is_group is absorbed by team fixed
    effects but is not the focal term.
    """
    try:
        df_test = df.copy()
        df_test['is_group_f'] = df_test['is_group'].astype(float)
        df_test['dg_isgrp'] = df_test['delta_gender'] * df_test['is_group_f']

        formula = (
            "y ~ delta_gender + delta_internship_exp + delta_certificates + "
            "delta_university + delta_gpa + dg_isgrp | team_id"
        )
        model = pf.feglm(fml=formula, data=df_test, family="logit",
                          vcov={"CRV1": "team_id"})
        coef, p = _get_coef_p(model, 'dg_isgrp')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


def test_hypothesis_h7(df, alpha=0.05):
    """
    Test H7: among team decisions, teams with fair AI exhibit lower gender bias
    than teams without AI.

    Team fixed effects absorb between-team variation; the interaction
    delta_gender x has_ai is identified by within-team variation through
    delta_gender. A negative coefficient supports H7.
    """
    try:
        df_g = df[df['is_group'] == 1].copy()
        df_g['has_ai_f'] = df_g['has_ai'].astype(float)
        df_g['dg_ai'] = df_g['delta_gender'] * df_g['has_ai_f']

        formula = (
            "y ~ delta_gender + delta_internship_exp + delta_certificates + "
            "delta_university + delta_gpa + dg_ai | team_id"
        )
        model = pf.feglm(fml=formula, data=df_g, family="logit",
                          vcov={"CRV1": "team_id"})
        coef, p = _get_coef_p(model, 'dg_ai')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


def test_hypothesis_h8(df, alpha=0.05):
    """
    Test H8: the bias-reducing effect of fair AI is more pronounced in team
    settings than in individual decisions.

    A negative coefficient on the three-way interaction
    delta_gender x is_group x has_ai supports H8. The interaction is
    identified from within-decision-maker variation in delta_gender; the
    main effects of is_group and has_ai are absorbed by team fixed effects
    but are not the focal term.
    """
    try:
        df_test = df.copy()
        df_test['is_group_f'] = df_test['is_group'].astype(float)
        df_test['has_ai_f'] = df_test['has_ai'].astype(float)
        df_test['dg_isgrp'] = df_test['delta_gender'] * df_test['is_group_f']
        df_test['dg_ai'] = df_test['delta_gender'] * df_test['has_ai_f']
        df_test['dg_isgrp_ai'] = df_test['delta_gender'] * df_test['is_group_f'] * df_test['has_ai_f']

        formula = (
            "y ~ delta_gender + delta_internship_exp + delta_certificates + "
            "delta_university + delta_gpa + "
            "dg_isgrp + dg_ai + dg_isgrp_ai | team_id"
        )
        model = pf.feglm(fml=formula, data=df_test, family="logit",
                          vcov={"CRV1": "team_id"})
        coef, p = _get_coef_p(model, 'dg_isgrp_ai')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


def test_hypothesis_h9(df, alpha=0.05):
    """
    Test H9: among team decisions, mixed-gender teams exhibit lower gender bias
    than homogeneous teams, regardless of AI presence.

    Team fixed effects absorb between-team variation. AI presence is included
    as a within-team interaction control. A negative coefficient on
    delta_gender x is_mixed supports H9.
    """
    try:
        df_g = df[df['is_group'] == 1].copy()
        df_g['is_mixed'] = (df_g['group_composition'] == 'mixed').astype(float)
        df_g['has_ai_f'] = df_g['has_ai'].astype(float)
        df_g['dg_mixed'] = df_g['delta_gender'] * df_g['is_mixed']
        df_g['dg_ai'] = df_g['delta_gender'] * df_g['has_ai_f']

        formula = (
            "y ~ delta_gender + delta_internship_exp + delta_certificates + "
            "delta_university + delta_gpa + dg_mixed + dg_ai | team_id"
        )
        model = pf.feglm(fml=formula, data=df_g, family="logit",
                          vcov={"CRV1": "team_id"})
        coef, p = _get_coef_p(model, 'dg_mixed')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


def test_hypothesis_h10(df, alpha=0.05):
    """
    Test H10: among team decisions, mixed-gender teams benefit more from fair
    AI than homogeneous teams.

    A negative coefficient on the three-way interaction
    delta_gender x is_mixed x has_ai supports H10.
    """
    try:
        df_g = df[df['is_group'] == 1].copy()
        df_g['is_mixed'] = (df_g['group_composition'] == 'mixed').astype(float)
        df_g['has_ai_f'] = df_g['has_ai'].astype(float)
        df_g['dg_mixed'] = df_g['delta_gender'] * df_g['is_mixed']
        df_g['dg_ai'] = df_g['delta_gender'] * df_g['has_ai_f']
        df_g['dg_mixed_ai'] = df_g['delta_gender'] * df_g['is_mixed'] * df_g['has_ai_f']

        formula = (
            "y ~ delta_gender + delta_internship_exp + delta_certificates + "
            "delta_university + delta_gpa + "
            "dg_mixed + dg_ai + dg_mixed_ai | team_id"
        )
        model = pf.feglm(fml=formula, data=df_g, family="logit",
                          vcov={"CRV1": "team_id"})
        coef, p = _get_coef_p(model, 'dg_mixed_ai')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


def test_hypothesis_h11(df, alpha=0.05):
    """
    Test H11: shared responsibility (median split) moderates the bias-reducing
    effect of fair AI in team decisions.

    A negative coefficient on the three-way interaction
    delta_gender x has_ai x high_responsibility supports H11.
    """
    try:
        df_g = df[df['is_group'] == 1].copy()
        df_g = df_g.dropna(subset=['high_responsibility'])
        if len(df_g) == 0:
            return False
        df_g['high_resp'] = df_g['high_responsibility'].astype(float)
        df_g['has_ai_f'] = df_g['has_ai'].astype(float)
        df_g['dg_ai'] = df_g['delta_gender'] * df_g['has_ai_f']
        df_g['dg_resp'] = df_g['delta_gender'] * df_g['high_resp']
        df_g['dg_ai_resp'] = df_g['delta_gender'] * df_g['has_ai_f'] * df_g['high_resp']

        formula = (
            "y ~ delta_gender + delta_internship_exp + delta_certificates + "
            "delta_university + delta_gpa + "
            "dg_ai + dg_resp + dg_ai_resp | team_id"
        )
        model = pf.feglm(fml=formula, data=df_g, family="logit",
                          vcov={"CRV1": "team_id"})
        coef, p = _get_coef_p(model, 'dg_ai_resp')
        return (p < alpha) and (coef < 0)
    except Exception:
        return False


# ============================================================
# Simulation driver
# ============================================================

def simulate_study2_power_analysis(n_participants_list, pairs_per_participant_indiv,
                                    pairs_per_participant_group, effect_sizes,
                                    n_sim=1000, alpha=0.05, mode='all'):
    """
    Run the simulation across a list of sample sizes.

    Parameters
    ----------
    mode : {'all', 'h6789', 'h1011'}
        Selects which hypotheses to evaluate. Hypotheses not in the selected
        mode are reported as None in the output.
    """
    if mode not in ('all', 'h6789', 'h1011'):
        raise ValueError(f"mode must be 'all', 'h6789', or 'h1011'; got {mode!r}")

    if mode == 'all':
        hypotheses_to_test = ['h6', 'h7', 'h8', 'h9', 'h10', 'h11']
    elif mode == 'h6789':
        hypotheses_to_test = ['h6', 'h7', 'h8', 'h9']
    else:
        hypotheses_to_test = ['h10', 'h11']

    results = {
        'n_participants': n_participants_list,
        'mode': mode,
        'convergence_failures': [], 'exceptions': []
    }
    for h in ['h6', 'h7', 'h8', 'h9', 'h10', 'h11']:
        results[f'power_{h}'] = []

    print(f"Study 2 Power Analysis — mode='{mode}'")
    print(f"Individual pairs per participant: {pairs_per_participant_indiv}")
    print(f"Group pairs per group: {pairs_per_participant_group}")
    print(f"Hypotheses tested: {hypotheses_to_test}")

    test_funcs = {
        'h6': test_hypothesis_h6,
        'h7': test_hypothesis_h7,
        'h8': test_hypothesis_h8,
        'h9': test_hypothesis_h9,
        'h10': test_hypothesis_h10,
        'h11': test_hypothesis_h11,
    }

    for n in n_participants_list:
        if n % 12 != 0:
            print(f"WARNING: {n} participants not a multiple of 12. Adjusting to {n//12*12}")
            n = n // 12 * 12

        n_groups = n // 3
        print(f"\nRunning simulation for {n} participants ({n_groups} groups)...")

        power_counts = {h: 0 for h in hypotheses_to_test}
        convergence_fails = 0
        exceptions = 0

        for sim in tqdm(range(n_sim), desc=f"n={n}", leave=False):
            try:
                df = generate_study2_data(
                    n, pairs_per_participant_indiv, pairs_per_participant_group, effect_sizes
                )
                for col in ['has_ai', 'is_group', 'delta_gender']:
                    df[col] = df[col].astype(float)

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
        for h in ['h6', 'h7', 'h8', 'h9', 'h10', 'h11']:
            if h in hypotheses_to_test and valid_sims > 0:
                results[f'power_{h}'].append(power_counts[h] / valid_sims)
            else:
                results[f'power_{h}'].append(None)
        results['convergence_failures'].append(convergence_fails / n_sim)
        results['exceptions'].append(exceptions / n_sim)

    return results


def plot_study2_power_curves(results, pairs_per_participant_indiv, pairs_per_participant_group, output_dir=None):
    """Plot power curves, skipping hypotheses that were not tested in this run."""
    hypotheses = [
        ('power_h6', 'H6: Teams reduce gender bias vs individuals', 'blue'),
        ('power_h7', 'H7: AI reduces bias in teams', 'green'),
        ('power_h8', 'H8: Team x AI x gender interaction', 'red'),
        ('power_h9', 'H9: Mixed teams have lower bias', 'purple'),
        ('power_h10', 'H10: Mixed teams benefit more from AI', 'orange'),
        ('power_h11', 'H11: Responsibility moderates AI effect', 'brown'),
    ]

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
        ns = [n for n, p in zip(results['n_participants'], results[power_key]) if p is not None]
        ps = [p for p in results[power_key] if p is not None]
        plt.plot(ns, ps, 'o-', linewidth=2, markersize=6, color=color)
        plt.axhline(0.8, color='orange', linestyle='--', alpha=0.7, label='Power = 0.8')
        plt.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='Power = 0.95')
        plt.title(f'{title}\n(Indiv: {pairs_per_participant_indiv}, Grp: {pairs_per_participant_group})',
                  fontsize=10)
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


def find_study2_sample_requirements(results, pairs_per_participant_indiv, pairs_per_participant_group):
    """Print the smallest sample size at which each tested hypothesis reaches 80% and 95% power."""
    hypotheses = [
        ('power_h6', 'H6: Teams reduce gender bias vs individuals'),
        ('power_h7', 'H7: AI reduces bias in teams'),
        ('power_h8', 'H8: Team x AI x gender interaction'),
        ('power_h9', 'H9: Mixed teams have lower bias'),
        ('power_h10', 'H10: Mixed teams benefit more from AI'),
        ('power_h11', 'H11: Responsibility moderates AI effect'),
    ]
    for target_power in [0.8, 0.95]:
        print(f"\n=== {target_power*100}% Power Requirements ===")
        for power_key, hypothesis_name in hypotheses:
            powers = results[power_key]
            if all(p is None for p in powers):
                continue
            found = False
            for i, power in enumerate(powers):
                if power is not None and power >= target_power:
                    n_participants = results['n_participants'][i]
                    n_groups = n_participants // 3
                    individual_decisions = n_participants * pairs_per_participant_indiv
                    group_decisions = n_groups * pairs_per_participant_group
                    total_decisions = individual_decisions + group_decisions
                    print(f"{hypothesis_name}: {n_participants} participants "
                          f"({n_groups} groups, {total_decisions:,} decisions)")
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
        description="Study 2 power analysis simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # H6, H7, H8, H9 over a smaller sample range
  python Study2_PowerAnalysis.py --mode h6789

  # H10, H11 over a larger sample range
  python Study2_PowerAnalysis.py --mode h1011

  # All hypotheses
  python Study2_PowerAnalysis.py --mode all

  # Quick check with fewer simulations
  python Study2_PowerAnalysis.py --mode h6789 --n_sim 50
"""
    )
    parser.add_argument('--mode', choices=['all', 'h6789', 'h1011'], default='h6789',
                        help="Which hypotheses to test (default: h6789)")
    parser.add_argument('--n_sim', type=int, default=1000,
                        help="Number of simulations per sample size (default: 1000)")
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument('--n_list', type=str, default=None,
                        help="Override sample size list (comma-separated, e.g. '60,120,180')")
    args = parser.parse_args()

    pairs_per_participant_indiv = 10
    pairs_per_participant_group = 30

    # Default sample size ranges per mode. All values must be multiples of 12.
    if args.mode == 'h6789':
        default_range = [60, 120, 180]
    elif args.mode == 'h1011':
        default_range = [180, 240, 300, 360]
    else:
        default_range = [60, 120, 180, 240, 300, 360]

    if args.n_list is not None:
        participants_range = [int(x) for x in args.n_list.split(',')]
    else:
        participants_range = default_range
    participants_range = [n for n in participants_range if n % 12 == 0]

    # Effect sizes on the log-odds scale.
    # Cohen's d = 0.8 (OR ~ 4.27) for the main team and AI effects;
    # d = 0.3 (OR ~ 1.72) for the team-composition main effect.
    # Centered effects are halved so that the contrast between conditions equals the full effect.
    effect_sizes = {
        'group_reduces_bias': -1.451,
        'ai_additional_effect': -1.451 / 2,
        'mixed_additional_effect': -0.542 / 2,
        'responsibility_additional_effect': 0,
        'mixed_ai_synergy': -1.451 / 2,
        'ai_responsibility_synergy': -1.451 / 2,
    }

    n_sim = args.n_sim
    alpha = args.alpha
    mode = args.mode

    output_dir = setup_output_dir("study2", mode=mode)
    print(f"Results will be saved to: {output_dir}/")

    print(f"=== STUDY 2 POWER ANALYSIS — mode='{mode}' ===")
    print(f"Participants range: {participants_range}")
    print(f"Individual pairs per participant: {pairs_per_participant_indiv}")
    print(f"Group pairs per group: {pairs_per_participant_group}")
    print(f"Number of simulations: {n_sim}")
    print("Effect sizes:")
    for key, value in effect_sizes.items():
        print(f"  {key}: {value:.3f}")

    results = simulate_study2_power_analysis(
        n_participants_list=participants_range,
        pairs_per_participant_indiv=pairs_per_participant_indiv,
        pairs_per_participant_group=pairs_per_participant_group,
        effect_sizes=effect_sizes,
        n_sim=n_sim,
        alpha=alpha,
        mode=mode
    )

    print("\n=== SAVING RESULTS ===")
    save_results(results, output_dir, effect_sizes,
                 pairs_per_participant_indiv, pairs_per_participant_group, n_sim, alpha, mode=mode)
    plot_study2_power_curves(results, pairs_per_participant_indiv,
                              pairs_per_participant_group, output_dir=output_dir)
    find_study2_sample_requirements(results, pairs_per_participant_indiv, pairs_per_participant_group)

    print(f"\n=== DETAILED RESULTS ===")
    tested = [h for h in ['h6', 'h7', 'h8', 'h9', 'h10', 'h11']
              if any(v is not None for v in results[f'power_{h}'])]
    label_map = {'h6': 'H6', 'h7': 'H7', 'h8': 'H8', 'h9': 'H9', 'h10': 'H10', 'h11': 'H11'}
    for i, n in enumerate(results['n_participants']):
        n_groups = n // 3
        parts = [f"n={n:3d} ({n_groups:2d} grps):"]
        for h in tested:
            val = results[f'power_{h}'][i]
            parts.append(f"{label_map[h]}={val:.3f}" if val is not None else f"{label_map[h]}=N/A")
        print(" ".join(parts))

    print(f"\nAll results saved in: {output_dir}/")
