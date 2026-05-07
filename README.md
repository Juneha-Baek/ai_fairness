# Power Analysis: AI Fairness and Hiring Decisions

Simulation-based power analysis for the project **"Does a Fair Algorithm Lead
to Fair Hiring? The Role of AI Fairness Definitions, Culture, and Decision-
Making Structure."** The repository contains two simulation scripts (one per
study), a re-plot utility, and a Windows helper that runs all four analysis
modes in parallel.

## Studies and hypotheses

**Study 1** is an online individual-decision experiment in which each
participant evaluates paired hiring profiles in four AI conditions (no AI,
biased AI, group-fair AI, individual-fair AI) and a culture indicator is
measured between subjects.

| Hypothesis | Statement |
| --- | --- |
| H1 | Fair AI reduces gender bias relative to no AI and biased AI. |
| H2 | Biased AI amplifies gender bias relative to no AI and fair AI. |
| H3 | The two fair-AI definitions (group vs individual) differ in effectiveness. |
| H4 | Individual fairness is more effective in individualist cultures. |
| H5 | Group fairness is more effective in collectivist cultures. |

**Study 2** is an in-person team-decision experiment with a mixed factorial
design. Each participant first makes individual hiring decisions, then joins a
three-person group that makes additional decisions. AI presence (between),
group composition (homogeneous vs mixed gender, between at the group level),
and shared responsibility (between at the group level) are crossed in a
balanced 2 × 2 design.

| Hypothesis | Statement |
| --- | --- |
| H6  | Teams exhibit lower gender bias than individuals. |
| H7  | Among teams, those with fair AI exhibit lower bias than those without. |
| H8  | The bias-reducing effect of fair AI is stronger in teams than in individuals. |
| H9  | Mixed-gender teams exhibit lower bias than homogeneous teams. |
| H10 | Mixed-gender teams benefit more from fair AI than homogeneous teams. |
| H11 | Shared responsibility (median split) moderates the bias-reducing effect of fair AI. |

## Statistical model

Each iteration draws a synthetic dataset under the assumed effect sizes,
fits a fixed-effects logistic regression, and tests the relevant coefficient
or contrast. Power is the proportion of iterations in which the corresponding
hypothesis is supported, with a directional sign check on the coefficient
where the hypothesis is directional. For Study 1 H1 and H2, Holm-Bonferroni
correction is applied within the family of contrasts.

### Estimator

All hypothesis tests are estimated by logistic regression with:

- **Fixed effects** absorbed via the within-cluster demeaning algorithm
  implemented in `pyfixest`. Including a fixed effect for each cluster (e.g.
  each participant) controls for any time-invariant heterogeneity at that
  level — equivalent to including a dummy variable per cluster, but
  computationally efficient even with thousands of clusters.
- **Cluster-robust standard errors (CRV1)** at the same level as the fixed
  effects. This accounts for any remaining within-cluster correlation in the
  residuals (e.g., heteroskedasticity, residual serial correlation).

Because the cluster-level fixed effects absorb between-cluster variation,
the coefficient estimates are identified from within-cluster variation only.
Time-invariant between-cluster variables (e.g., a participant's culture) are
collinear with the fixed effects and cannot be estimated as main effects, but
their cross-level interactions with within-cluster variables are identified.

### Unit of analysis

In Study 2, each row of the simulated data corresponds to one decision made
by one *evaluation team*: a team of size 1 in the individual phase (the
participant alone) and a team of size 3 in the group phase. The group's
collective decision is recorded once per pair, not replicated across the
three members. Each row carries a unique `team_id`.

This unifies the two phases under a single decision-making unit and avoids
inflating the group-phase sample size by member-level replication.

### Fixed-effect specification

| Hypotheses | Fixed effects | Cluster |
| --- | --- | --- |
| Study 1 H1–H5 | Participant | Participant |
| Study 2 H6–H11 | Team (decision maker) | Team |

For all Study 2 hypotheses, the focal interactions involve `delta_gender`,
which has within-team variation and is therefore identified after absorbing
team fixed effects. Main effects of team-level variables (e.g., `is_group`,
`has_ai`) are absorbed by the fixed effects but are not the focal terms.

### Simulation parameters

| Setting | Value |
| --- | --- |
| Significance level | α = 0.05 |
| Simulations per sample size | 1,000 |
| Study 1 ICC (participant) | 0.25 |
| Study 2 ICC (participant / group) | 0.25 / 0.10 |
| Study 1 pairs per treatment | 5 (20 per participant) |
| Study 2 pairs per participant (individual) | 10 |
| Study 2 pairs per group (group phase) | 30 |

Effect sizes are specified on the log-odds scale and correspond to Cohen's
*d* = 0.3 for most main interactions and *d* = 0.5 for the H3 contrast. See
the docstrings in the two scripts for the full mapping.

## Requirements

```
python >= 3.10        # required by recent versions of pyfixest
numpy
pandas
pyfixest
scipy
matplotlib
tqdm
statsmodels           # Study 1 only (Holm-Bonferroni correction for H1 and H2)
```

Recommended setup with conda (creates an isolated environment named
`power_analysis` and installs everything needed):

```bash
conda create -n power_analysis python=3.11 -y
conda activate power_analysis
pip install numpy pandas pyfixest scipy statsmodels matplotlib tqdm
```

Or, with an existing Python 3.10+ installation:

```bash
pip install numpy pandas pyfixest scipy statsmodels matplotlib tqdm
```

## Usage

Each simulation script accepts a `--mode` flag that selects which hypotheses
to test and which sample-size range to scan. Modes exist because H4, H5, H10,
and H11 require larger samples than the other hypotheses; running everything
in one pass would be inefficient.

### Study 1

```bash
# H1, H2, H3 (sample range 50 to 500)
python Study1_PowerAnalysis.py --mode h1h2h3

# H4, H5 (sample range 800 to 1500)
python Study1_PowerAnalysis.py --mode h4h5

# All hypotheses
python Study1_PowerAnalysis.py --mode all
```

### Study 2

```bash
# H6, H7, H8, H9 (sample range 60 to 180, in multiples of 12)
python Study2_PowerAnalysis.py --mode h6789

# H10, H11 (sample range 180 to 360, in multiples of 12)
python Study2_PowerAnalysis.py --mode h1011

# All hypotheses
python Study2_PowerAnalysis.py --mode all
```

For a quick smoke test, pass a smaller `--n_sim` value, for example
`--n_sim 50`. Run any script with `--help` to see all options, including
overrides for the sample-size grid.

### Running everything in parallel (Windows)

`run_all_parallel.bat` launches the four modes simultaneously, each in its own
command-prompt window. Place the batch file in the same folder as the two
simulation scripts and double-click it. Total wall-clock time is roughly the
longest single mode, on the order of a few hours on a typical workstation
depending on CPU. Each window stays open when its mode finishes so that
progress messages remain visible.

The batch file activates a conda environment named `power_analysis` before
running each script. If you used a different environment name, edit the
`CONDA_ENV` variable near the top of the batch file.

### Re-plotting power curves

`replot.py` reads an existing results folder and writes a new figure with a
2-column layout that adapts to the number of tested hypotheses. The original
figure files are preserved; the re-plotted versions are written as
`power_curves_replot.png` and `power_curves_replot.pdf` inside the same
folder.

```bash
# Single folder
python replot.py results_study1_h1h2h3_20260507_140000/

# Multiple folders via shell glob
python replot.py results_*/
```

Layouts:

| Number of hypotheses | Layout |
| --- | --- |
| 2 (Study 1 h4h5; Study 2 h1011) | 1 × 2 |
| 3 (Study 1 h1h2h3) | 2 × 2 (last cell empty) |
| 4 (Study 2 h6789) | 2 × 2 |
| 5 (Study 1 all) | 3 × 2 (last cell empty) |
| 6 (Study 2 all) | 3 × 2 |

## Output

Each simulation run writes its outputs to a timestamped folder named
`results_studyN_<mode>_<YYYYMMDD_HHMMSS>/` in the current directory. Each
folder contains:

| File | Description |
| --- | --- |
| `power_results.csv` | Estimated power per hypothesis at each sample size. |
| `metadata.json` | Run configuration: mode, hypotheses tested, simulations, alpha, effect sizes. |
| `summary_report.txt` | Human-readable summary including the smallest sample size at which each hypothesis reaches 80% and 95% power. |
| `power_curves.png` | Power-curve figure (PNG). |
| `power_curves.pdf` | Power-curve figure (PDF). |

`replot.py` adds two additional files:

| File | Description |
| --- | --- |
| `power_curves_replot.png` | Re-plotted figure with 2-column layout. |
| `power_curves_replot.pdf` | Same in PDF. |

Because folder names are mode- and timestamp-specific, parallel runs do not
overwrite each other.

## Repository layout

```
.
├── README.md
├── Study1_PowerAnalysis.py
├── Study2_PowerAnalysis.py
├── replot.py
└── run_all_parallel.bat
```
