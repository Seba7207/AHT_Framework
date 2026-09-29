#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AHT Decision-Support Framework

Reproducible implementation accompanying:

"A Knowledge-Based Decision-Support Framework for Structured Assessment
of Suspected Abusive Head Trauma"

Authors:
    Sebastian Glowinski
    Alina Glowinska

Version:
    2.0

Python:
    >=3.9

Notes
-----
Figure 1 in the manuscript is the conceptual flowchart and is not generated
programmatically by this script.

This script generates Figures 2-6 and the numerical data used for the
scenario, contribution, contour, clinical-case, and coefficient-sensitivity
analyses. The framework output is a diagnostic support score and is NOT a
statistical estimate of the probability of AHT.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Directories
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Framework parameters
# =============================================================================

BETA0 = 0.50

DOMAIN_WEIGHTS = {
    "C": 0.15,
    "I": 0.25,
    "R": 0.30,
    "H": 0.10,
    "D": 0.10,
    "U": 0.10,
}

CLINICAL_WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
IMAGING_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.20])
RETINAL_WEIGHTS = np.array([0.25, 0.25, 0.30, 0.20])
HISTORY_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15])
DIFFERENTIAL_WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15])
UNCERTAINTY_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.20])

REFERENCE_SCENARIO = {
    "C": 0.70,
    "I": 0.70,
    "R": 0.70,
    "H": 0.50,
    "D": 0.30,
    "U": 0.30,
}

MINIMUM_SCENARIO = {
    "C": 0.00,
    "I": 0.00,
    "R": 0.00,
    "H": 0.00,
    "D": 1.00,
    "U": 1.00,
}

MAXIMUM_SCENARIO = {
    "C": 1.00,
    "I": 1.00,
    "R": 1.00,
    "H": 1.00,
    "D": 0.00,
    "U": 0.00,
}

SCENARIO_2 = {
    "C": 0.90,
    "I": 1.00,
    "R": 0.90,
    "H": 0.80,
    "D": 0.10,
    "U": 0.10,
}

SCENARIO_3 = {
    "C": 0.70,
    "I": 0.50,
    "R": 0.40,
    "H": 0.50,
    "D": 0.70,
    "U": 0.60,
}

CLINICAL_CASE_SEED = 20260811
CLINICAL_CASE_N = 10000


# =============================================================================
# Core functions
# =============================================================================

def logistic(score):
    """Apply the logistic transformation to the linear predictor."""
    return 1.0 / (1.0 + np.exp(-np.asarray(score)))


def weighted_score(values, weights):
    """Calculate a weighted average within one diagnostic domain."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("Values and weights must have the same length.")
    return float(np.sum(values * weights))


def compute_C(values):
    return weighted_score(values, CLINICAL_WEIGHTS)


def compute_I(values):
    return weighted_score(values, IMAGING_WEIGHTS)


def compute_R(values):
    return weighted_score(values, RETINAL_WEIGHTS)


def compute_H(values):
    return weighted_score(values, HISTORY_WEIGHTS)


def compute_D(values):
    return weighted_score(values, DIFFERENTIAL_WEIGHTS)


def compute_U(values):
    return weighted_score(values, UNCERTAINTY_WEIGHTS)


def compute_support_score(C, I, R, H, D, U, beta0=BETA0,
                          domain_weights=DOMAIN_WEIGHTS):
    """
    Compute the bounded diagnostic support score.

    The returned value is a computational framework output. It must not be
    interpreted as a clinical probability of AHT.
    """
    linear_score = (
        beta0
        + domain_weights["C"] * C
        + domain_weights["I"] * I
        + domain_weights["R"] * R
        + domain_weights["H"] * H
        - domain_weights["D"] * D
        - domain_weights["U"] * U
    )
    return float(logistic(linear_score)), float(linear_score)


def scenario_to_array(scenario):
    return np.array(
        [scenario["C"], scenario["I"], scenario["R"],
         scenario["H"], scenario["D"], scenario["U"]],
        dtype=float,
    )


def calculate_scenario(scenario, beta0=BETA0, domain_weights=DOMAIN_WEIGHTS):
    values = scenario_to_array(scenario)
    return compute_support_score(*values, beta0=beta0,
                                 domain_weights=domain_weights)


# =============================================================================
# Figure 2: baseline parameter sensitivity
# =============================================================================

def beta_sensitivity():
    """Generate data for Figure 2."""
    beta_values = np.arange(-0.50, 0.501, 0.05)
    scores = []

    for beta in beta_values:
        score, _ = calculate_scenario(
            REFERENCE_SCENARIO,
            beta0=float(beta),
        )
        scores.append(score)

    df = pd.DataFrame({
        "beta0": beta_values,
        "Diagnostic_support_score": scores,
        "Diagnostic_support_score_percent": np.asarray(scores) * 100,
    })

    df.to_csv(DATA_DIR / "Figure2_beta_sensitivity.csv", index=False)
    return df


def plot_beta_sensitivity(df):
    plt.figure(figsize=(8, 6))
    plt.plot(
        df["beta0"],
        df["Diagnostic_support_score_percent"],
        "-o",
        linewidth=2,
        markersize=5,
    )
    plt.xlabel(r"$\beta_0$")
    plt.ylabel("Diagnostic support score (%)")
    plt.title("Effect of the baseline parameter ($\\beta_0$)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure2_beta_sensitivity.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# Figure 3: minimum/reference/maximum scenarios
# =============================================================================

def scenario_analysis():
    """Generate data for Figure 3."""
    beta_values = np.arange(-0.50, 0.501, 0.05)

    rows = []
    for beta in beta_values:
        row = {"beta0": float(beta)}
        for name, scenario in (
            ("Minimum", MINIMUM_SCENARIO),
            ("Reference", REFERENCE_SCENARIO),
            ("Maximum", MAXIMUM_SCENARIO),
        ):
            value, _ = calculate_scenario(scenario, beta0=float(beta))
            row[name] = value
            row[f"{name}_percent"] = value * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "Figure3_scenarios.csv", index=False)
    return df


def plot_scenarios(df):
    plt.figure(figsize=(8, 6))
    plt.plot(
        df["beta0"], df["Minimum_percent"],
        linewidth=2, label="Minimum scenario"
    )
    plt.plot(
        df["beta0"], df["Reference_percent"],
        linewidth=2, label="Reference scenario"
    )
    plt.plot(
        df["beta0"], df["Maximum_percent"],
        linewidth=2, label="Maximum scenario"
    )
    plt.xlabel(r"$\beta_0$")
    plt.ylabel("Diagnostic support score (%)")
    plt.title("Diagnostic support score across representative scenarios")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_scenarios.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# Figure 4: domain contribution analysis
# =============================================================================

def domain_contribution_analysis():
    """
    Calculate relative contributions under the maximum scenario.

    D and U are zero in the maximum scenario and are therefore excluded.
    Contributions are normalized to sum to 100%.
    """
    active = {
        "Retinal (R)": DOMAIN_WEIGHTS["R"] * MAXIMUM_SCENARIO["R"],
        "Imaging (I)": DOMAIN_WEIGHTS["I"] * MAXIMUM_SCENARIO["I"],
        "Clinical (C)": DOMAIN_WEIGHTS["C"] * MAXIMUM_SCENARIO["C"],
        "History (H)": DOMAIN_WEIGHTS["H"] * MAXIMUM_SCENARIO["H"],
    }

    labels = list(active.keys())
    raw = np.asarray(list(active.values()), dtype=float)
    contribution = raw / raw.sum() * 100.0
    cumulative = np.cumsum(contribution)

    df = pd.DataFrame({
        "Component": labels,
        "Contribution_percent": contribution,
        "Cumulative_percent": cumulative,
    })

    df.to_csv(DATA_DIR / "Figure4_domain_contributions.csv", index=False)
    return df


def plot_domain_contributions(df):
    fig, ax1 = plt.subplots(figsize=(9, 6))

    x = np.arange(len(df))
    ax1.bar(
        x,
        df["Contribution_percent"],
        edgecolor="black",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Component"], rotation=30, ha="right")
    ax1.set_ylabel("Relative contribution (%)")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["Cumulative_percent"],
        "-o",
        linewidth=2,
    )
    ax2.set_ylabel("Cumulative contribution (%)")
    ax2.set_ylim(0, 110)

    plt.title("Relative contribution of active diagnostic domains")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure4_domain_contributions.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Figure 5: joint influence of R and I
# =============================================================================

def contour_analysis():
    """
    Generate the Figure 5 heatmap/contour matrix.

    I and R range from 0.2 to 1.0. C, H, D and U remain fixed at the
    reference-scenario values.
    """
    i_values = np.linspace(0.20, 1.00, 81)
    r_values = np.linspace(0.20, 1.00, 81)

    z = np.zeros((len(i_values), len(r_values)))

    for i, imaging in enumerate(i_values):
        for j, retinal in enumerate(r_values):
            score, _ = compute_support_score(
                REFERENCE_SCENARIO["C"],
                imaging,
                retinal,
                REFERENCE_SCENARIO["H"],
                REFERENCE_SCENARIO["D"],
                REFERENCE_SCENARIO["U"],
            )
            z[i, j] = score * 100.0

    matrix = pd.DataFrame(
        z,
        index=np.round(i_values, 4),
        columns=np.round(r_values, 4),
    )
    matrix.index.name = "I"
    matrix.columns.name = "R"
    matrix.to_csv(DATA_DIR / "Figure5_contour_matrix.csv")

    return r_values, i_values, z


def plot_contour(r_values, i_values, z):
    fig, ax = plt.subplots(figsize=(8, 7))

    contour = ax.contourf(
        r_values,
        i_values,
        z,
        levels=25,
    )

    ax.contour(
        r_values,
        i_values,
        z,
        levels=10,
        colors="black",
        linewidths=0.6,
        linestyles="dashed",
    )

    ax.set_xlabel("Retinal findings (R)")
    ax.set_ylabel("Information domain (I)")

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Diagnostic support score (%)")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure5_contour.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Figure 6: illustrative clinical case
# =============================================================================

def clinical_case_simulation(n=CLINICAL_CASE_N, seed=CLINICAL_CASE_SEED):
    """
    Simulate the illustrative clinical case using the case-specific ranges
    reported in the manuscript.
    """
    rng = np.random.default_rng(seed)

    C = np.full(n, 0.80)
    I = rng.uniform(0.80, 1.00, n)
    R = rng.uniform(0.70, 0.90, n)
    H = rng.uniform(0.40, 0.60, n)
    D = rng.uniform(0.30, 0.50, n)
    U = rng.uniform(0.40, 0.60, n)

    scores = logistic(
        BETA0
        + DOMAIN_WEIGHTS["C"] * C
        + DOMAIN_WEIGHTS["I"] * I
        + DOMAIN_WEIGHTS["R"] * R
        + DOMAIN_WEIGHTS["H"] * H
        - DOMAIN_WEIGHTS["D"] * D
        - DOMAIN_WEIGHTS["U"] * U
    )

    df = pd.DataFrame({
        "C": C,
        "I": I,
        "R": R,
        "H": H,
        "D": D,
        "U": U,
        "Diagnostic_support_score": scores,
    })

    df.to_csv(DATA_DIR / "Figure6_clinical_case_simulation.csv", index=False)

    summary = pd.DataFrame([{
        "n": n,
        "seed": seed,
        "mean": float(np.mean(scores)),
        "sd": float(np.std(scores, ddof=1)),
        "2.5th_percentile": float(np.percentile(scores, 2.5)),
        "97.5th_percentile": float(np.percentile(scores, 97.5)),
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
    }])

    summary.to_csv(
        DATA_DIR / "Figure6_clinical_case_summary.csv",
        index=False,
    )

    return scores, summary


def plot_clinical_case(scores, summary):
    mean_score = float(summary.loc[0, "mean"])

    plt.figure(figsize=(8, 6))
    plt.hist(
        scores,
        bins=30,
        edgecolor="black",
    )
    plt.axvline(
        mean_score,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_score:.3f}",
    )

    plt.xlabel("Diagnostic support score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure6_clinical_case.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# Additional illustrative scenarios
# =============================================================================

def additional_scenarios():
    rows = []
    for name, scenario in (
        ("Scenario 2", SCENARIO_2),
        ("Scenario 3", SCENARIO_3),
    ):
        score, linear_score = calculate_scenario(scenario)
        rows.append({
            "Scenario": name,
            "C": scenario["C"],
            "I": scenario["I"],
            "R": scenario["R"],
            "H": scenario["H"],
            "D": scenario["D"],
            "U": scenario["U"],
            "beta0": BETA0,
            "linear_score": linear_score,
            "diagnostic_support_score": score,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "additional_scenarios.csv", index=False)
    return df


# =============================================================================
# Sensitivity and uncertainty analysis
# =============================================================================

def perturb_weights(weights, rng):
    """Perturb weights independently by ±50% and renormalize."""
    perturbed = np.asarray(weights, dtype=float) * rng.uniform(
        0.50, 1.50, size=len(weights)
    )
    return perturbed / perturbed.sum()


def compute_within_domain_scores(values, within_weight_sets):
    """Compute C, I, R, H, D and U from perturbed within-domain weights."""
    return np.array([
        weighted_score(values["C"], within_weight_sets["C"]),
        weighted_score(values["I"], within_weight_sets["I"]),
        weighted_score(values["R"], within_weight_sets["R"]),
        weighted_score(values["H"], within_weight_sets["H"]),
        weighted_score(values["D"], within_weight_sets["D"]),
        weighted_score(values["U"], within_weight_sets["U"]),
    ])


def sensitivity_analysis(n=10000, seed=20260811):
    """
    Perform the ±50% coefficient perturbation analysis described in Section 2.11.

    The manuscript does not specify a random seed for this analysis. A fixed
    seed is used here to make the public implementation reproducible; the
    reported manuscript summary values should therefore be interpreted as
    rounded reference results rather than as a claim that this seed was used
    in the manuscript calculations.
    """
    rng = np.random.default_rng(seed)

    base_domain = np.array([
        DOMAIN_WEIGHTS["C"],
        DOMAIN_WEIGHTS["I"],
        DOMAIN_WEIGHTS["R"],
        DOMAIN_WEIGHTS["H"],
        DOMAIN_WEIGHTS["D"],
        DOMAIN_WEIGHTS["U"],
    ])

    scenario_arrays = {
        "Minimum": scenario_to_array(MINIMUM_SCENARIO),
        "Reference": scenario_to_array(REFERENCE_SCENARIO),
        "Maximum": scenario_to_array(MAXIMUM_SCENARIO),
    }

    domain_results = {name: [] for name in scenario_arrays}

    within_base = {
        "C": CLINICAL_WEIGHTS,
        "I": IMAGING_WEIGHTS,
        "R": RETINAL_WEIGHTS,
        "H": HISTORY_WEIGHTS,
        "D": DIFFERENTIAL_WEIGHTS,
        "U": UNCERTAINTY_WEIGHTS,
    }

    reference_components = {
        "C": np.array([0.70, 0.70, 0.70, 0.70]),
        "I": np.array([0.70, 0.70, 0.70, 0.70]),
        "R": np.array([0.70, 0.70, 0.70, 0.70]),
        "H": np.array([0.50, 0.50, 0.50, 0.50]),
        "D": np.array([0.30, 0.30, 0.30, 0.30]),
        "U": np.array([0.30, 0.30, 0.30, 0.30]),
    }

    within_results = []
    joint_results = []

    for _ in range(n):
        domain_w = perturb_weights(base_domain, rng)
        domain_dict = dict(zip(
            ["C", "I", "R", "H", "D", "U"],
            domain_w,
        ))

        for name, values in scenario_arrays.items():
            score = compute_support_score(
                *values,
                beta0=BETA0,
                domain_weights=domain_dict,
            )[0]
            domain_results[name].append(score)

        within_w = {
            key: perturb_weights(value, rng)
            for key, value in within_base.items()
        }

        within_scores = compute_within_domain_scores(
            reference_components,
            within_w,
        )

        within_score = compute_support_score(
            *within_scores,
            beta0=BETA0,
        )[0]
        within_results.append(within_score)

        joint_score = compute_support_score(
            *within_scores,
            beta0=BETA0,
            domain_weights=domain_dict,
        )[0]
        joint_results.append(joint_score)

    def summarize(values):
        values = np.asarray(values)
        return {
            "Mean": np.mean(values),
            "Median": np.median(values),
            "2.5th_percentile": np.percentile(values, 2.5),
            "97.5th_percentile": np.percentile(values, 97.5),
            "Minimum": np.min(values),
            "Maximum": np.max(values),
        }

    table3 = pd.DataFrame({
        scenario: summarize(values)
        for scenario, values in domain_results.items()
    }).T
    table3.index.name = "Scenario"
    table3.to_csv(DATA_DIR / "Table3_domain_level_sensitivity.csv")

    table4 = pd.DataFrame({
        "Within-domain coefficients": summarize(within_results),
        "Joint domain-level and within-domain coefficients": summarize(joint_results),
    }).T
    table4.index.name = "Analysis"
    table4.to_csv(DATA_DIR / "Table4_within_domain_joint_sensitivity.csv")

    return table3, table4


# =============================================================================
# Verification
# =============================================================================

def verify_key_results():
    """Print key numerical checks against the manuscript specification."""
    minimum_score, _ = calculate_scenario(MINIMUM_SCENARIO)
    reference_score, _ = calculate_scenario(REFERENCE_SCENARIO)
    maximum_score, _ = calculate_scenario(MAXIMUM_SCENARIO)
    scenario2_score, _ = calculate_scenario(SCENARIO_2)
    scenario3_score, _ = calculate_scenario(SCENARIO_3)

    print("\nKey numerical checks")
    print("--------------------")
    print(f"Minimum scenario:   {minimum_score:.6f}")
    print(f"Reference scenario: {reference_score:.6f}")
    print(f"Maximum scenario:   {maximum_score:.6f}")
    print(f"Scenario 2:         {scenario2_score:.6f}")
    print(f"Scenario 3:         {scenario3_score:.6f}")

    scores, summary = clinical_case_simulation()
    print("\nClinical case simulation")
    print("------------------------")
    print(f"Mean:               {summary.loc[0, 'mean']:.6f}")
    print(f"SD:                 {summary.loc[0, 'sd']:.6f}")
    print(f"2.5th percentile:   {summary.loc[0, '2.5th_percentile']:.6f}")
    print(f"97.5th percentile:  {summary.loc[0, '97.5th_percentile']:.6f}")

    expected = {
        "minimum": 0.574442,
        "reference": 0.727108,
        "maximum": 0.785835,
        "scenario2": 0.771182,
        "scenario3": 0.683521,
        "clinical_mean": 0.739717,
    }

    actual = {
        "minimum": minimum_score,
        "reference": reference_score,
        "maximum": maximum_score,
        "scenario2": scenario2_score,
        "scenario3": scenario3_score,
        "clinical_mean": float(summary.loc[0, "mean"]),
    }

    for key, expected_value in expected.items():
        if not np.isclose(actual[key], expected_value, atol=1e-5):
            raise AssertionError(
                f"Verification failed for {key}: "
                f"{actual[key]:.8f} != {expected_value:.8f}"
            )

    print("\nVerification: PASS")


# =============================================================================
# Main program
# =============================================================================

def main():
    print("\n==============================================")
    print(" AHT Decision-Support Framework")
    print(" Reproducible implementation v2.0")
    print("==============================================")

    verify_key_results()

    print("\nGenerating Figure 2...")
    df2 = beta_sensitivity()
    plot_beta_sensitivity(df2)

    print("Generating Figure 3...")
    df3 = scenario_analysis()
    plot_scenarios(df3)

    print("Generating Figure 4...")
    df4 = domain_contribution_analysis()
    plot_domain_contributions(df4)

    print("Generating Figure 5...")
    r_values, i_values, z = contour_analysis()
    plot_contour(r_values, i_values, z)

    print("Generating Figure 6...")
    scores, summary = clinical_case_simulation()
    plot_clinical_case(scores, summary)

    print("Generating additional illustrative scenarios...")
    additional_scenarios()

    print("Running coefficient sensitivity analysis...")
    sensitivity_analysis()

    print("\n==============================================")
    print("Analysis completed successfully.")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Data:    {DATA_DIR}")
    print("==============================================")


if __name__ == "__main__":
    main()
