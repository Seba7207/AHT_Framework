#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AHT Decision-Support Framework

Implementation accompanying the manuscript:

"A Conceptual Decision-Support Framework for Structured Assessment
of Suspected Abusive Head Trauma"

Authors:
    Sebastian Glowinski
    Alina Glowinska

Version:
    1.0

Python:
    >=3.9
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

BETA0 = -0.50

DOMAIN_WEIGHTS = {
    "C": 0.15,
    "I": 0.25,
    "R": 0.30,
    "H": 0.10,
    "D": 0.15,
    "U": 0.05,
}


# =============================================================================
# Within-domain contribution coefficients
# =============================================================================

CLINICAL_WEIGHTS = np.array(
    [0.40, 0.20, 0.20, 0.20]
)

IMAGING_WEIGHTS = np.array(
    [0.30, 0.30, 0.20, 0.20]
)

RETINAL_WEIGHTS = np.array(
    [0.25, 0.25, 0.30, 0.20]
)

HISTORY_WEIGHTS = np.array(
    [0.40, 0.25, 0.20, 0.15]
)

DIFFERENTIAL_WEIGHTS = np.array(
    [0.35, 0.30, 0.20, 0.15]
)

UNCERTAINTY_WEIGHTS = np.array(
    [0.30, 0.30, 0.20, 0.20]
)


# =============================================================================
# Utility functions
# =============================================================================

def logistic(score):
    """
    Logistic transformation.

    Parameters
    ----------
    score : float
        Linear score.

    Returns
    -------
    float
        Probability.
    """

    return 1.0 / (1.0 + np.exp(-score))


def weighted_score(values, weights):
    """
    Calculates weighted contribution of one diagnostic domain.
    """

    values = np.asarray(values, dtype=float)

    return np.sum(values * weights)


# =============================================================================
# Diagnostic domains
# =============================================================================

def compute_C(values):
    """Clinical domain."""
    return weighted_score(values, CLINICAL_WEIGHTS)


def compute_I(values):
    """Imaging domain."""
    return weighted_score(values, IMAGING_WEIGHTS)


def compute_R(values):
    """Retinal domain."""
    return weighted_score(values, RETINAL_WEIGHTS)


def compute_H(values):
    """History domain."""
    return weighted_score(values, HISTORY_WEIGHTS)


def compute_D(values):
    """Differential diagnosis domain."""
    return weighted_score(values, DIFFERENTIAL_WEIGHTS)


def compute_U(values):
    """Diagnostic uncertainty domain."""
    return weighted_score(values, UNCERTAINTY_WEIGHTS)


# =============================================================================
# Probability model
# =============================================================================

def compute_probability(C, I, R, H, D, U, beta0=BETA0):
    """
    Computes AHT probability.
    """

    score = (
        beta0
        + DOMAIN_WEIGHTS["C"] * C
        + DOMAIN_WEIGHTS["I"] * I
        + DOMAIN_WEIGHTS["R"] * R
        + DOMAIN_WEIGHTS["H"] * H
        - DOMAIN_WEIGHTS["D"] * D
        - DOMAIN_WEIGHTS["U"] * U
    )

    probability = logistic(score)

    return probability, score


# =============================================================================
# Interpretation
# =============================================================================

def classify_probability(probability):
    """
    Returns qualitative interpretation.
    """

    if probability < 0.20:
        return "Very low"

    if probability < 0.40:
        return "Low"

    if probability < 0.60:
        return "Indeterminate"

    if probability < 0.80:
        return "High"

    return "Very high"


# =============================================================================
# Example calculation
# =============================================================================

def run_example():
    """
    Demonstration of framework operation.
    """

    C = compute_C([0.8, 0.7, 0.8, 0.6])
    I = compute_I([0.9, 0.8, 0.7, 0.8])
    R = compute_R([0.9, 0.8, 0.9, 0.8])
    H = compute_H([0.6, 0.5, 0.5, 0.4])
    D = compute_D([0.4, 0.3, 0.2, 0.3])
    U = compute_U([0.5, 0.4, 0.4, 0.5])

    probability, score = compute_probability(
        C, I, R, H, D, U
    )

    print("\n===== AHT Framework =====")

    print(f"C = {C:.3f}")
    print(f"I = {I:.3f}")
    print(f"R = {R:.3f}")
    print(f"H = {H:.3f}")
    print(f"D = {D:.3f}")
    print(f"U = {U:.3f}")

    print(f"\nScore = {score:.3f}")
    print(f"P(AHT) = {probability:.3f}")
    print(
        f"Interpretation = "
        f"{classify_probability(probability)}"
    )


# =============================================================================
# Figure 1
# =============================================================================

def beta_sensitivity():
    """
    Sensitivity analysis of the baseline parameter β0.
    """

    beta_values = np.arange(-0.50, 0.51, 0.05)

    probabilities = []

    C = 0.70
    I = 0.70
    R = 0.70
    H = 0.50
    D = 0.30
    U = 0.30

    for beta in beta_values:

        probability, _ = compute_probability(
            C,
            I,
            R,
            H,
            D,
            U,
            beta,
        )

        probabilities.append(probability * 100)

    df = pd.DataFrame(
        {
            "beta0": beta_values,
            "Probability": probabilities,
        }
    )

    df.to_csv(
        DATA_DIR / "beta_sensitivity.csv",
        index=False,
    )

    return df



# =============================================================================
# Figure 1
# =============================================================================

def plot_beta_sensitivity(df):
    """
    Generates Figure 1 showing the influence of β0.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["beta0"],
        df["Probability"],
        "-o",
        linewidth=3,
        markersize=6,
    )

    plt.axhspan(0, 20, color="#c6dbef", alpha=0.35)
    plt.axhspan(20, 40, color="#9ecae1", alpha=0.35)
    plt.axhspan(40, 60, color="#fee391", alpha=0.35)
    plt.axhspan(60, 80, color="#fdae6b", alpha=0.35)
    plt.axhspan(80, 100, color="#fb6a4a", alpha=0.35)

    plt.xlabel(r"$\beta_0$", fontsize=12)

    plt.ylabel(
        "Diagnostic support score (%)",
        fontsize=12,
    )

    plt.title(
        "Influence of baseline parameter",
        fontsize=13,
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "Figure1_beta_sensitivity.png",
        dpi=600,
    )

    plt.close()


# =============================================================================
# Scenario analysis
# =============================================================================
    
    
    
    
    
# =============================================================================
# Scenario analysis
# =============================================================================

def scenario_analysis():
    """
    Computes three representative diagnostic scenarios.
    """

    beta_values = np.arange(-0.50, 0.51, 0.05)

    scenarios = {
        "Minimum": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        "Reference": (0.7, 0.7, 0.7, 0.5, 0.3, 0.3),
        "Maximum": (1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    }

    results = {}

    for name, values in scenarios.items():

        curve = []

        for beta in beta_values:

            probability, _ = compute_probability(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                beta,
            )

            curve.append(probability * 100)

        results[name] = curve

    df = pd.DataFrame(
        {
            "beta0": beta_values,
            "Minimum": results["Minimum"],
            "Reference": results["Reference"],
            "Maximum": results["Maximum"],
        }
    )

    df.to_csv(
        DATA_DIR / "beta_scenarios.csv",
        index=False,
    )

    return df


# =============================================================================
# Figure 2
# =============================================================================

def plot_scenarios(df):
    """
    Generates Figure 2.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["beta0"],
        df["Minimum"],
        linewidth=3,
        label="Minimum scenario",
    )

    plt.plot(
        df["beta0"],
        df["Reference"],
        linewidth=3,
        label="Reference scenario",
    )

    plt.plot(
        df["beta0"],
        df["Maximum"],
        linewidth=3,
        label="Maximum scenario",
    )

    plt.axhspan(0, 20, color="#c6dbef", alpha=0.35)
    plt.axhspan(20, 40, color="#9ecae1", alpha=0.35)
    plt.axhspan(40, 60, color="#fee391", alpha=0.35)
    plt.axhspan(60, 80, color="#fdae6b", alpha=0.35)
    plt.axhspan(80, 100, color="#fb6a4a", alpha=0.35)

    plt.xlabel(r"$\beta_0$")
    plt.ylabel("Diagnostic support score (%)")
    plt.title("Scenario analysis")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "Figure2_scenarios.png",
        dpi=600,
    )

    plt.close()


# =============================================================================
# Pareto analysis
# =============================================================================

def pareto_analysis(C, I, R, H, D, U):
    """
    Computes relative contribution of each diagnostic domain.
    """

    labels = [
        "Retinal (R)",
        "Imaging (I)",
        "Clinical (C)",
        "History (H)",
        "Differential diagnosis (D)",
        "Uncertainty (U)",
    ]

    values = np.array([
        DOMAIN_WEIGHTS["R"] * R,
        DOMAIN_WEIGHTS["I"] * I,
        DOMAIN_WEIGHTS["C"] * C,
        DOMAIN_WEIGHTS["H"] * H,
        DOMAIN_WEIGHTS["D"] * D,
        DOMAIN_WEIGHTS["U"] * U,
    ])

    contribution = values / np.sum(values) * 100
    cumulative = np.cumsum(contribution)

    df = pd.DataFrame({
        "Component": labels,
        "Contribution": contribution,
        "Cumulative": cumulative,
    })

    df.to_csv(
        DATA_DIR / "pareto_data.csv",
        index=False,
    )

    return df


# =============================================================================
# Figure 3
# =============================================================================

def plot_pareto(df):
    """
    Generates Pareto chart of domain contributions.
    """

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.bar(
        df["Component"],
        df["Contribution"],
        color="lightgray",
        edgecolor="black",
    )

    ax1.set_ylabel("Contribution (%)")

    plt.xticks(rotation=40, ha="right")

    ax2 = ax1.twinx()

    ax2.plot(
        df["Cumulative"],
        "-o",
        color="red",
        linewidth=2,
    )

    ax2.set_ylabel(
        "Cumulative (%)",
        color="red",
    )

    ax2.tick_params(
        axis="y",
        colors="red",
    )

    plt.title("Pareto analysis")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "Figure3_pareto.png",
        dpi=600,
    )

    plt.close(fig)
    









    

# =============================================================================
# Figure 4
# =============================================================================
# =============================================================================
# Contour analysis
# =============================================================================

def contour_analysis():
    """
    Generates contour plot matrix.
    """

    r_values = np.linspace(10, 100, 91)
    i_values = np.linspace(10, 100, 91)

    z = np.zeros((len(i_values), len(r_values)))

    C = 0.70
    H = 0.50
    D = 0.30
    U = 0.30

    for i, imaging in enumerate(i_values):

        for j, retinal in enumerate(r_values):

            probability, _ = compute_probability(
                C,
                imaging / 100,
                retinal / 100,
                H,
                D,
                U,
            )

            z[i, j] = probability * 100

    pd.DataFrame(z).to_csv(
        DATA_DIR / "heatmap_matrix.csv",
        index=False,
    )

    return r_values, i_values, z



def plot_contour(r_values, i_values, z):
    """
    Generates contour plot of the diagnostic support score.
    """

    fig = plt.figure(figsize=(8, 7))

    contour = plt.contourf(
        r_values,
        i_values,
        z,
        levels=25,
        cmap="jet",
    )

    plt.contour(
        r_values,
        i_values,
        z,
        colors="black",
        linewidths=1,
        linestyles="dashed",
    )

    plt.xlabel("Retinal score (R %)")

    plt.ylabel("Imaging score (I %)")

    colorbar = plt.colorbar(contour)

    colorbar.set_label(
        "Diagnostic support score (%)"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "Figure4_contour.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)






# =============================================================================
# Figure 5
# =============================================================================
# =============================================================================
# Monte Carlo simulation
# =============================================================================

def monte_carlo_simulation(n=10000):
    """
    Performs Monte Carlo simulation.
    """

    probabilities = np.zeros(n)

    for i in range(n):

        C = np.random.rand()
        I = np.random.rand()
        R = np.random.rand()
        H = np.random.rand()
        D = np.random.rand()
        U = np.random.rand()

        probability, _ = compute_probability(
            C,
            I,
            R,
            H,
            D,
            U,
        )

        probabilities[i] = probability * 100

    pd.DataFrame(
        {"Probability": probabilities}
    ).to_csv(
        DATA_DIR / "monte_carlo.csv",
        index=False,
    )

    return probabilities






def plot_monte_carlo(probabilities):
    """
    Generates Monte Carlo histogram.
    """

    fig = plt.figure(figsize=(8, 6))

    plt.hist(
        probabilities,
        bins=30,
        edgecolor="black",
    )

    plt.xlabel("Diagnostic support score (%)")

    plt.ylabel("Frequency")

    plt.title("Monte Carlo simulation")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "Figure5_monte_carlo.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Main program
# =============================================================================

def main():
    """
    Executes the complete AHT framework.
    """

    print("\n======================================")
    print(" AHT Decision-Support Framework")
    print("======================================")

    # -------------------------------------------------------------------------
    # Example
    # -------------------------------------------------------------------------

    run_example()

    # -------------------------------------------------------------------------
    # Figure 1
    # -------------------------------------------------------------------------

    print("\nGenerating Figure 1...")

    beta_df = beta_sensitivity()

    plot_beta_sensitivity(beta_df)

    # -------------------------------------------------------------------------
    # Figure 2
    # -------------------------------------------------------------------------

    print("Generating Figure 2...")

    scenario_df = scenario_analysis()

    plot_scenarios(scenario_df)

    # -------------------------------------------------------------------------
    # Example case
    # -------------------------------------------------------------------------

    C = compute_C([0.8, 0.7, 0.8, 0.6])
    I = compute_I([0.9, 0.8, 0.7, 0.8])
    R = compute_R([0.9, 0.8, 0.9, 0.8])
    H = compute_H([0.6, 0.5, 0.5, 0.4])
    D = compute_D([0.4, 0.3, 0.2, 0.3])
    U = compute_U([0.5, 0.4, 0.4, 0.5])

    # -------------------------------------------------------------------------
    # Figure 3
    # -------------------------------------------------------------------------

    print("Generating Figure 3...")

    pareto_df = pareto_analysis(
        C,
        I,
        R,
        H,
        D,
        U,
    )

    plot_pareto(pareto_df)

    # -------------------------------------------------------------------------
    # Figure 4
    # -------------------------------------------------------------------------

    print("Generating Figure 4...")

    r_values, i_values, z = contour_analysis()

    plot_contour(
        r_values,
        i_values,
        z,
    )

    # -------------------------------------------------------------------------
    # Figure 5
    # -------------------------------------------------------------------------

    print("Generating Figure 5...")

    mc = monte_carlo_simulation()

    plot_monte_carlo(mc)

    print("\n======================================")
    print("Analysis completed successfully.")
    print("======================================")

    print(f"\nFigures saved to : {FIGURES_DIR}")
    print(f"CSV files saved to: {DATA_DIR}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()