import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mritk


def main(c_target, SNR=25):
    # Physical Parameters
    r1 = 3.2  # Longitudinal relaxivity (s^-1 L mmol^-1)
    T10 = 4.5  # Native T1 time of CSF (seconds)
    # SNR = 25.0  # Signal-to-Noise Ratio
    sigma = 1.0 / SNR  # Theoretical max signals approach 1.0 (based on M0=1.0)
    M0 = 1.0
    # # Sequence Parameters
    TR = 9.6  # Taken from Gonzo paper
    TI = 2.65  # Taken from Gonzo paper
    t_LL = np.linspace(0.115, 2.754, 14)  # Look-Locker: 14 data points over 2.75s same as Gonzo

    N = 5000

    T1_target = mritk.concentration.T1_from_concentration_expr(c=c_target, t1_0=T10, r1=r1)
    np.random.seed(42)  # For reproducibility
    c_true_array = np.full(N, c_target)
    T1_true = mritk.concentration.T1_from_concentration_expr(c=c_true_array, t1_0=T10, r1=r1)

    # Generate Noisy T1 Estimates
    T1_est_LL = mritk.looklocker.T1_to_noisy_T1_looklocker(
        T1_true,
        t_LL=t_LL,
        M0=M0,
        sigma=sigma,
    )
    T1_est_LL /= 1000.0  # Convert ms to seconds for consistency
    T1_range = np.linspace(0.1, 10.0, 5000)
    S_SE_range, S_IR_range = mritk.mixed.T1_to_mixed_signals(T1_range, TR=TR, TI=TI)
    ratio_range = S_IR_range / S_SE_range
    T1_est_mixed = mritk.mixed.T1_to_noisy_T1_mixed(
        T1_true,
        TR=TR,
        TI=TI,
        f_grid=ratio_range,
        t_grid=T1_range,
        sigma=sigma,
    )

    T1_est_hybrid = mritk.hybrid.compute_hybrid_t1_array(
        ll_data=T1_est_LL,
        mixed_data=T1_est_mixed,
        mask=None,
        threshold=1.5,
    )

    T1_mixed = T1_est_mixed[~np.isnan(T1_est_mixed)]
    T1_LL = T1_est_LL[~np.isnan(T1_est_LL)]
    T1_hybrid = T1_est_hybrid[~np.isnan(T1_est_hybrid)]

    c_mixed = mritk.concentration.concentration_from_T1_expr(T1_mixed, t1_0=T10, r1=r1)
    c_LL = mritk.concentration.concentration_from_T1_expr(T1_LL, t1_0=T10, r1=r1)
    c_hybrid = mritk.concentration.concentration_from_T1_expr(T1_hybrid, t1_0=T10, r1=r1)

    # Create 3 subplots sharing the x-axis
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))  # , sharex="col")

    # 1. Mixed Sequence Subplot
    sns.histplot(T1_mixed, bins=60, stat="density", color="orange", alpha=0.5, ax=axes[0, 0], label="Mixed Distribution")
    sns.kdeplot(T1_mixed, color="darkorange", linestyle="-", ax=axes[0, 0])
    axes[0, 0].axvline(T1_target, color="red", linestyle="solid", linewidth=2, label=f"True T1 = {T1_target:.2f} s")
    axes[0, 0].set_title("Mixed Sequence T1 Estimation")
    axes[0, 0].legend()

    sns.histplot(c_mixed, bins=60, stat="density", color="orange", alpha=0.5, ax=axes[0, 1], label="Mixed Distribution")
    sns.kdeplot(c_mixed, color="darkorange", linestyle="-", ax=axes[0, 1])
    axes[0, 1].axvline(c_target, color="red", linestyle="solid", linewidth=2, label=f"True c = {c_target}")
    axes[0, 1].set_title("Mixed Sequence Concentration Estimation")
    axes[0, 1].legend()

    # 2. Look-Locker Sequence Subplot
    sns.histplot(T1_LL, bins=60, stat="density", color="blue", alpha=0.5, ax=axes[1, 0], label="Look-Locker Distribution")
    sns.kdeplot(T1_LL, color="darkblue", linestyle="-", ax=axes[1, 0])
    axes[1, 0].axvline(T1_target, color="red", linestyle="solid", linewidth=2, label=f"True T1 = {T1_target:.2f} s")
    axes[1, 0].set_title("Look-Locker Sequence T1 Estimation")
    axes[1, 0].legend()

    sns.histplot(c_LL, bins=60, stat="density", color="blue", alpha=0.5, ax=axes[1, 1], label="Look-Locker Distribution")
    sns.kdeplot(c_LL, color="darkblue", linestyle="-", ax=axes[1, 1])
    axes[1, 1].axvline(c_target, color="red", linestyle="solid", linewidth=2, label=f"True c = {c_target}")
    axes[1, 1].set_title("Look-Locker Sequence Concentration Estimation")
    axes[1, 1].legend()

    # 3. Hybrid Logic Subplot
    sns.histplot(T1_hybrid, bins=60, stat="density", color="purple", alpha=0.5, ax=axes[2, 0], label="Hybrid Distribution")
    sns.kdeplot(T1_hybrid, color="indigo", linestyle="-", ax=axes[2, 0])
    axes[2, 0].axvline(T1_target, color="red", linestyle="solid", linewidth=2, label=f"True T1 = {T1_target:.2f} s")
    axes[2, 0].set_title("Final Hybrid Pipeline T1 Estimation")
    axes[2, 0].set_xlabel("T1 Relaxation Time (seconds)")
    axes[2, 0].legend()

    sns.histplot(c_hybrid, bins=60, stat="density", color="purple", alpha=0.5, ax=axes[2, 1], label="Hybrid Distribution")
    sns.kdeplot(c_hybrid, color="indigo", linestyle="-", ax=axes[2, 1])
    axes[2, 1].axvline(c_target, color="red", linestyle="solid", linewidth=2, label=f"True c = {c_target}")
    axes[2, 1].set_title("Final Hybrid Pipeline Concentration Estimation")
    axes[2, 1].set_xlabel("Concentration (mmol/L)")
    axes[2, 1].legend()

    for ax in axes.flatten():
        ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(f"hybrid_pipeline_histograms_c{c_target}_{SNR}_{TR}_{TI}.png", dpi=300)


if __name__ == "__main__":
    # for c_target in [0.0, 0.05, 0.1]:
    #     for SNR in [7.0, 25.0]:
    #         print(f"Running main() with c_target={c_target}, SNR={SNR}")
    #         main(c_target=c_target, SNR=SNR)
    main(c_target=0.05, SNR=25)
