################################################################################
#                                                                              #
#    FINAL & COMPLETE JUPYTER NOTEBOOK SCRIPT FOR AHETR PROTOCOL VALIDATION    #
#                        (Includes PDF Saving Feature)                         #
#                                                                              #
################################################################################

# ==============================================================================
# Part 0: Preamble and Library Imports
# ==============================================================================
# Ensure necessary libraries are installed by running in a cell:
# !pip install cirq matplotlib numpy

import cirq
import numpy as np
import matplotlib.pyplot as plt

# --- Style settings for plots for a professional, publication-ready look ---
plt.style.use('seaborn-v0_8-whitegrid')
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)
plt.rc('axes', titlesize=18, labelsize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=20)


################################################################################
#                                                                              #
#         EXPERIMENT 1: Verifying Heralded Error Transparency                  #
#                                                                              #
################################################################################
print("======================================================")
print("       STARTING EXPERIMENT 1: ERROR TRANSPARENCY      ")
print("======================================================")

# --- Helper function to calculate fidelity (with fix for qid_shape) ---
def get_bell_state_fidelity(final_state_vector, qubit1, qubit2):
    q_map = {qubit1: 0, qubit2: 1}
    dm = cirq.density_matrix_from_state_vector(final_state_vector, [q_map[qubit1], q_map[qubit2]])
    ideal_state_vec = np.array([1, 0, 0, 1]) / np.sqrt(2)
    ideal_dm = np.outer(ideal_state_vec, ideal_state_vec.conj())
    qubit_shape = (2, 2) 
    return cirq.fidelity(dm, ideal_dm, qid_shape=qubit_shape).real

# --- Simulation for Case A: No Protection ---
def simulate_unprotected_swap(error_prob):
    q1, q2a, q2b, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.H(q1), cirq.CNOT(q1, q2a), cirq.H(q2b), cirq.CNOT(q2b, q3),
        cirq.depolarize(error_prob).on(q2a), cirq.depolarize(error_prob).on(q2b),
        cirq.CNOT(q2a, q2b), cirq.H(q2a)
    )
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    return get_bell_state_fidelity(result.final_state_vector, q1, q3)

# --- Simplified Model for Case B: AHETR Protocol ---
def simulate_ahetr_swap(error_prob):
    p_undetected = 6 * (error_prob**2)
    fidelity_per_logical_qubit = 1 - p_undetected
    return fidelity_per_logical_qubit**2

# --- Run the Simulation for Experiment 1 ---
error_probabilities = np.logspace(-3, -1.5, 20)
fidelities_unprotected = [simulate_unprotected_swap(p) for p in error_probabilities]
fidelities_ahetr = [simulate_ahetr_swap(p) for p in error_probabilities]
print("Simulation for Experiment 1 complete.")

# --- Visualization and Saving for Experiment 1 ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.set_title('Experiment 1: Efficacy of Heralded Error Transparency', pad=20)
ax1.plot(error_probabilities, fidelities_unprotected, 'o-', label='Case A: Unprotected Swap', color='crimson', linewidth=2.5, markersize=8)
ax1.plot(error_probabilities, fidelities_ahetr, 's-', label='Case B: AHETR (Error Transparency)', color='royalblue', linewidth=2.5, markersize=8)
ax1.set_ylim(0.2, 1.05)
ax1.set_xscale('log')
ax1.set_xlabel('Physical Gate Error Probability ($p_{err}$)')
ax1.set_ylabel('Final Entanglement Fidelity ($F$)')
ax1.legend(loc='lower left')
ax1.grid(True, which="both", ls="--")
p_example_anno = 0.005
fidelity_at_p_anno = np.interp(p_example_anno, error_probabilities, fidelities_unprotected)
ax1.annotate(
    'Fidelity collapses without protection', xy=(p_example_anno, fidelity_at_p_anno), 
    xytext=(p_example_anno * 1.1, fidelity_at_p_anno - 0.3),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", lw=1, alpha=0.7)
)
# NEW: Save the figure to a PDF file
plt.savefig("experiment1_fidelity.pdf", bbox_inches='tight')
print("Figure for Experiment 1 saved as 'experiment1_fidelity.pdf'")
plt.show()


################################################################################
#                                                                              #
#         EXPERIMENT 2: Validating the Dynamic Encoding Logic                  #
#                                                                              #
################################################################################
print("\n======================================================")
print("      STARTING EXPERIMENT 2: ENCODING CROSSOVER       ")
print("======================================================")

def calculate_rates(distances_km):
    loss_db_per_km, eta_d, c_km_per_s, t_op_s = 0.2, 0.9, 2e5, 1e-6
    rates_dv, rates_robust = [], []
    for L in distances_km:
        eta_sublink = 10**(-loss_db_per_km * (L / 2) / 10)
        time_per_attempt = t_op_s + (L / 2) / c_km_per_s
        prob_success_dv = (eta_sublink * eta_d)**2
        rates_dv.append(prob_success_dv / time_per_attempt)
        prob_success_robust = 2 * (eta_sublink * eta_d) * (1 - (eta_sublink * eta_d))
        rates_robust.append(prob_success_robust / time_per_attempt)
    return np.array(rates_dv), np.array(rates_robust)

# --- Run the Simulation for Experiment 2 ---
link_distances = np.linspace(1, 60, 100)
rates_dv, rates_robust = calculate_rates(link_distances)
crossover_idx = np.argmin(np.abs(rates_dv - rates_robust))
crossover_dist = link_distances[crossover_idx]
crossover_rate = rates_dv[crossover_idx]
print(f"Analysis for Experiment 2 complete. Crossover at ~{crossover_dist:.1f} km.")

# --- Visualization and Saving for Experiment 2 ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.set_title('Experiment 2: Performance Crossover of Encoding Modes', pad=20)
# FIX: Use raw strings (r'...') to prevent syntax warnings
ax2.plot(link_distances, rates_dv, '-', label=r'High-Fidelity (DV) Mode ($R \propto \eta^2$)', color='darkorange', linewidth=2.5)
ax2.plot(link_distances, rates_robust, '--', label=r'High-Robustness Mode ($R \propto \eta$)', color='darkviolet', linewidth=2.5)
ax2.plot(crossover_dist, crossover_rate, 'ko', markersize=10, label=f'Crossover Point (~{crossover_dist:.1f} km)')
ax2.axvline(x=crossover_dist, color='gray', linestyle=':', linewidth=2)
ax2.annotate(
    'Optimal mode changes here', xy=(crossover_dist, crossover_rate), 
    xytext=(crossover_dist + 5, crossover_rate * 5),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=13, bbox=dict(boxstyle="round,pad=0.3", fc="skyblue", ec="black", lw=1, alpha=0.7)
)
ax2.set_yscale('log')
ax2.set_xlabel('Elementary Link Distance (km)')
ax2.set_ylabel('Effective Entanglement Generation Rate (pairs/sec)')
ax2.legend(loc='upper right')
ax2.grid(True, which="both", ls="--")

# NEW: Save the figure to a PDF file
plt.savefig("experiment2_rate_crossover.pdf", bbox_inches='tight')
print("Figure for Experiment 2 saved as 'experiment2_rate_crossover.pdf'")
plt.show()

print("\nAll simulations and savings are complete.")