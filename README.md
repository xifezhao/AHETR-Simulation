# AHETR Protocol Validation: Simulation and Analysis

This repository contains a Jupyter Notebook script that validates the core principles of the Adaptive Hybrid Error-Transparent Repeater (AHETR) protocol through numerical simulations. The AHETR protocol is designed to bridge the implementation gap in quantum repeater architectures, offering a balance between high performance and near-term feasibility.

## Table of Contents

- [AHETR Protocol Validation: Simulation and Analysis](#ahetr-protocol-validation-simulation-and-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Experiment 1: Heralded Error Transparency](#experiment-1-heralded-error-transparency)
  - [Experiment 2: Dynamic Encoding Logic](#experiment-2-dynamic-encoding-logic)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

The AHETR protocol introduces two key innovations:
1.  **Heralded Error Transparency:** A lightweight error-detecting code identifies and discards locally corrupted quantum states, preventing error propagation without full quantum error correction overhead.
2.  **Dynamic Encoding:** The repeater autonomously switches between high-fidelity (DV) and high-robustness (Robust) entanglement generation schemes based on real-time channel loss measurements, maximizing entanglement distribution rates under varying conditions.

This script simulates these two core mechanisms to demonstrate the protocol's effectiveness.

## Features

*   **Experiment 1:** Simulates and compares the entanglement fidelity of an unprotected quantum swap against the AHETR protocol's error transparency mechanism as a function of physical gate error probability.
*   **Experiment 2:** Calculates and visualizes the entanglement generation rate for both High-Fidelity (DV) and High-Robustness modes across varying link distances, demonstrating the optimal crossover point for dynamic encoding.
*   **Publication-Ready Plots:** Generates professional-looking plots using `matplotlib` with `seaborn` style, suitable for academic papers.
*   **PDF Output:** Automatically saves the generated figures as PDF files (`.pdf`) for easy integration into documents.

## Installation

To run this script, you will need Python 3 and the following libraries:

```bash
pip install cirq matplotlib numpy seaborn
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourGitHubUsername/ahetr-protocol-validation.git
    cd ahetr-protocol-validation
    ```
2.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook ahetr_protocol_validation.ipynb # Assuming you save the script as a .ipynb
    ```
    Alternatively, you can copy the script content into a Python file (`.py`) and run it from your terminal:
    ```bash
    python ahetr_protocol_validation.py
    ```

The script will execute the two experiments, print progress messages to the console, and generate PDF plots in the same directory.

## Experiment 1: Heralded Error Transparency

This experiment demonstrates how the AHETR protocol, by employing a "detect-and-discard" strategy, maintains high entanglement fidelity even in the presence of physical gate errors, unlike unprotected schemes.

**Output:** `experiment1_fidelity.pdf`

![Experiment 1 Fidelity Plot Placeholder](experiment1_fidelity.pdf)
*Figure 1: Efficacy of Heralded Error Transparency. The plot shows the final entanglement fidelity against physical gate error probability ($p_{err}$). The AHETR protocol (blue) maintains near-unity fidelity by detecting and discarding states with single-qubit errors, while the unprotected scheme's fidelity (red) collapses catastrophically.*

## Experiment 2: Dynamic Encoding Logic

This experiment validates the need for dynamic encoding by illustrating the performance crossover between the High-Fidelity (DV) mode and the High-Robustness (Robust) mode. The plot identifies the optimal link distance at which to switch between modes to maximize the entanglement generation rate.

**Output:** `experiment2_rate_crossover.pdf`

![Experiment 2 Rate Crossover Plot Placeholder](experiment2_rate_crossover.pdf)
*Figure 2: Performance Crossover of Encoding Modes. The entanglement generation rate for DV mode (orange) and Robust mode (violet) is plotted against link distance, showing a clear crossover point where optimal mode changes are necessary.*

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
