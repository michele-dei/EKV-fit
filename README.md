# EKV-fit
Heuristic Enz-Krummenacher-Vittoz (EKV) Model Fitting for Low-Power IC Design

This repository contains the Python implementation of a heuristic iterative sub-ranging technique for fitting the Enz-Krummenacker-Vittoz (EKV) model to MOSFET data. The tool is designed for low-power integrated circuit (IC) design, with a focus on accurate parameter extraction in weak and moderate inversion regions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview
The EKV model is widely used for MOSFET modeling in low-power IC design due to its accuracy and simplicity. This project provides a Python-based tool for extracting key EKV model parameters, including:
- Threshold voltage (\(V_{TH}\))
- Specific current (\(I_S\))
- Subthreshold slope parameter (\(n\))

The tool uses an iterative sub-ranging technique to refine the threshold voltage and ensure accurate fitting across different regions of operation (weak, moderate, and strong inversion).

## Features
- **Iterative Sub-Ranging**: Progressively narrows the voltage range to focus on the region of interest.
- **Robustness Against Noise**: Handles measurement noise and uncertainties effectively.
- **Flexible Fitting**: Allows control over parameters such as `fit_range_parameter`, `max_iter`, and `error_margin`.
- **Visualization**: Generates plots to visualize the measured data, fitted curve, and fit quality region.
- **Open-Source**: Built using Python and open-source libraries (NumPy, SciPy, Pandas, Matplotlib).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ekv-fitting.git
   cd ekv-fitting
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or later installed. Then, install the required libraries:
   ```bash
   pip install numpy scipy pandas matplotlib
   ```

3. **Run the Script**:
   Use the provided Python script to fit EKV model parameters to your data.

## Usage
The core functionality is encapsulated in the `fit_data` function. Here’s an example of how to use it:

```python
import mos_extract_ekv as ekv

# Define the input file and parameters
filename = 'data.csv'  # Path to your CSV file

# Perform the fitting
fitter = ekv.Fitter(filename='data.csv', 
                    temperature=27,  # Temperature in Celsius
                    fit_range_parameter=1.2
                    # key parameter for controlling the range of fitting
                   )

# Print the (main) results
print(f"VTH: {fitter.VTH_cf:.3f} V")
print(f"IS: {fitter.IS_cf:.3e} A")
print(f"n: {fitter.n_cf:.3f}")
```

### Input Data Format
The input CSV file should contain voltage and current data in the following format:
```
Voltage1, Current1, Voltage2, Current2, ...
V1, I1, V2, I2, ...
```

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this tool in your research, please cite the following paper:
```
Dei, M. Heuristic Enz–Krummenacher–Vittoz (EKV) Model Fitting for Low-Power Integrated Circuit Design: An Open-Source Implementation. Electronics 2025, 14, 1162. https://doi.org/10.3390/electronics14061162

```

For the SciPy library used in this project, please cite:
```
Virtanen, P.; et al. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nat. Methods 2020, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
```

---

## Contact
For questions or feedback, please contact:
- **Michele Dei**  
  Email: michele.dei [at] unipi.it  
  GitHub: [https://github.com/michele-dei]
```
