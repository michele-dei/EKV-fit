#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 21:42:38 2024. Version 0.1.

@author: Michele Dei (michele,dei@unipi.it)

This script contains functions for:

1. **Data Handling:**
   - `extract_voltage_current_pair()`: Extracts voltage and current data from a CSV file.
   - `extract_column_names()`: Extracts column names for a specified voltage-current pair.
   - `check_extraction()`: Checks data extraction and provides error handling.

2. **Model Fitting:**
   - `ekv_model_full_bulk_source_grounded()`: Implements the EKV model for MOSFETs.
   - `ekv_fit_full()`: Fits the EKV model to measured data.

3. **Data Analysis:**
   - `fit_quality()`: Evaluates the quality of the fitted curve.

4. **Utility Functions:**
   - `calculate_UT()`: Calculates the thermal voltage.
   - `outlog()`: Generates formatted log output.
   - `check_dictionary_compliance()`: Checks if a given dictionary contains all the necessary arguments 
   to call a specific function.

This script is designed to analyze MOSFET characteristics by fitting the EKV model 
to measured current-voltage data. It includes functions for data extraction, 
model fitting, and quality assessment.

**Key Features:**

* Iterative fitting procedure for threshold voltage refinement.
* Handles potential errors during data extraction.
* Provides informative log output and plot generation.
* Includes quality assessment metrics for the fitted model.

**Usage:**

1. **Import necessary libraries:**
   ```python
   import numpy as np
   from scipy.optimize import curve_fit
   import scipy.constants as const
   import pandas as pd
   import matplotlib.pyplot as plt
   
2. **Import and use functions:**
    # Example usage:
    import mos_extract_ekv as ekv
    filename = 'data.csv' 
    temperature = 27 
    fitted_params = ekv.fit_data(filename, temperature) 

"""

import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
import pandas as pd
import matplotlib.pyplot as plt
import inspect 

def outlog(symbol='#', line_lenght=80, step_number=None):
    """
    Outputs a formatted log line with optional step number.

    Args:
        symbol: The character to use for the log line border. 
                Defaults to '#'.
        line_length: The desired length of the log line. 
                     Defaults to 80 characters.
        step_number: An optional step number to include in the log line. 
                     Defaults to None.

    Returns:
        s: The formatted log line as a string.
    """
    
    if step_number is None:
        s = symbol*line_lenght
    else:
        n = ' '+str(step_number)+' '
        l1 = (line_lenght - len(n))//2 + 1
        s1 = symbol*l1 + n
        s2 = symbol*(line_lenght-len(s1))
        s = s1 + s2
    return s

def calculate_UT(T=27):
    """
    Calculates the thermal voltage (UT) for a given temperature in Celsius degrees.

    Args:
        T: Temperature in Celsius degrees.

    Returns:
        The thermal voltage (UT) in volts.
    """
    T_kelvin = T + 273.15  # Convert Celsius to Kelvin
    UT = (const.k * T_kelvin) / const.e  # Calculate thermal voltage
    return UT

def extract_voltage_current_pair(filename, pair_index):
    """
    Extracts voltage and current data from a CSV file.

    Args:
    filename: Path to the CSV file.
    pair_index: Index of the voltage-current pair to extract (0-indexed).

    Returns:
    A tuple containing:
      - voltage: Array of voltage values.
      - current: Array of current values.
    """
    try:
       # Read the CSV file into a pandas DataFrame
       df = pd.read_csv(filename, header=None) 

       # Get the header row (column names)
       header = df.iloc[0].tolist() 

       # Determine the number of voltage-current pairs
       num_pairs = len(header) // 2

       # Validate pair_index
       if pair_index < 0 or pair_index >= num_pairs:
           raise ValueError(f"Invalid pair_index. Must be between 0 and {num_pairs - 1}.")

       # Extract column indices for the selected pair
       voltage_col = pair_index * 2
       current_col = pair_index * 2 + 1

       # Extract voltage and current data
       voltage = df.iloc[1:, voltage_col].astype(float).values
       current = df.iloc[1:, current_col].astype(float).values

       return voltage, current

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def extract_column_names(filename, pair_index):
    """
    Extracts the column names (string information) for a specified 
    voltage-current pair from a CSV file.
    
    Args:
      filename: Path to the CSV file.
      pair_index: Index of the voltage-current pair (0-indexed).
    
    Returns:
      A tuple containing:
        - voltage_col_name: Name of the voltage column.
        - current_col_name: Name of the current column.
    """

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filename, header=None)
      
        # Get the header row (column names)
        header = df.iloc[0].tolist()
      
        # Determine the number of voltage-current pairs
        num_pairs = len(header) // 2
      
        # Validate pair_index
        if pair_index < 0 or pair_index >= num_pairs:
          raise ValueError(f"Invalid pair_index. Must be between 0 and {num_pairs - 1}.")
      
        # Extract column indices for the selected pair
        voltage_col = pair_index * 2
        current_col = pair_index * 2 + 1
      
        # Extract column names
        voltage_col_name = header[voltage_col]
        current_col_name = header[current_col]
      
        return voltage_col_name, current_col_name
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def check_extraction(filename, pair_index, verbose=True, returns_values=True):
    """
    Checks data extraction from a CSV file for a specific voltage-current pair. 
    
    Args:
      filename: Path to the CSV file.
      pair_index: Index of the voltage-current pair (0-indexed).
      verbose: If True, prints extraction information to the console. 
               Defaults to True.
      returns_values: If True, returns extracted data along with success status. 
                      Defaults to True.
    
    Returns:
      A tuple containing:
        - success: True if extraction was successful, False otherwise.
        - voltage: Array of voltage values (if extraction successful and returns_values is True).
        - current: Array of current values (if extraction successful and returns_values is True).
        - voltage_col_name: Name of the voltage column (if extraction successful and returns_values is True).
        - current_col_name: Name of the current column (if extraction successful and returns_values is True).
    
    This function attempts to extract voltage and current data from a CSV file 
    for a specified pair index. It calls the `extract_voltage_current_pair` 
    and `extract_column_names` functions internally. 
    
    If the extraction is successful:
      - If `verbose` is True, it prints the extracted data to the console.
      - If `returns_values` is True, it returns True along with the extracted 
        voltage and current arrays, and the corresponding column names.
      - If `returns_values` is False, it returns True.
    
    If the extraction fails:
      - If `verbose` is True, it prints an error message to the console.
      - If `returns_values` is True, it returns False along with None values 
        for all data.
      - If `returns_values` is False, it returns False.
    """
    
    voltage_col_name, current_col_name = extract_column_names(filename, pair_index)
    voltage, current = extract_voltage_current_pair(filename, pair_index)
    
    if voltage is not None and current is not None:
        if verbose:
            print(f"Extracted data from {filename}, pair index {pair_index}:")
            print(f"Voltage ({voltage_col_name}):", voltage)
            print(f"Current ({current_col_name}):", current)
            print("Array lenght:", len(voltage))
        if returns_values:
            return True, voltage, current, voltage_col_name, current_col_name
        else:
            return True
    else:
        if verbose:
            print(f'Extraction from {filename}, pair index {pair_index} failed')
        if returns_values:
            return False, None, None, None, None
        else:
            return False

def ekv_model_full_bulk_source_grounded(VG_normalized, VTH_normalized, IS, n):
    """
    EKV model from weak to strong inversion, saturation region.
    Source is grounded, bulk is grounded.

    Args:
        VG_normalized: Normalized gate voltage (V_G/UT).
        VTH_normalized: Normalized threshold voltage (V_TH/UT).
        IS: Specific current.
        n: Subthreshold slope parameter.

    Returns:
        ID: Drain current.
    """
    VP_normalized = (VG_normalized - VTH_normalized)/n
    E = np.exp(VP_normalized/2)
    L = np.log(1+E)
    ID = IS*L**2
    return ID

def ekv_model_full_bulk_source_grounded_log_scaled(VG_normalized, VTH_normalized, IS, n):
    """
    EKV model from weak to strong inversion, saturation region.
    Source is grounded, bulk is grounded. Logaritmic scaled current is returned.

    Args:
        VG_normalized: Normalized gate voltage (V_G/UT).
        VTH_normalized: Normalized threshold voltage (V_TH/UT).
        IS: Specific current.
        n: Subthreshold slope parameter.

    Returns:
        ID: log10 of the drain current.
    """
    ID = ekv_model_full_bulk_source_grounded(VG_normalized, VTH_normalized, IS, n)
    return np.log10(ID)

def ekv_fit_full(voltage, current, temperature=27, log_scaled=False):
    """
    Fits the EKV model of a MOSFET.

    Args:
        voltage: Array of voltage values.
        current: Array of corresponding current values.
        temperature: Temperature in Celsius. Defaults to 27°C.
        log_scaled: if True the fitting is done on the logarithmic scaled current values.

    Returns:
        A tuple containing:
            - popt: Array of optimized parameters [VTH_normalized, IS, n].
            - pcov: Covariance matrix of the optimized parameters.
    """
    UT = calculate_UT(temperature)
    VTH_normalized_guess = np.mean(voltage) / UT
    IS_guess = np.mean(current)
    subthreshold_slope_guess = 1.2
    initial_guess = [VTH_normalized_guess, IS_guess, subthreshold_slope_guess]
    VG_normalized = voltage / UT
    if log_scaled:
        assert np.all(current>0)
        log10_current = np.log10(current)
        popt, pcov = curve_fit(ekv_model_full_bulk_source_grounded_log_scaled, VG_normalized, log10_current, p0=initial_guess, maxfev=10000)
    else:
        popt, pcov = curve_fit(ekv_model_full_bulk_source_grounded, VG_normalized, current, p0=initial_guess, maxfev=10000)
    return popt, pcov

def fit_quality(voltage, current, current_fit, VTH, quality_margin=0.05):
    """
    Calculates the quality of a fitted current-voltage (I-V) curve by comparing it 
    to the original data within a specified quality margin around the threshold voltage.

    Args:
        voltage: Array of voltage values from the original data.
        current: Array of current values from the original data.
        current_fit: Array of current values from the fitted curve.
        VTH: Threshold voltage.
        quality_margin: Fractional deviation allowed between the original and 
                        fitted current values. Defaults to 0.05 (5%).

    Returns:
        A tuple containing:
            - VL: Lower voltage bound where the fit quality is within the margin.
            - VH: Upper voltage bound where the fit quality is within the margin.
            - IL: Current value at VL.
            - IH: Current value at VH.
            - idx: indexes of voltage and current vectors within the quality margin.
    """
    
    # Calculate the absolute fractional error between the original and fitted currents
    error = np.abs(current - current_fit) / current 

    # Find the indices where the error is within the quality margin
    idx = np.where(error <= quality_margin)[0]  
    
    # Find discontinuities in idx
    discontinuities = np.where(np.diff(idx) != 1)[0] + 1
    
    # Split idx into continuous segments
    segments = np.split(idx, discontinuities)
    
    # Filter out segments with length < 2
    segments = [segment for segment in segments if len(segment) >= 2]

    # Extract the lower and upper voltage bounds from the valid indices
    if len(segments)>1:
        VL, VH = None, None
    else:
        VL, VH = voltage[segments[0][0]], voltage[segments[0][-1]]  

    # Extract the corresponding current values at the lower and upper bounds
    if len(segments)>1:
        IL, IH = None, None
    else:
        IL, IH = current[segments[0][0]], current[segments[0][-1]]  

    return VL, VH, IL, IH, idx

def fit_data(filename, temperature=27, pair_index=0, max_iter=100, error_margin=0.01, 
             quality_margin=0.05, fit_range_parameter=1.01, plot_fit=True, verbose=True, 
             relax_convergence_check=False, log_scaled=False, plot_title=None):
    """
    Performs an iterative curve fitting procedure to determine MOSFET parameters 
    using the EKV model.

    Args:
        filename: Path to the CSV file containing the measured data.
        temperature: Temperature in Celsius. Defaults to 27.
        pair_index: Index of the voltage-current pair to extract from the file. 
                    Defaults to 0.
        max_iter: Maximum number of iterations for the threshold voltage update. 
                    Defaults to 100.
        error_margin: Relative percentage error margin in threshold voltage 
                     update between curve fit iterations. Defaults to 0.01.
        quality_margin: Fractional deviation allowed between original and 
                        fitted current values for fit quality assessment. 
                        Defaults to 0.05.
        fit_range_parameter: Factor for updating the threshold voltage in each iteration. 
                    Values close to 1 tend to emphasise convergence towards
                    weak inversion operation.
                    Defaults to 1.01.
        plot_fit: If True, plots the measured data and the fitted curve. 
                    Defaults to True.
        verbose: If True, prints detailed information to the console. 
                    Defaults to True.
        relax_convergence_check: if True stops iterations if new voltage 
                                 interval below fit_range_parameter*VTH_previous is 
                                 not smaller than previous.
        log_scaled: if True the fitting is done on the logarithmic scaled current values.
        plot_title: if None the default title of the plot conveys fit_range_parameter information.
                    Otherwise a custom string can be shown.

    Returns:
        A tuple containing:
            - VTH_cf: Fitted threshold voltage (V).
            - IS_cf: Fitted specific current (A).
            - n_cf: Fitted subthreshold slope parameter.
            - VL: Lower voltage bound for the fit quality region (V).
            - VH: Upper voltage bound for the fit quality region (V).
            - IL: Current at the lower voltage bound (A).
            - IH: Current at the upper voltage bound (A).
            - iteration: Number of curve-fit iterations.
            - err: residual error margin at last curve-fit iteration.
            - status: tuple [integer, string] describing the outcome of the 
                      function. If status[0] is equal to zero, the data fitting
                      terminated succesfully. If status[0] is negative, the
                      function returned an error. If status[0] is greater than
                      zero, the function returned a warning. status[1] contains
                      a descriptive message.

    This function performs an iterative curve fitting procedure to determine 
    MOSFET parameters using the EKV model. It starts with an initial full-range 
    fit, then iteratively refines the threshold voltage estimate by fitting 
    the model to a progressively narrower voltage range. The fit quality is 
    assessed within a specified voltage range around the threshold voltage. 
    The function returns the fitted parameters, the voltage and current 
    bounds of the fit quality region, and optionally plots the results.
    Optionally, a logaritmic scaling can be applied to current and the fitting
    will be done on a logarithmic model accordignly.
    """
    outcomes = {'OK':                    [0,  "Fitting terminated succesfully"],
                'E:fit_range_parameter': [-1, "Error: 'fit_range_parameter' must be greater than 1."],
                'E:data_extrarction':    [-2, "Error: data extraction from CSV file failed."],
                'E:0neg_current':        [-3, "Error: zeros or negative values of current while attempting logarithmic scaling."],
                'W:stop_iter':           [1,  "Warning: iterations stopped: subrange equals range size."],
                'W:max_iter':            [2,  "Warning: maximum number of iterations reached."],
                'W:quality_fit_undone':  [3,  "Warning: quality assesment of the fit cannot be done with the current settings."]
                }
    status = outcomes['OK']

    def vprint(s):
        if verbose:
            print(s)
    
    try:
        assert(fit_range_parameter > 1) 
    except AssertionError:
        status = outcomes['E:fit_range_parameter']
        vprint(status[1])
        return None, None, None, None, None, None, None, None, None, status
    
    vprint(outlog())
    vprint(f"Temperature: {temperature} Celsius")
    vprint(f"Error margin parameter: {error_margin*100}%")
    vprint(outlog(symbol='.'))
    go_ahead, V, I, vname, iname = check_extraction(filename, pair_index, verbose=verbose)
    UT = calculate_UT(temperature)
    
    if log_scaled:
        try:
            assert(np.all(I) > 0) 
        except AssertionError:
            status = outcomes['E:0neg_current']
            vprint(status[1])
            return None, None, None, None, None, None, None, None, None, status
    
    if go_ahead:
        
        # FITTING ON FULL RANGE
        vprint(outlog(symbol='#'))
        vprint(f'Update VTH parameter: {fit_range_parameter}')
        vprint(outlog(symbol='.', step_number=0))
        vprint('Curve fit parameters, full characteristic:')
        popt, pcov = ekv_fit_full(V, I, temperature=temperature, log_scaled=log_scaled)
        VTH_cf = popt[0]*UT
        IS_cf = popt[1]
        n_cf = popt[2]
        vprint(f"VTH: {VTH_cf*1e3} mV")
        vprint(f"IS: {IS_cf*1e9} nA")
        vprint(f"n: {n_cf}")
        idx = np.arange(len(V))
        err = None
        
        # FITTING ITERATIONS
        iteration = 1
        while iteration < max_iter:
            VTH_cf_prev = VTH_cf
            idx_prev = idx
            vprint(outlog(symbol='.', step_number=iteration))
            vprint('Curve fit parameters:')
            idx = np.where(V<fit_range_parameter*VTH_cf_prev)[0]
            if not(relax_convergence_check):
                if not(len(idx)<len(idx_prev)):
                    status = outcomes['W:stop_iter']
                    vprint(status[1])
                    break
            popt, pcov = ekv_fit_full(V[idx], I[idx], temperature=temperature, log_scaled=log_scaled)
            VTH_cf = popt[0]*UT
            IS_cf = popt[1]
            n_cf = popt[2]
            err = np.abs(VTH_cf-VTH_cf_prev)/np.abs((VTH_cf+VTH_cf_prev)*2)
            vprint(f"VTH: {VTH_cf*1e3} mV")
            vprint(f"IS: {IS_cf*1e9} nA")
            vprint(f"n: {n_cf}")
            vprint(f"Error margin: {err}")
            iteration += 1
            if err < error_margin:
                vprint('Converged within the given error margin.')
                break
        
        if iteration == max_iter:
            status = outcomes['W:max_iter']
            vprint(status[1])
        
        vprint(outlog(symbol='#'))
        vprint(f'Evaluation of fit quality. Error margin: {quality_margin*100}%')
        ID_cf = ekv_model_full_bulk_source_grounded(V/UT, *popt)
        try:
            VL, VH, IL, IH, idx = fit_quality(V, I, ID_cf, VTH_cf, quality_margin)
            vprint(f"Lower voltage bound (VL): {VL*1e3:.3f} mV")
            vprint(f"Upper voltage bound (VH): {VH*1e3:.3f} mV")
            vprint(f"Current at VL (IL): {IL*1e9:.3e} nA")
            vprint(f"Current at VH (IH): {IH*1e9:.3e} nA")
        except:
            VL, VH, IL, IH, idx = 0, 0, 0, 0, None
            status = outcomes['W:quality_fit_undone']
            vprint(status[1])
            
        # PLOT          
        if plot_fit:
            fig = plt.figure()
            ax0 = fig.add_subplot(111)
            ax0.semilogy(V, I*1e6, 'o:', color='orange', label='Data', markevery=10)
            ax0.semilogy(V, ID_cf*1e6, 'k--', label='Fit:\n $V_{TH}$ = '+f'{VTH_cf*1e3:.3f} mV\n $I_S$ = {IS_cf*1e9:.3f} nA\n $n$ = {n_cf:.3f}')
            if not(idx is None):
                # Find discontinuities in idx
                discontinuities = np.where(np.diff(idx) != 1)[0] + 1
                
                # Split idx into continuous segments
                segments = np.split(idx, discontinuities)
                print(segments)
                
                # Filter out segments with length < 2
                segments = [segment for segment in segments if len(segment) >= 2]
                
                # Plot each continuous segment
                for segment in segments:
                    if len(segments) <= 1:
                        ax0.semilogy(V[segment], ID_cf[segment]*1e6, 'k', lw=2.5, label=f'Fit within {quality_margin*100}% error:\n $V_G \in $'+f'({VL*1e3:.3f}, {VH*1e3:.3f}) mV\n $I_D \in $ ({IL*1e9:.3f}, {IH*1e9:.3f}) nA')
                    else:
                        ax0.semilogy(V[segment], ID_cf[segment]*1e6, 'k', lw=2.5, label=f'Fit within {quality_margin*100}% error')
                
            ax0.grid(linestyle=':')
            ax0.set_xlabel('$V_{G}$ [V]', fontsize=12)
            ax0.set_ylabel('$I_D$ [$\mu$A]', fontsize=12)
            ax0.legend(fontsize=10)
            ax0.set_ylim([min(I*1e6)*0.5, max(I*1e6)*2])
            #ax0.set_title(filename + ': ' + vname + ';' + iname + f';\n T={temperature:0.1f} C; error margin={error_margin*100}%; Threshold update={fit_range_parameter}', fontsize=10)
            if plot_title is None:
                ax0.set_title(f'fit_range_pararmeter={fit_range_parameter}', fontsize=10)
            else:
                ax0.set_title(plot_title)
                
        vprint(outlog())
        return VTH_cf, IS_cf, n_cf, VL, VH, IL, IH, iteration, err, status
    else:
        status = outcomes['E:data_extrarction']
        vprint(status[1])
        vprint(outlog())
        return None, None, None, None, None, None, None, None, None, status
    
def check_dictionary_compliance(dictionary, function):
    """
    Checks if a given dictionary contains all the necessary arguments 
    to call a specific function.

    Args:
        dictionary: The dictionary containing the arguments.
        function: The function to check against.

    Returns:
        A tuple containing:
            - `status`: A boolean indicating whether the dictionary is compliant.
            - `message`: A string describing the compliance status. 
    """

    outcomes = {
        'OK': [True, 'Dictionary is compliant with function arguments.'],
        'OK-filter': [True, 'Some keys of the dictionary have been ignored.'],
        'E:missing': [False, 'Missing compulsory arguments: ']
    }
    status = outcomes['OK']

    # Get argument names of the function
    info = inspect.getfullargspec(function)
    args = list(info.args)

    # Check for extra keys in the dictionary and filter them
    if not all(key in args for key in dictionary.keys()):
        dictionary_filtered = {key: dictionary[key] for key in dictionary if key in args}
        status = outcomes['OK-filter']
    else:
        dictionary_filtered = dictionary

    # Check for missing compulsory arguments
    compulsory_args = [x for x, y in inspect.signature(function).parameters.items() if y.default is y.empty]
    missing_args = [key for key in compulsory_args if key not in dictionary_filtered]
    if missing_args:
        estring = f"{missing_args}"
        status = outcomes['E:missing']
        status[1] += estring

    return status
###############################################################################