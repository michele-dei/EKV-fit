#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 21:42:38 2024. 
Updated on Wed Feb 5 14:40:55 2035. Version 0.2.

                            **MOS_EXTRACT_EKV.PY**

@author: Michele Dei (michele.dei@unipi.it)

This script contains functions for:

1. **Data Handling:**
    - `read_data_from_csv_file()`: Reads data from a CSV file into a Pandas DataFrame.
    - `dataframe_to_array_list()`: Converts a Pandas DataFrame to a list of NumPy arrays.
    - `extract_voltage_current_pair()`: Extracts voltage and current data from a CSV file.
    - `extract_column_names()`: Extracts column names for a specified voltage-current pair.
    - `check_extraction()`: Checks data extraction and provides error handling.
    - `filter_positive_current()`: Filters voltage and current arrays to include only positive current values.
    - `write_data_to_csv()`: Writes data from a NumPy array to a CSV file.
    - `noisy_data_to_csv()`: Generates noisy current data and saves it to a CSV file.

2. **Model Fitting:**
    - `ekv_model_full_bulk_source_grounded()`: Implements the EKV model for MOSFETs.
    - `ekv_model_full_bulk_source_grounded_log_scaled()`: Implements the EKV model for MOSFETs with logarithmic scaling.
    - `ekv_fit_full()`: Fits the EKV model to measured data.
    - `Fitter` class: Performs iterative EKV model fitting with refined threshold voltage estimation.
    - `FitterECP` class: Performs EKV model fitting with the ECP method.

3. **Data Analysis:**
    - `fit_quality()`: Evaluates the quality of the fitted curve.
    - `stats_noise()`: Performs statistical noise analysis on fitted parameters.

4. **Utility Functions:**
    - `calculate_UT()`: Calculates the thermal voltage.
    - `outlog()`: Generates formatted log output.
    - `check_dictionary_compliance()`: Checks if a given dictionary contains all the necessary arguments to call a specific function.
    - `logspace_octaves()`: Generates logarithmically spaced points over a specified number of octaves.
    - `find_segments()`: Finds continuous segments in an array.

This script is designed to analyze MOSFET characteristics by fitting the EKV model
to measured current-voltage data. It includes functions for data extraction,
model fitting, and quality assessment.

**Key Features:**

* Iterative fitting procedure for threshold voltage refinement.
* Handles potential errors during data extraction.
* Provides informative log output and plot generation.
* Includes quality assessment metrics for the fitted model.
* Class-based structure for organized fitting procedures (`Fitter` and `FitterECP`).
* Noise analysis utilities for simulating and analyzing noisy data.
* ECP (Enz, Chicco, Pezzotta) fitting method for comparison.
* Collection class for comparing different fitters (`FitterCollection`).

**Usage:**

1. **Import necessary libraries:**
    ```python
    import numpy as np
    from scipy.optimize import curve_fit
    import scipy.constants as const
    import pandas as pd
    import matplotlib.pyplot as plt
    ```

2. **Import and use functions:**
    ```python
    import mos_extract_ekv as ekv
    filename = 'data.csv'
    temperature = 27
    fitter = ekv.Fitter(filename, temperature) # Create a Fitter object
    # Access fitted parameters:
    if fitter.status[0] == 0:
        print(f"VTH: {fitter.VTH_cf:.3f} V")
        print(f"IS: {fitter.IS_cf:.3e} A")
        print(f"n: {fitter.n_cf:.3f}")
    else:
        print(f"Fitting failed: {fitter.status[1]}")
    ```

**Changes:**
- Fitter class added for better handling different comparisons
- 'fit_data' is now a method of Fitter
- Added 'plot_fit' method outside of 'fit_data' method
- 'fit_data' returns 'err' as array containing the value of all iterations
- Added `read_data_from_csv_file`, `dataframe_to_array_list`, `filter_positive_current`, `write_data_to_csv`, `noisy_data_to_csv`, `logspace_octaves`, and `stats_noise` functions.
- Added `FitterECP` class for ECP fitting method.
- Added `FitterCollection` class for comparing different fitters.
- Added `find_segments` function.

**License:**
    This work is licensed under the Creative Commons Attribution 4.0 International License.
    To view a copy of this license, visit:
    http://creativecommons.org/licenses/by/4.0/

    You are free to:
    - Share: Copy and redistribute the material in any medium or format.
    - Adapt: Remix, transform, and build upon the material for any purpose, even commercially.
    Under the following terms:
    - Attribution: You must give appropriate credit, provide a link to the license, and indicate
      if changes were made. You may do so in any reasonable manner, but not in any way that
      suggests the licensor endorses you or your use.

    For more information, see the full license text at the link above.
"""

import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
tableau_colors = list(mcolors.TABLEAU_COLORS.values())
import csv
import inspect 

###############################################################################
# GENERAL UTILITY FUNCTIONS :
    
def read_data_from_csv_file(csv_file):
    """
    Reads data from a list of CSV file into Pandas DataFrame.

    Args:
        csv_file: each string is a CSV filename.

    Returns:
        Pandas DataFrame.  Returns an empty list if there are errors 
        reading the file.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return []  # Return empty list if any file is not found
    except pd.errors.ParserError as e: # Catch errors during the parsing of a file
        print(f"Error parsing file {csv_file}: {e}")
        return [] # Return empty list if any error occurs
    except Exception as e: # Catch other possible exceptions
        print(f"An unexpected error occurred while reading file {csv_file}: {e}")
        return [] # Return empty list if any error occurs

    return df
    
def dataframe_to_array_list(df):
    """
    Converts the columns of a Pandas DataFrame to a list of NumPy arrays.

    Args:
        df: The Pandas DataFrame.

    Returns:
        A list of NumPy arrays, where each array represents a column from the DataFrame.
        Returns an empty list if the input is not a DataFrame or if an error occurs.
    """

    if not isinstance(df, pd.DataFrame):
        return []  # Return empty list if input is not a DataFrame

    array_list = []
    try:
        for column_name in df.columns:
            array = df[column_name].to_numpy()  # Convert each column to a NumPy array
            array_list.append(array)
        return array_list
    except Exception as e: # Catch other possible exceptions
        print(f"An unexpected error occurred while processing the dataframe: {e}")
        return [] # Return empty list if any error occurs

def logspace_octaves(value, num_octaves, points_per_octave):
    """
    Generates an array of points ranging from `value` to `value * 2^num_octaves`,
    with `points_per_octave` points per octave, equally spaced in logarithmic space.

    Args:
        value (float): Initial value.
        num_octaves (int): Number of octaves to span.
        points_per_octave (int): Number of points per octave.

    Returns:
        numpy.ndarray: Array of logarithmically spaced points.
    """
    # Calculate the total number of points
    total_points = num_octaves * points_per_octave + 1
    
    # Generate logarithmically spaced points
    log_start = np.log2(value)
    log_end = log_start + num_octaves
    log_space = np.linspace(log_start, log_end, total_points)
    
    # Convert back to linear space
    points = 2 ** log_space
    
    return points

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

def find_segments(idx, min_length=3):
    """
    Find continuous segments in idx of at least minimum length (min_length).
    
    Parameters
    ----------
    idx : integer array
        Input array.
    min_length : TYPE, optional
        Minimum length of segments. The default is 3.

    Returns
    -------
    segments : list of array
        Indexes in idx of continuous segments.

    """
    segments = []
    if not(idx is None):
        # Find discontinuities in idx
        discontinuities = np.where(np.diff(idx) != 1)[0] + 1
        
        # Split idx into continuous segments
        segments = np.split(idx, discontinuities)
        
        # Filter out segments with length < 2
        segments = [segment for segment in segments if len(segment) >= min_length-1]
    
    return segments

def fit_quality(voltage, current, current_fit, VTH, quality_margin=0.05, multi_segment=True):
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
        multi_segment: in multi-segment case, calculation is performed in each segment

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
    
    segments = find_segments(idx)

    # Extract the lower and upper voltage bounds from the valid indices
    if len(segments)>1:
        if multi_segment:
            VL, VH = [], []
            for s in segments:
                VL.append(voltage[s[0]])
                VH.append(voltage[s[-1]])
            VL = np.array(VL)
            VH = np.array(VH)
        else:
            VL, VH = None, None
    else:
        VL, VH = voltage[segments[0][0]], voltage[segments[0][-1]]  

    # Extract the corresponding current values at the lower and upper bounds
    if len(segments)>1:
        if multi_segment:
            IL, IH = [], []
            for s in segments:
                IL.append(current[s[0]])
                IH.append(current[s[-1]])
            IL = np.array(VL)
            IH = np.array(VH)
        else:
            IL, IH = None, None
    else:
        IL, IH = current[segments[0][0]], current[segments[0][-1]]  

    return VL, VH, IL, IH, idx

###############################################################################
# FITTER CLASS :

class Fitter:

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
            - err: residual error margin array for each curve-fit iteration.
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

    def __init__(self, filename, 
                 temperature=27, 
                 pair_index=0, 
                 max_iter=100, 
                 error_margin=0.01, 
                 quality_margin=0.05, 
                 fit_range_parameter=1.01, 
                 relax_convergence_check=False, 
                 log_scaled=False, 
                 verbose=False,
                 plot_fit=True, 
                 plot_title=None,
                 figsize=[6.4, 4.8]):
        
        self.filename = filename
        self.temperature = temperature
        self.pair_index = pair_index
        self.max_iter = max_iter
        self.error_margin = error_margin
        self.quality_margin = quality_margin
        self.fit_range_parameter = fit_range_parameter
        self.verbose = verbose
        self.relax_convergence_check = relax_convergence_check
        self.log_scaled = log_scaled
        # Initialize fitted parameters
        self.ID_cf = None
        self.VTH_cf = None  
        self.IS_cf = None
        self.n_cf = None
        self.VL = None
        self.VH = None
        self.IL = None
        self.IH = None
        self.iteration = None
        self.err = None
        self.status = None
        # Call fit_data method
        self.fit_data()
        # Call calculate_rel_err method
        self.calculate_rel_err()
        # Call plot method
        if plot_fit:
            self.plot(plot_title=plot_title, figsize=figsize)

    def fit_data(self):
        """
        Performs the EKV model fitting.
        """
        outcomes = {'OK': [0, "Fitting terminated successfully"],
                    'E:fit_range_parameter': [-1, "Error: 'fit_range_parameter' must be greater than 1."],
                    'E:data_extrarction': [-2, "Error: data extraction from CSV file failed."],
                    'E:0neg_current': [-3, "Error: zeros or negative values of current while attempting logarithmic scaling."],
                    'W:stop_iter': [1, "Warning: iterations stopped: subrange equals range size."],
                    'W:max_iter': [2, "Warning: maximum number of iterations reached."],
                    'W:quality_fit_undone': [3, "Warning: quality assesment of the fit cannot be done with the current settings."]
                    }
        self.status = outcomes['OK']

        def vprint(s):
            if self.verbose:
                print(s)

        # try:
        #     assert (self.fit_range_parameter > 1)
        # except AssertionError:
        #     self.status = outcomes['E:fit_range_parameter']
        #     vprint(self.status[1])
        #     return self  # Return self for chaining

        vprint(outlog())
        vprint(f"Temperature: {self.temperature} Celsius")
        vprint(f"Error margin parameter: {self.error_margin * 100}%")
        vprint(outlog(symbol='.'))
        go_ahead, V, I, vname, iname = check_extraction(self.filename, self.pair_index, verbose=False)
        UT = calculate_UT(self.temperature)

        if self.log_scaled:
            try:
                assert (np.all(I) > 0)
            except AssertionError:
                self.status = outcomes['E:0neg_current']
                vprint(self.status[1])
                return self  # Return self for chaining

        if go_ahead:
            
            # FITTING ON FULL RANGE
            vprint(outlog(symbol='#'))
            vprint(f'Update VTH parameter: {self.fit_range_parameter}')
            vprint(outlog(symbol='.', step_number=0))
            vprint('Curve fit parameters, full characteristic:')
            popt, pcov = ekv_fit_full(V, I, temperature=self.temperature, log_scaled=self.log_scaled)
            VTH_cf = popt[0]*UT
            IS_cf = popt[1]
            n_cf = popt[2]
            vprint(f"VTH: {VTH_cf*1e3} mV")
            vprint(f"IS: {IS_cf*1e9} nA")
            vprint(f"n: {n_cf}")
            idx = np.arange(len(V))
            err = []
            
            # FITTING ITERATIONS
            iteration = 1
            while iteration < self.max_iter:
                VTH_cf_prev = VTH_cf
                idx_prev = idx
                vprint(outlog(symbol='.', step_number=iteration))
                vprint('Curve fit parameters:')
                idx = np.where(V<self.fit_range_parameter*VTH_cf_prev)[0]
                if not(self.relax_convergence_check):
                    if not(len(idx)<len(idx_prev)):
                        status = outcomes['W:stop_iter']
                        vprint(status[1])
                        break
                popt, pcov = ekv_fit_full(V[idx], I[idx], temperature=self.temperature, log_scaled=self.log_scaled)
                VTH_cf = popt[0]*UT
                IS_cf = popt[1]
                n_cf = popt[2]
                err.append(np.abs(VTH_cf-VTH_cf_prev)/np.abs((VTH_cf+VTH_cf_prev)*2))
                vprint(f"VTH: {VTH_cf*1e3} mV")
                vprint(f"IS: {IS_cf*1e9} nA")
                vprint(f"n: {n_cf}")
                vprint(f"Error margin: {err[-1]}")
                iteration += 1
                if err[-1] < self.error_margin:
                    vprint('Converged within the given error margin.')
                    break
            
            if iteration == self.max_iter:
                status = outcomes['W:max_iter']
                vprint(status[1])
            
            vprint(outlog(symbol='#'))
            vprint(f'Evaluation of fit quality. Error margin: {self.quality_margin*100}%')
            ID_cf = ekv_model_full_bulk_source_grounded(V/UT, *popt)
            try:
                VL, VH, IL, IH, idx = fit_quality(V, I, ID_cf, VTH_cf, self.quality_margin)
                if isinstance(VL, np.ndarray):
                    vprint('Multi segmented VL, VH ranges')
                    for i in range(len(VL)):
                        vprint(f'Range {i}')
                        vprint(f"   Lower voltage bound (VL): {VL[i]*1e3:.3f} mV")
                        vprint(f"   Upper voltage bound (VH): {VH[i]*1e3:.3f} mV")
                        vprint(f"   Current at VL (IL): {IL[i]*1e9:.3e} nA")
                        vprint(f"   Current at VH (IH): {IH[i]*1e9:.3e} nA")
                else:
                    vprint(f"Lower voltage bound (VL): {VL*1e3:.3f} mV")
                    vprint(f"Upper voltage bound (VH): {VH*1e3:.3f} mV")
                    vprint(f"Current at VL (IL): {IL*1e9:.3e} nA")
                    vprint(f"Current at VH (IH): {IH*1e9:.3e} nA")
            except:
                VL, VH, IL, IH, idx = 0, 0, 0, 0, None
                status = outcomes['W:quality_fit_undone']
                vprint(status[1])
            
            self.VG = V
            self.ID = I
            self.ID_cf = ID_cf
            self.VTH_cf = VTH_cf 
            self.IS_cf = IS_cf 
            self.n_cf = n_cf 
            self.VL = VL 
            self.VH = VH 
            self.IL = IL
            self.IH = IH 
            self.iteration = iteration
            self.err = err
            
            return self  # Return self for chaining

        else:
            self.status = outcomes['E:data_extrarction']
            vprint(self.status[1])
            vprint(outlog())
            return self  # Return self for chaining
        
    def calculate_rel_err(self):
        ID = self.ID
        IDf = self.ID_cf
        self.rel_err = np.abs(ID-IDf)/ID
        
    def plot(self, plot_title=None, figsize=[6.4, 4.8]):
        """Plots the measured data and the fitted EKV curve."""
        
        V, I = self.VG, self.ID
        ID_cf, VTH_cf, IS_cf, n_cf = self.ID_cf, self.VTH_cf, self.IS_cf, self.n_cf
        VL, VH, IL, IH = self.VL, self.VH, self.IL, self.IH        
        quality_margin = self.quality_margin
        
        try:
            VL0, VH0, IL0, IH0, idx = fit_quality(V, I, ID_cf, VTH_cf, quality_margin)      
        except IndexError:
            idx = None, None, None, None, None
        
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111)
        ax0.semilogy(V, I*1e6, ':', color='orange', label='Data')
        ax0.semilogy(V, ID_cf*1e6, 'k--', label='$V_{TH}$ = '+f'{VTH_cf*1e3:.0f} mV\n $I_S$ = {IS_cf*1e9:.0f} nA\n $n$ = {n_cf:.3f}')
        if not(idx is None):

            segments = find_segments(idx)
            
            # Plot each continuous segment
            for i, segment in enumerate(segments):
                if len(segments) <= 1:
                    ax0.semilogy(V[segment], ID_cf[segment]*1e6, 'k', lw=2.5, label='$V_G \in $'+f'({VL*1e3:.0f}, {VH*1e3:.0f}) mV\n $I_D \in $ ({IL*1e9:.0f}, {IH*1e9:.0f}) nA')
                else:
                    if i == 0:
                        ax0.semilogy(V[segment], ID_cf[segment]*1e6, 'k', lw=2.5, label=f'Fit within {quality_margin*100}% error')
                    else:
                        ax0.semilogy(V[segment], ID_cf[segment]*1e6, 'k', lw=2.5)
                        
        ax0.grid(linestyle=':')
        ax0.set_xlabel('$V_{G}$ [V]', fontsize=12)
        ax0.set_ylabel('$I_D$ [$\mu$A]', fontsize=12)
        ax0.legend(fontsize=8)
        ax0.set_ylim([min(I*1e6)*0.5, max(I*1e6)*2])
        #ax0.set_title(filename + ': ' + vname + ';' + iname + f';\n T={temperature:0.1f} C; error margin={error_margin*100}%; Threshold update={fit_range_parameter}', fontsize=10)
        
        if plot_title is None:
            ax0.set_title(f'fit_range_pararmeter={self.fit_range_parameter}', fontsize=10)
        else:
            ax0.set_title(plot_title, fontsize=10)
        plt.tight_layout()
            
        return fig

###############################################################################
# ECP (Enz, Chicco, Pezzotta) FITTER CLASS : [doi: 10.1109/MSSC.2017.2712318]
    
class FitterECP:
    
    def __init__(self, filename, 
                 temperature=27, 
                 pair_index=0, 
                 quality_margin = 0.05,
                 verbose=False,
                 plot_fit=True, 
                 plot_title=None,
                 figsize=[6.4, 4.8]):
        
        self.filename = filename
        self.temperature = temperature
        self.pair_index = pair_index
        self.verbose = verbose
        self.quality_margin = quality_margin
        self.figsize = figsize
        # Initialize fitted parameters
        self.ID_cf = None
        self.VTH_cf = None  
        self.IS_cf = None
        self.n_cf = None
        self.status = None
        # Call fit_data method
        self.fit_data()
        # Call calculate_rel_err method
        self.calculate_rel_err()
        # Call plot method
        if plot_fit:
            self.plot(plot_title=plot_title)
    
    def fit_data(self, skip_initial_points=3):
        outcomes = {'OK': [0, "ECP Fitting terminated successfully"],
                    'E:data_extrarction': [-2, "Error: data extraction from CSV file failed."],
                    }
        self.status = outcomes['OK']

        def vprint(s):
            if self.verbose:
                print(s)

        vprint(outlog())
        vprint(f"Temperature: {self.temperature} Celsius")
        vprint(outlog(symbol='.'))
        go_ahead, V, I, vname, iname = check_extraction(self.filename, self.pair_index, verbose=False)
        UT = calculate_UT(self.temperature)
        
        if go_ahead:
            
            # Find n_cf
            gm = np.diff(I)/np.diff(V)
            IM = (I[:-1] + I[1:])/2
            curve1 = IM/(gm*UT)
            n_cf = np.min(curve1[skip_initial_points])
            vprint(f"n: {n_cf}")
                        
            # Find IS_cf
            IS = n_cf**2/curve1**2*IM
            IS_cf = np.max(IS)
            vprint(f"IS: {IS_cf*1e9} nA")
            
            """
            plt.semilogx(IM, IS)
            curve2 = n_cf*(IM/IS_cf)**0.5
            plt.loglog(IM, curve1)
            plt.loglog(IM, n_cf*np.ones_like(IM))
            plt.loglog(IM, curve2)            
            """
        
            # Find VTH_cf 
            def fit_vth_normalized(VG_normalized, ID, IS, n, VTH_guess=0.0):
                """
                Fits VTH_normalized to the given ID data using the EKV model.
            
                Args:
                    VG_normalized: NumPy array of normalized gate voltages.
                    ID: NumPy array of drain currents.
                    IS: Specific current (scalar).
                    n: Subthreshold slope parameter (scalar).
                    VTH_guess: Initial guess for VTH_normalized.
            
                Returns:
                    The fitted VTH_normalized value.
                """
            
                def ekv_fixed_is_n(VG_normalized, VTH_normalized):
                    return ekv_model_full_bulk_source_grounded(VG_normalized, VTH_normalized, IS, n)
            
                popt, pcov = curve_fit(ekv_fixed_is_n, VG_normalized, ID, p0=[VTH_guess])
                return popt[0]
            
                       
            VG_normalized = V/UT
            VTH_normalized_guess = np.mean(VG_normalized)            
            VTH_fitted_normalized = fit_vth_normalized(VG_normalized, I, IS_cf, n_cf, VTH_guess=VTH_normalized_guess) # guess of 3
            VTH_cf = VTH_fitted_normalized*UT
            
            # Find ID_cf 
            ID_cf = ekv_model_full_bulk_source_grounded(VG_normalized, VTH_fitted_normalized, IS_cf, n_cf)
            
            # Log
            vprint(f"VTH: {VTH_cf*1e3} mV")
            
            # Class attributes
            self.VG = V
            self.ID = I
            self.ID_cf = ID_cf
            self.VTH_cf = VTH_cf 
            self.IS_cf = IS_cf 
            self.n_cf = n_cf 
            
            return self
        
    def calculate_rel_err(self):
        ID = self.ID
        IDf = self.ID_cf
        self.rel_err = np.abs(ID-IDf)/ID
        
    def plot(self, plot_title=None, figsize=[6.4, 4.8]):
        
        # Uses the plot metod from Fitter class
        fit = Fitter(self.filename, plot_fit=False, verbose=False, max_iter=1)
        fit.VG, fit.ID = self.VG, self.ID
        fit.ID_cf, fit.VTH_cf, fit.IS_cf, fit.n_cf = self.ID_cf, self.VTH_cf, self.IS_cf, self.n_cf
        fit.quality_margin = self.quality_margin
        fit.plot(plot_title=plot_title, figsize=self.figsize)
        
    
###############################################################################
# Fitter Collection

class FitterCollection():
    
    def __init__(self):
        """
        Initializes the FitterComparison class.
        
        Args:
            fitters: List of Fitter instances to compare.
        """
        self.fitters = []   

    def add_fitter(self, fitter, label=None):
        """Adds a Fitter instance to the comparison."""
        if label is None:
            n = len(self.fitters) + 1
            label = f'Fitter{n}'
        fitter.label = label
        self.fitters.append(fitter)        
        
    def compare_parameters(self, parameter_name):
        """Compares a specific parameter across all Fitters."""
        values = [getattr(fitter, parameter_name) for fitter in self.fitters]
        labels = [f.label for i, f in enumerate(self.fitters)]

        plt.figure()
        plt.bar(labels, values)
        plt.title(f"Comparison of {parameter_name}")
        plt.ylabel(parameter_name)
        plt.show()
        
    def generic_plot(self, x_data_name, y_data_name, plot_type='plot'):
        """Generates a synoptic plot of data from all Fitters."""
        plt.figure()
        labels = [f.label for f in self.fitters]  # Assuming each Fitter has a 'label' attribute
        for i, fitter in enumerate(self.fitters):
            x_data = getattr(fitter, x_data_name)
            y_data = getattr(fitter, y_data_name)

            if plot_type == 'plot':
                plt.plot(x_data, y_data, label=labels[i])
            elif plot_type == 'semilogx':
                plt.semilogx(x_data, y_data, label=labels[i])
            elif plot_type == 'semilogy':
                plt.semilogy(x_data, y_data, label=labels[i])
            elif plot_type == 'loglog':
                plt.loglog(x_data, y_data, label=labels[i])
            else:
                raise ValueError(f"Invalid plot_type: {plot_type}. Choose from 'plot', 'semilogx', 'semilogy', 'loglog'.")

        plt.title("Synoptic Plot")
        plt.xlabel(x_data_name)
        plt.ylabel(y_data_name)
        plt.legend()
        plt.show()

    def synoptic_plot(self, plot_title=None, figsize=[6.4, 4.8], quality_margin=0.05):
        """Generates a synoptic plot of data from all Fitters."""
        
        fig = plt.figure(figsize=figsize)
        ax0 = fig.add_subplot(111)
        labels = [f.label for i, f in enumerate(self.fitters)]
        color_cycle = iter(tableau_colors)

        for j, fitter in enumerate(self.fitters):
            V = getattr(fitter, 'VG')
            I = getattr(fitter, 'ID')
            ID_cf = getattr(fitter, 'ID_cf')
            VTH_cf = getattr(fitter, 'VTH_cf')
            IS_cf = getattr(fitter, 'IS_cf')
            n_cf = getattr(fitter, 'n_cf')
            
            VL, VH, IL, IH, idx = fit_quality(V, I, ID_cf, VTH_cf, quality_margin)        
            
            color = next(color_cycle)  # Get the next color from the cycle
            ax0.semilogy(V, I*1e6, ':', color=color, label=labels[j], markevery=10)
            label0 = '$V_{TH}$ = '+f'{VTH_cf*1e3:.0f} mV; $I_S$ = {IS_cf*1e9:.0f} nA; $n$ = {n_cf:.3f}'
            ax0.semilogy(V, ID_cf*1e6, color=color, alpha=0.7, lw=1)
            if not(idx is None):

                segments = find_segments(idx)
                
                # Plot each continuous segment
                for i, segment in enumerate(segments):
                    if len(segments) <= 1:                        
                        label1 = f'$V_G \in $'+f'({VL*1e3:.0f}, {VH*1e3:.0f}) mV; $I_D \in $ ({IL*1e9:.1f}, {IH*1e9:.1f}) nA'
                        ax0.semilogy(V[segment], ID_cf[segment]*1e6, color=color, lw=2.5, label=label0+'\n'+label1)
                    else:
                        if i == 0:
                            #label0 = f'Fit within {quality_margin*100}% error'
                            ax0.semilogy(V[segment], ID_cf[segment]*1e6, color=color, lw=2.5, label=label0)
                        else:
                            ax0.semilogy(V[segment], ID_cf[segment]*1e6, color=color, lw=2.5)
                            
        ax0.grid(linestyle=':')
        ax0.set_xlabel('$V_{G}$ [V]', fontsize=12)
        ax0.set_ylabel('$I_D$ [$\mu$A]', fontsize=12)
        leg = ax0.legend(fontsize=8)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        ax0.set_ylim([min(I*1e6)*0.5, max(I*1e6)*2])
        if plot_title is None:
            pass
        else:
            ax0.set_title(plot_title, fontsize=10)
        
        plt.tight_layout()
        return fig
    
    
###############################################################################
# UTILITIES FOR NOISE ANALYSIS :

def filter_positive_current(V, I):
    """
    Filters the arrays V and I to return only the values where I > 0.

    Args:
        V (array-like): Array of voltage values.
        I (array-like): Array of current values.

    Returns:
        V_filtered (numpy.ndarray): Filtered voltage values where I > 0.
        I_filtered (numpy.ndarray): Filtered current values where I > 0.
    """
    # Convert inputs to numpy arrays for easier manipulation
    V = np.array(V)
    I = np.array(I)
    
    # Check if V and I have the same length
    if len(V) != len(I):
        raise ValueError("V and I must have the same length.")
    
    # Find indices where I > 0
    positive_indices = I > 0
    
    # Filter V and I using the positive indices
    V_filtered = V[positive_indices]
    I_filtered = I[positive_indices]
    
    return V_filtered, I_filtered

def write_data_to_csv(data, filename):
  """
  Writes data from a NumPy array to a CSV file.

  Args:
    data: A NumPy array containing the data.
    filename: The name of the output CSV file.
  """

  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write column headers
    writer.writerow(["V", "I"]) 
    for row in data:
      writer.writerow(row[:2])  # Write only the first two columns

def noisy_data_to_csv(V, I, sigma_I, filename):
    current_noise = np.random.normal(0, sigma_I, len(I))
    In = I + current_noise
    V2, I2 = filter_positive_current(V, In)
    data_new = np.stack((V2, I2), axis=1)
    write_data_to_csv(data_new, filename)    
    
def stats_noise(filename, fit_range_parameter=1.1, sigma_current=1e-9, nsamples=100):
    V, I = extract_voltage_current_pair(filename=filename, pair_index=0)
    VTH, IS, nst = [], [], []
    for n in range(nsamples):
        noisy_data_to_csv(V, I, sigma_current, filename+'.noise')
        res = Fitter(filename+'.noise', fit_range_parameter=fit_range_parameter, verbose=False, plot_fit=False)
        VTH.append(res.VTH_cf)
        IS.append(res.IS_cf)
        nst.append(res.n_cf)
    VTH = np.array(VTH)
    IS = np.array(IS)
    nst = np.array(nst)
    print(f'VTH: mean {VTH.mean()}; std {VTH.std()}')
    print(f'IS: mean {IS.mean()}; std {IS.std()}')
    print(f'n: mean {nst.mean()}; std {nst.std()}')
    return VTH, IS, nst    
