#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:45:49 2025

                    **TEST_MOS_EXTRACT_EKV.PY**

@author: Michele Dei (michele.dei@unipi.it)

Script for producing plots for the manuscript: 
    "Heuristic Enz-Krummenacher-Vittoz (EKV) Model Fitting for Low-Power 
    IC Design: An Open-Source Implementation"
    submitted to Electronics (MDPI)
    
    It uses the mos_extract_ekv.py library.
    
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
import matplotlib.pyplot as plt
import mos_extract_ekv as ekv
import os
import pandas as pd

## FUNCTIONS
def inhomgeneous_list_to_np_array(lst, sort=True):
    l = 1
    for item in lst:
        if isinstance(item, np.ndarray):
            l0 = len(item)
            if l0 > l:
                l = l0
    arr = np.zeros((len(lst), l))
    for i, item in enumerate(lst):
        if isinstance(item, np.ndarray):
            arr[i,:len(item)] = item
        else:
            arr[i,0] = item  
    if sort:
        arr = np.sort(arr, axis=1)
    return arr

plt.close('all')
directory_path = "./"
file = "test07_40_33.csv", "test02_400_330.csv", "test01_4000_3300.csv"
labels = 'W/L = 0.4 um/0.33 um', 'W/L = 4 um/3.3 um', 'W/L = 40 um/33 um'
filename = [os.path.join(directory_path, f) for f in file]
figsize = (4.8, 3.6)
fit_range_pararmeter = [1.001, 1.1, 1.6]

if True:
    # Demonstrate heuristic iterative subranging for the test cases/different fitting ranges
    collection_his = []
    for frp in fit_range_pararmeter:
        col = ekv.FitterCollection()
        fits = [ekv.Fitter(f, plot_fit=False, fit_range_parameter=frp) for f in filename]
        for i, f in enumerate(fits):
            col.add_fitter(fitter=f, label=labels[i])
        col.synoptic_plot(plot_title=f'Herustic Iterative Subranging: fit_range_parameter={frp}', figsize=(5.8, 4.4))
        collection_his.append(col)

if True:
    # fit_range_parameter sweep for 40/33 nm/nm case
    fit_range_pararmeter_values = np.arange(1.0, 2, 0.025)
    IS, VTH, n, VL, VH, IL, IH = [], [], [], [], [], [], []
    for frp in fit_range_pararmeter_values:
        fitter_frp = ekv.Fitter(file[0], plot_fit=False, fit_range_parameter=frp)
        IS.append(fitter_frp.IS_cf)
        VTH.append(fitter_frp.VTH_cf)
        n.append(fitter_frp.n_cf)
        VL.append(fitter_frp.VL)
        VH.append(fitter_frp.VH)
        IL.append(fitter_frp.IL)
        IH.append(fitter_frp.IH)
        
    IS = np.array(IS)
    VTH = np.array(VTH)
    n = np.array(n)
    VL = inhomgeneous_list_to_np_array(VL)
    VH = inhomgeneous_list_to_np_array(VH)
    IL = inhomgeneous_list_to_np_array(IL)
    IH = inhomgeneous_list_to_np_array(IH)

    a1 = np.ones_like(fit_range_pararmeter_values)
    
    fitter_logs = ekv.Fitter(file[0], plot_fit=False, log_scaled=True, max_iter=1)
    VTH_logs = fitter_logs.VTH_cf*a1
    IS_logs = fitter_logs.IS_cf*a1
    n_logs = fitter_logs.n_cf*a1
    
    fitter_ecp = ekv.FitterECP(file[0], plot_fit=False)
    VTH_ecp = fitter_ecp.VTH_cf*a1
    IS_ecp = fitter_ecp.IS_cf*a1
    n_ecp = fitter_ecp.n_cf*a1
    
    fig20 = plt.figure(figsize=(4.8*3, 3.6))
    ax20_1 = fig20.add_subplot(131)
    ax20_2 = fig20.add_subplot(132)
    ax20_3 = fig20.add_subplot(133)
    
    # VTH, VL, VH vd fit_range_parameter        
    for i in range(VH.shape[1]):
        mask = (VH[:,i]>0)&(VH[:,i]>VL[:,i])&(VH[:,i]>VTH[i])&(VH[:,i]>0.75)
        if i == 0:
            ax20_1.plot(fit_range_pararmeter_values[mask], VH[:,i][mask], 'x', markersize=4, color='g', label='$V_H$')
        else:
            ax20_1.plot(fit_range_pararmeter_values[mask], VH[:,i][mask], 'x', markersize=4, color='g')
    
    ax20_1.plot(fit_range_pararmeter_values, VTH, 'k', label='$V_{TH}$ (Heuristic Iterative Subranging', lw=2.5)
    ax20_1.plot(fit_range_pararmeter_values, VTH_logs, 'k:', label='$V_{TH}$ (Log-scaling)', lw=1.5)
    ax20_1.plot(fit_range_pararmeter_values, VTH_ecp, 'k-.', label='$V_{TH}$ (Enz-Chicco-Pezzotta)', lw=1.5)
    
    for i in range(VL.shape[1]):
        mask = (VL[:,i]>0)&(VL[:,i]<VH[:,i])&(VL[:,i]<0.75)
        if i == 0:
            ax20_1.plot(fit_range_pararmeter_values[mask], VL[:,i][mask], '+', markersize=4, color='c', label='$V_L$')
        else:
            ax20_1.plot(fit_range_pararmeter_values[mask], VL[:,i][mask], '+', markersize=4, color='c')
            
    ax20_1.grid(linestyle=':')
    ax20_1.legend(fontsize=8)
    ax20_1.set_xlabel('fit_range_parameter', fontsize=12)
    ax20_1.set_ylabel('$V_{TH}, V_H, V_L$ [V]', fontsize=12)
    
    # IS
    ax20_2.plot(fit_range_pararmeter_values, IS*1e6, 'k', label='$I_{S}$ (Heuristic Iterative Subranging)', lw=2.5)
    ax20_2.plot(fit_range_pararmeter_values, IS_logs*1e6, 'k:', label='$I_S$ (Log-scaling)', lw=1.5)
    ax20_2.plot(fit_range_pararmeter_values, VTH_ecp, 'k-.', label='$I_S$ (Enz-Chicco-Pezzotta)', lw=1.5)
    ax20_2.grid(linestyle=':')
    ax20_2.legend(fontsize=8)
    ax20_2.set_xlabel('fit_range_parameter', fontsize=12)
    ax20_2.set_ylabel('$I_{S}$ [$\mu$A]', fontsize=12)

    # n
    ax20_3.plot(fit_range_pararmeter_values, n, 'k', label='$n$ (Heuristic Iterative Subranging)', lw=2.5)
    ax20_3.plot(fit_range_pararmeter_values, n_logs, 'k:', label='$n$ (Log-scaling)', lw=1.5)
    ax20_3.plot(fit_range_pararmeter_values, n_ecp, 'k-.', label='$n$ (Enz-Chicco-Pezzotta)', lw=1.5)
    ax20_3.grid(linestyle=':')
    ax20_3.legend(fontsize=8)
    ax20_3.set_xlabel('fit_range_parameter', fontsize=12)
    ax20_3.set_ylabel('$n$ [-]', fontsize=12)
    
    plt.tight_layout()
    
if True:
    # Log-scaled fitting
    fits_lsf = [ekv.Fitter(f, plot_fit=False, log_scaled=True, max_iter=1) for f in filename]
    collection_lsf = ekv.FitterCollection()
    for i, f in enumerate(fits_lsf):
        collection_lsf.add_fitter(fitter=f, label=labels[i])
    collection_lsf.synoptic_plot(plot_title='Log-Scaled fitting', figsize=figsize)
    
    # Enz Chicco Pezzotta fitting
    fits_ecp = [ekv.FitterECP(f, plot_fit=False) for f in filename]
    collection_ecp = ekv.FitterCollection()
    for i, f in enumerate(fits_ecp):
        collection_ecp.add_fitter(fitter=f, label=labels[i])
    collection_ecp.synoptic_plot(plot_title='Enz-Chicco-Pezzota fitting', figsize=figsize)
        
    # Fetch relative errors from collections, for the 40/33 case, fit_range_pararmeter = 1.1
    i = 0
    V = collection_his[1].fitters[0].VG
    err1 = collection_his[1].fitters[0].rel_err
    err2 = collection_lsf.fitters[0].rel_err
    err3 = collection_ecp.fitters[0].rel_err
    VH = collection_his[1].fitters[0].VH
    VL = collection_his[1].fitters[0].VL
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    # Error reference line
    ax1.semilogy(V, 0.05*np.ones_like(err1), 'g-', lw=2, label='5% error threshold', alpha=0.8)
    
    # Log-scaled
    ax1.semilogy(V, err2, label='Log-scaled fitting',  
                 color='orange', marker='o', markersize=5, linestyle='-',
                 markeredgecolor='orange', markerfacecolor='white')
    
    # ECP
    ax1.semilogy(V, err3, label='Enz-Chicco-Pezzotta fitting',  
                 color='blueviolet', marker='s', markersize=5, linestyle='-',
                 markeredgecolor='blueviolet', markerfacecolor='white')
    
    # Iterative subranging
    ax1.semilogy(V, err1, label='Iterative subranging\nfit_range_parameter=1.1',  
                 color='k', marker='o', markersize=5, linestyle='-',
                 markeredgecolor='k', markerfacecolor='k')
    
    ax1.set_xlabel("$V_G$ [V]")
    ax1.set_ylabel("$|I_{D,\mathrm{meas}}-I_{D,\mathrm{model}}|/I_{D,\mathrm{meas}}$ [A]")
    ax1.set_xlim([min(V), max(V)])
    ax1.axvspan(VL, VH, facecolor='green', alpha=0.2)
    ax1.legend(fontsize=8)
    ax1.grid(which="both", linestyle=':')

if True:
    # Parameter extraction for example design case (wide swing cascode mirror)
    # Use Fitter to extract the EKV parameters for MOSFET in test20.cir and test21.cir
    file_wscm = "test20.csv", "test21.csv"
    filename_wscm = [os.path.join(directory_path, f) for f in file_wscm]
    frp_wscm = 1.65 # adjusted to current range 1-40 uA
    fits_wscm = [ekv.Fitter(f, fit_range_parameter=frp_wscm, plot_fit=False) for f in filename_wscm]
    
    print('|'*80)
    print('DESIGN CASE: WIDE-SWING CASCODE MIRROR')
    print('|'*80)
    print('sky130_fd_pr__nfet_01v8__model.31 l=1.0u w=2u')
    print(f'fit_range_parameter: {frp_wscm }')
    print(f'VTH={fits_wscm[0].VTH_cf*1e3} mV')
    print(f'VH={fits_wscm[0].VH} V')
    print(f'VL={fits_wscm[0].VL} V')
    print(f'IH={fits_wscm[0].IH*1e6} uA')
    print(f'IL={fits_wscm[0].IL*1e6} uA')
    print(f'IS={fits_wscm[0].IS_cf*1e6} uA')
    print(f'n={fits_wscm[0].n_cf}')
    print('|'*80)
    print('sky130_fd_pr__nfet_01v8__model.32 l=0.5u w=2u')
    print(f'fit_range_parameter: {frp_wscm }')
    print(f'VTH={fits_wscm[1].VTH_cf*1e3} mV')
    print(f'VH={fits_wscm[1].VH} V')
    print(f'VL={fits_wscm[1].VL} V')
    print(f'IH={fits_wscm[1].IH*1e6} uA')
    print(f'IL={fits_wscm[1].IL*1e6} uA')
    print(f'IS={fits_wscm[1].IS_cf*1e6} uA')
    print(f'n={fits_wscm[1].n_cf}')    
    print('|'*80)

    # SPICE simulation data
    file_wscm_spice = "wscm.txt"
    df = pd.read_csv(file_wscm_spice, delim_whitespace=True, header=None)
    iin = df.iloc[:, 0]
    vg1 = df.iloc[:, 1]
    vs3 = df.iloc[:, 3]
    
    # analytical model (strong inversion approximation)
    n1 = fits_wscm[0].n_cf
    vth1 = fits_wscm[0].VTH_cf
    is1 = fits_wscm[0].IS_cf
    n3 = fits_wscm[1].n_cf
    vth3 = fits_wscm[1].VTH_cf
    is3 = fits_wscm[1].IS_cf
    ut = ekv.calculate_UT()
    vp1_eq = 2*ut*np.sqrt(iin/is1)
    vg1_eq = vth1 + n1*vp1_eq
    vc = 1.3 # from wscm.cir
    vp3_eq = (vc-vth3)/n3
    vs3_eq = 1/n3*(vp3_eq - n3/n1*np.sqrt(is1/is3)*vp1_eq)
    print(f'VC = {vc} V')
    print(f'VC1 = {vth3} V')
    print(f'VC2 = {vth3+n3**2/(1+n3)*vth1} V')
    print('|'*80)
    
    fig_de = plt.figure()
    ax_de = fig_de.add_subplot(111)
    ax_de.plot(iin*1e6, vg1, 'g--', lw=1.5, label='$V_{G1}$ SPICE')
    ax_de.plot(iin*1e6, vg1_eq, 'g', lw=2, label='$V_{G1}$ equation')
    ax_de.plot(iin*1e6, vs3, 'b--', lw=1.5, label='$V_{S3}$ SPICE')
    ax_de.plot(iin*1e6, vs3_eq, 'b', lw=2, label='$V_{S3}$ equation')
    ax_de.set_xlim([0, 60])
    ax_de.set_ylim([0.2, 1.2])
    ax_de.set_xlabel('$I_{in}$ [$\mu$A]', fontsize=12)
    ax_de.set_ylabel('$V_{G1}, V_{S3}$ [V]', fontsize=12)
    ax_de.legend(fontsize=10, ncols=2)
    ax_de.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig('wscm_cmp.pdf')

if True:
    # Sensitivity to noise: statistics for fits for different values of sigma (noise)
    sigmas = ekv.logspace_octaves(0.1e-9, num_octaves=5, points_per_octave=3)
    VTH_m, VTH_s = [], []
    IS_m, IS_s = [], []
    n_m, n_s = [], []
    for sigma_current in sigmas:
        print(sigma_current)
        VTH, IS, n = ekv.stats_noise(filename[0], fit_range_parameter=1.1, sigma_current=sigma_current, nsamples=1000)
        VTH_m.append(VTH.mean())
        VTH_s.append(VTH.std())
        IS_m.append(IS.mean())
        IS_s.append(IS.std())
        n_m.append(n.mean())
        n_s.append(n.std())
        
    VTH_m = np.array(VTH_m)
    VTH_s = np.array(VTH_s)
    IS_m = np.array(IS_m)
    IS_s = np.array(IS_s)
    n_m = np.array(n_m)
    n_s = np.array(n_s)
    #
    fig_n = plt.figure(figsize=(4.8*3, 3.6))
    #
    ax1 = fig_n.add_subplot(131)
    ax1.errorbar(sigmas, VTH_m*1e3, yerr=VTH_s*1e3, label='$V_{TH}$',  
                 color='orange', marker='o', markersize=5, linestyle='dotted',
                 markeredgecolor='k', markerfacecolor='k')
    ax1.set_xscale("log")
    ax1.set_xlabel("Current noise standard deviation [A]")
    ax1.set_ylabel("$V_{TH}$ [mV]")
    ax1.grid(which="both", linestyle=':')
    #
    ax2 = fig_n.add_subplot(132)
    ax2.errorbar(sigmas, IS_m*1e6, yerr=IS_s*1e6, label='$I_{S}$',  
                 color='orange', marker='o', markersize=5, linestyle='dotted',
                 markeredgecolor='k', markerfacecolor='k')
    ax2.set_xscale("log")
    ax2.set_ylim([0, 3])
    ax2.set_xlabel("Current noise standard deviation [A]")
    ax2.set_ylabel("$I_{S}$ [$\mu$A]")
    ax2.grid(which='both', linestyle=':')
    #
    ax3 = fig_n.add_subplot(133)
    ax3.errorbar(sigmas, n_m, yerr=n_s, label='$n$',  
                 color='orange', marker='o', markersize=5, linestyle='dotted',
                 markeredgecolor='k', markerfacecolor='k')
    ax3.set_xscale("log")
    ax3.set_xlabel("Current noise standard deviation [A]")
    ax3.set_ylabel("$n$ [-]")   
    ax3.grid(which='both', linestyle=':')
    #
    plt.tight_layout()
 
if True:
    # Sensitivity to noise: spot simulation
    sigma_current = 1e-9
    V, I = ekv.extract_voltage_current_pair(filename=filename[0], pair_index=0)
    current_noise = np.random.normal(0, sigma_current, len(I))
    In = I + current_noise
    V2, I2 = ekv.filter_positive_current(V, In)
    data_new = np.stack((V2, I2), axis=1)
    ekv.write_data_to_csv(data_new, "output_noise.csv")
    
    figsize=(4.6, 3.2)
    fnn_irs = ekv.Fitter("output_noise.csv", fit_range_parameter=1.1, figsize=figsize)
    fnn_lgs = ekv.Fitter("output_noise.csv", log_scaled=True, max_iter=1, plot_title='Log scaled fitting', figsize=figsize)
    try:
        fnn_ecp = ekv.FitterECP("output_noise.csv", plot_title='Enz-Chicco-Pezzotta fitting', figsize=figsize)
    except:
        print('ECP fitting unsuccesful')
