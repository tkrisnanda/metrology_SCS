from numpy import sum
from numpy.random import choice
import numpy as np
from numpy import ma as ma
from qutip import *
import h5py
from types import SimpleNamespace
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

COH_MAP = {
    '0.5': 0.258,
    '0.75': 0.4,
    '1': 0.558,
    '1.25': 0.7321,
    '1.5': 0.921,
    '1.6': 1,
    '1.75': 1.122,
    '2': 1.327,
    '2.1': 1.409,
    '2.2': 1.49,
    '2.3': 1.571,
    '2.4': 1.651,
    '2.5': 1.73,
}

N1_ALPHA_MAP = {
    '0.25': 2.1,
    '0.3': 1.95,
    '0.4': 1.745,
    '0.5': 1.6,
    '0.6': 1.49,
    '0.7': 1.39,
    '0.75': 1.345,
    '0.8': 1.3
}


def name_scs(alpha, alpha_coeff=0.5, omit05=True):
    # txt_ket(0) + '+' + txt_ket(fr'\alpha={alpha}')
    if (str(alpha_coeff) == '0.5') and omit05:
        return f'SCS({alpha})'
    return r'SCS$_{'+str(alpha_coeff)+r'}('+str(alpha)+r')$'


def compute_photon_number(alpha):
    cdim = 30
    psi = (coherent(cdim, 0) + coherent(cdim, alpha)).unit()
    a = destroy(cdim)
    # return np.abs(np.array(psi.dag() * a.dag()*a * psi)[0][0]): not sure why we use [0][0] before
    return np.abs(np.array(psi.dag() * a.dag()*a * psi))


def compute_FI(y, dx):
    dy = np.gradient(y, dx)
    return (dy**2 / (y) + dy**2 / (1-y)) # 这里简化了公式,对于binary measurement， dy是一样的


def compute_FI_4p(y1, y2, y3, y4, dx):
    dy1 = np.gradient(y1, dx) # pg_vac
    dy2 = np.gradient(y2, dx) # pe_vac
    dy3 = np.gradient(y3, dx) # pg_notvac
    dy4 = np.gradient(y4, dx) # pe_notvac
    Fg_vac = (dy1**2/(y1) + dy1**2/(1-y1)) 
    Fe_vac = (dy2**2/(y2) + dy2**2/(1-y2)) 
    Fg_notvac = (dy3**2/(y3) + dy3**3/(1-y3)) 
    Fe_notvac = (dy4**2/(y4) + dy4**2/(1-y4)) 
    F_total = (dy1**2/y1 + dy2**2/y2 + dy3**2/y3 +  dy4**2/y4)
    return   Fg_vac, Fe_vac, Fg_notvac, Fe_notvac, F_total


def print_FI(x, FI, SQL, label):
    idx = FI > SQL
    print(f'{label} max:  {np.round(np.max(FI),3)} --> {np.round(np.max(FI) / SQL,3)}')

    if (np.sum(idx) > 0):
        value = FI[idx]
        # FI_data[idx]
        print(
            f'{label} average: {np.round(np.mean(value),3)} --> {np.round(np.mean(value) / SQL,3)}')
        print(f'{label} bandwitdth: {np.round(x[idx][-1] - x[idx][0],3)}')


def load_experimental_data(path, threshold, pre_selection=True, flip=None, amplitude=False):
    file = h5py.File(path, 'r')
    data = file["data"]
    x = np.array(data["x"][:, 1])  # np.arange(4,100,1)
    # print('exp_time:', x)
    chi = 1.36e-3 * 2 * np.pi
    phase = x
    if not amplitude:
        phase = (4*x) * chi / 2
    data_i = data["I"][:]

    if (flip == None):
        flip = bool(threshold < 0)
    ss_data = np.where(data_i < threshold if (
        not flip) else data_i > threshold, 1, 0)

    if pre_selection:
        m0 = ss_data[:, 0::3]  # .mean(axis=0)  # .mean(axis=0)
        m1 = ss_data[:, 1::3]  # .mean(axis=0)
        m2 = ss_data[:, 2::3]

        # keep the second rr data with initial qubit in g
        m1_g = ma.masked_array(m1, mask=m0)
        # keep the third rr data with initial qubit in g
        m2_g = ma.masked_array(m2, mask=m0)
        # projection on vac given by g
        mx_g = ma.masked_array(m2_g, mask=m1_g)

        "export the data for Bayesian estimation"
        mx_e_test = 1 - m1_g
        output =  np.full_like(m1_g, np.nan, dtype=float)
        output[(mx_g==1) & (mx_e_test==1)] = 1
        output[(mx_g==0) | (mx_e_test==0)] = 0
        output[np.isnan(mx_g) | np.isnan(mx_e_test)] = np.nan
        mg0 = output

        "compute the FI through binary measurement p1, p2"

        mx_g_avg = mx_g.mean(axis=0)  
        # m1_g.mean(0) is the probability of pg gotten from second rr
        mx_g_avg = (1-m1_g.mean(0)) * mx_g_avg #解释：这里m1_g.mean(0)表示比特在｜e>的概率



        "compute the FI through four p, i.e., pg0,pe0,pgnot0,penot0"
        mx_e = ma.masked_array(m2_g, mask=np.logical_not(m1_g)) # projection on vac given by e
        # mx_e_not_vac = ma.masked_array(m2_g, mask=np.logical_not(m1_g)) # projection on not vac giveb by e

        # below four p is used to compute the FI_4p
        mx_g_vac = (1-m1_g.mean(0)) * mx_g.mean(0)
        mx_e_vac  = m1_g.mean(0) * mx_e.mean(0)
        mx_g_notvac = (1-m1_g.mean(0)) * (1 - mx_g.mean(0))
        mx_e_notvac = m1_g.mean(0) * (1- mx_e.mean(0)) 

    else:
        m1 = ss_data[:, 0::2]  # .mean(axis=0)  # .mean(axis=0)
        m1_g = m1
        m2 = ss_data[:, 1::2]  # .mean(axis=0)

        # Post selection Xiaozhou
        mx_g = ma.masked_array(m2, mask=m1)
        # mx_e = ma.masked_array(m2, mask=np.logical_not(m1))
        mx_g_avg = mx_g.mean(axis=0)  # 1-m1.mean(0)

        # Tanjung rescale
        mx_g_avg = (1-m1.mean(0)) * mx_g_avg

    result = {
        'y': mx_g_avg,
        'x': phase,
        'm1': m1_g,
        'y_raw': mx_g, 
        'mx_e_test': mx_e_test,
        'mx_e': mx_e,
        'mg0': mg0,
        
        'pg_vac': mx_g_vac, # pg_vac is same as y, here I want to make the notation be consistent, so rename it.
        'pe_vac': mx_e_vac,
        'pg_notvac': mx_g_notvac,
        'pe_notvac': mx_e_notvac, 
    }
    return SimpleNamespace(**result)




def extract_FI_params(x, y, SQL):
    x = np.array(x, dtype='float')
    y = np.array(y)

    FI = compute_FI(y, x[1] - x[0])
    x = x[FI < np.inf][1::]
    FI = FI[FI < np.inf][1::]

    # Extract parameters
    result_max = np.max(FI)
    idx = (FI > SQL) & (x >= 0)
    if (np.sum(idx) > 0):
        value = FI[idx]
        result_avg = np.mean(value)
        result_bandwidth = x[idx][-1] - x[idx][0]
        max_x = x[np.argmax(FI)]
    else:
        result_avg = 0
        result_bandwidth = 0
        max_x = x[np.argmax(FI)]
    return result_max, result_avg, result_bandwidth, max_x


def extract_FI_params_4p(x, y1, y2, y3, y4, SQL):
    x = np.array(x, dtype='float')
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)

    # FI = compute_FI(y, x[1] - x[0])
    _, _, _, _, FI = compute_FI_4p(y1, y2, y3, y4, x[1] - x[0])
    x = x[FI < np.inf][1::]
    FI = FI[FI < np.inf][1::]

    # Extract parameters
    if 0:
        result_max = np.max(FI)
    if 1:
        result_max = np.max(FI[: int(len(x)/2) ] ) # add the [:len(x/2)] is to avoid the sigular point happened at the end of data
    
    idx = (FI > SQL) & (x >= 0)
    
    if (np.sum(idx) > 0):
        value = FI[idx]
        result_avg = np.mean(value)
        result_bandwidth = x[idx][-1] - x[idx][0]
        max_x = x[np.argmax(FI)]
    else:
        result_avg = 0
        result_bandwidth = 0
        max_x = x[np.argmax(FI)]
    return result_max, result_avg, result_bandwidth, max_x


def txt_sqrt(num):
    return '$\\sqrt{' + str(num) + '}$'


def txt_ket(a):
    return '$|' + str(a) + '\\rangle$'


def estimate_delta_theta(data, y_fit, x_fit, xz_idx=False):
    dx = x_fit[1]-x_fit[0]
    FI = compute_FI(y_fit, dx)

    # Get the optimal point from fitting data
    idx = np.argmax(FI)
    # Index for experimental data
    idx_data = np.argmin(np.abs(data.x - x_fit[idx]))
    # Fit index closest to closest experimental point
    opt_slope_idx = np.argmin(np.abs(x_fit - data.x[idx_data]))
    opt_slope = np.gradient(y_fit, dx)[opt_slope_idx]

    # Get optimal data
    m1 = data.m1
    mx_g = data.y_raw
    opt_p_g = 1 - m1[:, idx_data]
    opt_p_gvac = mx_g[:, idx_data]
 
    # Find variance
    cov1 = np.cov(opt_p_g, opt_p_gvac)[0, 1]
    cov2 = np.cov(opt_p_g**2, opt_p_gvac**2)[0, 1]

    # # dependent variance 
    # var_p = (
    #     cov2 + np.mean(opt_p_g**2)*np.mean(opt_p_gvac**2) -
    #     (cov1+np.mean(opt_p_g)*np.mean(opt_p_gvac))**2
    # )

    ## independent variance
    var_direct = np.var(opt_p_g * opt_p_gvac)
    var_p = var_direct

    return var_p / opt_slope**2, var_p, opt_slope, idx_data # idx_data addded to export the data for Bayesian estimation





def make_figure(width=9, height=4, font_size=8):
    cm = 1 / 2.54  # centimeters in inches
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=(width * cm, height * cm))
    return fig


def bootstrap(arr: np.array, n_boot: int = 1000) -> np.array:
    length = len(arr)
    # Mean of the observed data
    mu_obs = np.mean(arr)
    # Mean of the boostrapped samples
    mu_boot = np.zeros(n_boot)
    for i in range(n_boot):
        # Choose random indexes and sample arr
        idx = choice(length, size=length, replace=True)
        samples = arr[idx]
        # Estimate mean
        mu_boot[i] = np.mean(samples)
    # Account for std estimation bias by shifting the distribution to the observed mean
    mu = np.mean(mu_boot)
    mu_boot += mu_obs - mu
    # Estimate std
    sigma = np.std(mu_boot)
    return mu, sigma, mu_boot


# 这个区别在于用来计算 error propragation
def bootstrap_dp(arr: np.array, n_boot: int = 1000) -> np.array:
    length = len(arr)
    # Mean of the observed data
    mu_obs = np.mean(arr)
    # Mean of the boostrapped samples
    mu_boot = np.zeros(n_boot)
    mu_boot_dp = np.zeros(n_boot)
    for i in range(n_boot):
        # Choose random indexes and sample arr
        idx = choice(length, size=length, replace=True)
        samples = arr[idx]
        # Estimate mean
        mu_boot[i] = np.mean(samples)
        mu_boot_dp[i] = np.var(samples)
    # Account for std estimation bias by shifting the distribution to the observed mean
    mu = np.mean(mu_boot)
    mu_boot += mu_obs - mu
    # Estimate std
    sigma = np.std(mu_boot)
    return mu, sigma, mu_boot, mu_boot_dp


def compute_many_FI_error(data, x_fit, fit_order, up_lim, n_boot=100):
    x_len = len(data.x)

    mu_boot_g = np.zeros([n_boot, x_len])
    mu_boot_gvac = np.zeros([n_boot, x_len])
    for i in tqdm(range(x_len)):
        # First measurement, g
        _, _, mu_boot_g[:, i] = bootstrap(data.m1[:, i], n_boot)

        # Second measurement, g vac
        _, _, mu_boot_gvac[:, i] = bootstrap(data.y_raw[:, i], n_boot)

    # Compute Fisher information
    idx = data.x < up_lim
    x = x_fit
    FIs = np.zeros([n_boot, len(x)])

    for i in range(n_boot):
        mu = (1-mu_boot_g[i, :]) * mu_boot_gvac[i, :]

        data_fit = np.polyfit(data.x[idx], mu[idx], fit_order)

        # Fisher information
        y_fit = np.poly1d(data_fit)(x)
        FIs[i, :] = compute_FI(y_fit, x[1] - x[0])



    # Observed FI
    data_fit = np.polyfit(data.x[idx], data.y[idx], fit_order)
    y_fit = np.poly1d(data_fit)(x)
    FI_data = compute_FI(y_fit, x[1] - x[0])

    # Account for std estimation bias by shifting the distribution to the observed mean
    FI_mu = np.mean(FIs, axis=0)
    FIs += FI_data - FI_mu
    FI_sigma = np.std(FIs, axis=0)

    return FI_mu, FI_sigma




def compute_many_FI_error_offset(data, x_fit, fit_order, up_lim, n_boot=100):
    x_len = len(data.x)

    mu_boot_g = np.zeros([n_boot, x_len])
    mu_boot_gvac = np.zeros([n_boot, x_len])
    
    for i in tqdm(range(x_len)):
        # First measurement, g
        _, _, mu_boot_g[:, i] = bootstrap(data.m1[:, i], n_boot)

        # Second measurement, g vac
        _, _, mu_boot_gvac[:, i] = bootstrap(data.y_raw[:, i], n_boot)

    # Compute Fisher information
    idx = data.x < up_lim
    x = x_fit
    FIs = np.zeros([n_boot, len(x)])
    offset_data_array =[]
    for i in range(n_boot):
        
        mu = (1-mu_boot_g[i, :]) * mu_boot_gvac[i, :] # this is pg_vac

        data_fit = np.polyfit(data.x[idx], mu[idx], fit_order)

        # Fisher information
        y_fit = np.poly1d(data_fit)(x)
        FIs[i, :] = compute_FI(y_fit, x[1] - x[0])

        # index = np.argmax(FIs[i,:])
        index = np.argmax(FIs[i, x_fit<1])
        # print(index)
        index_offset = np.argmin(np.abs(data.x-x_fit[index]))
        offset_data_array.append(data.x[index_offset])
    print(offset_data_array)    
    var_offset = np.var(offset_data_array)   
  
    # Observed FI
    data_fit = np.polyfit(data.x[idx], data.y[idx], fit_order)
    y_fit = np.poly1d(data_fit)(x)
    FI_data = compute_FI(y_fit, x[1] - x[0])

    # Account for std estimation bias by shifting the distribution to the observed mean
    FI_mu = np.mean(FIs, axis=0)
    FIs += FI_data - FI_mu
    FI_sigma = np.std(FIs, axis=0)

    return FI_mu, FI_sigma, np.sqrt(var_offset)


def replace_nan_with_adjacent_avg(arr):
    # Iterate through the array
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            # Handle edge cases where NaN is at the start or end of the array
            if i == 0:
                # If NaN is at the start, replace with the next value
                arr[i] = arr[i+1] if not np.isnan(arr[i+1]) else arr[i]
            elif i == len(arr) - 1:
                # If NaN is at the end, replace with the previous value
                arr[i] = arr[i-1] if not np.isnan(arr[i-1]) else arr[i]
            else:
                # Replace NaN with the average of the previous and next values
                prev_val = arr[i-1]
                next_val = arr[i+1]
                
                if not np.isnan(prev_val) and not np.isnan(next_val):
                    arr[i] = (prev_val + next_val) / 2
                elif not np.isnan(prev_val):  # If next_val is NaN, use prev_val
                    arr[i] = prev_val
                elif not np.isnan(next_val):  # If prev_val is NaN, use next_val
                    arr[i] = next_val   
    return arr

def compute_many_FI_error_offset_4p(data, x_fit, fit_order, up_lim, n_boot=100):
    phase = data.x

    x_len = len(data.x)

    mu_boot_g = np.zeros([n_boot, x_len])
    mu_boot_gvac = np.zeros([n_boot, x_len])
    mu_boot_evac = np.zeros([n_boot, x_len])
    
    for i in tqdm(range(x_len)):
        # First measurement, g
        _, _, mu_boot_g[:, i] = bootstrap(data.m1[:, i], n_boot)

        # Second measurement, projection on vac given by g
        _, _, mu_boot_gvac[:, i] = bootstrap(data.y_raw[:, i], n_boot)

        # second measurement, projection on vac given by e
        _, _, mu_boot_evac[:, i] = bootstrap(data.mx_e[:, i], n_boot)

    # Compute Fisher information
    idx = data.x < up_lim
    x = x_fit
    FIs_4p = np.zeros([n_boot, len(x)])
    offset_data_array =[]
    for i in range(n_boot):
        
        mu_gvac = (1-mu_boot_g[i, :]) * mu_boot_gvac[i, :]
        mu_evac = mu_boot_g[i, :] * mu_boot_evac[i, :]
        mu_g_notvac = (1-mu_boot_g[i, :]) * (1 - mu_boot_gvac[i, :])
        mu_e_notvac = mu_boot_g[i, :] * (1 - mu_boot_evac[i, :])

        if 1: # replace the nan value by the adjacent value
                mu_evac = replace_nan_with_adjacent_avg(mu_evac)
                mu_e_notvac =  replace_nan_with_adjacent_avg(mu_e_notvac)

        data_fit_pg_vac = np.polyfit(phase[phase < up_lim], mu_gvac[phase < up_lim], fit_order)
        data_fit_pe_vac = np.polyfit(phase[phase < up_lim], mu_evac[phase < up_lim], fit_order)
        data_fit_pg_notvac = np.polyfit(phase[phase < up_lim], mu_g_notvac[phase < up_lim], fit_order)
        data_fit_pe_notvac = np.polyfit(phase[phase < up_lim], mu_e_notvac[phase < up_lim], fit_order)

        y_data_fit_pg_vac =np.poly1d(data_fit_pg_vac)(x)
        y_data_fit_pe_vac =np.poly1d(data_fit_pe_vac)(x)
        y_data_fit_pg_notvac =np.poly1d(data_fit_pg_notvac)(x)
        y_data_fit_pe_notvac =np.poly1d(data_fit_pe_notvac)(x)
        # print(len(y_data_fit_pe_notvac))

        _, _, _, _, F_total = compute_FI_4p(y_data_fit_pg_vac, y_data_fit_pe_vac, y_data_fit_pg_notvac, y_data_fit_pe_notvac, x[1]-x[0])

        FIs_4p[i, :] = F_total
        
        # index = np.argmax(FIs[i,:])
        index = np.argmax(FIs_4p[i, x_fit<1])
        # print(index)
        index_offset = np.argmin(np.abs(data.x-x_fit[index]))
        offset_data_array.append(data.x[index_offset])

    # # print(offset_data_array)    
    var_offset = np.var(offset_data_array)   
  
    # Observed FI
   
    data_fit1 = np.polyfit(data.x[idx], data.pg_vac[idx], fit_order)
    data_fit2 = np.polyfit(data.x[idx], data.pe_vac[idx], fit_order)
    data_fit3 = np.polyfit(data.x[idx], data.pg_notvac[idx], fit_order)
    data_fit4 = np.polyfit(data.x[idx], data.pe_notvac[idx], fit_order)

    y_fit_1 = np.poly1d(data_fit1)(x)
    y_fit_2 = np.poly1d(data_fit2)(x)
    y_fit_3 = np.poly1d(data_fit3)(x)
    y_fit_4 = np.poly1d(data_fit4)(x)

    _, _, _, _, FI_data = compute_FI_4p(y_fit_1, y_fit_2, y_fit_3, y_fit_4, x[1] - x[0])

    # Account for std estimation bias by shifting the distribution to the observed mean
    FI_mu = np.mean(FIs_4p, axis=0)
    FIs_4p += FI_data - FI_mu
    FI_sigma = np.std(FIs_4p, axis=0)

    return FI_mu, FI_sigma , np.sqrt(var_offset)