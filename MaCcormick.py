import numpy as np
from os import curdir
from functools import partial

myname = 'TAO'
desired_directory = curdir + "/data/"

# Initialization

x_left = -1  # left end of computing area
x_right = 1  # right end of computing area
n_list = [5001]  # #grid points
t_list = [10, 20, 30, 40]

final_time = 40
cfl_list = [0.9]  # CFL number
max_wavespeed = 2

# Output data type

state_dtype = [
    ('x_i', np.float64),
    ('u_i', np.float64)
]

zero_dtype = [
    ('t_i', np.float64),
    ('u_i', np.float64)
]


# Fluxes
def constant_flux(x, num_solu, *args):
    if "maxspeed" in args:
        return 1.0
    else:
        return num_solu


def cosinusoidal_flux(x, num_solu, *args):
    if "maxspeed" in args:
        return 1.9
    else:
        return np.where(abs(x) <= 0.25, 1 + 0.9 * np.cos(2 * np.pi * x), 1) * num_solu


fluxes = {
    "CONSTANT": constant_flux,
    "COSINUSOIDAL": cosinusoidal_flux
}


# Output functions

def ostream_state_file(M, h, t, init, method, num_solu, cfl, flux_name):
    temp_matrix = np.zeros((M,), dtype=state_dtype)
    temp_matrix['x_i'] = np.arange(x_left, x_right + h / 2, h)
    temp_matrix['u_i'] = (np.append(num_solu, num_solu[0])).T
    np.savetxt('{directory}{myname}_{init}_{fluxfunc}_{cfl}_{method}_{M}_{t}_STATE.csv'.format(
        directory=desired_directory,
        myname=myname,
        init=init,
        fluxfunc=flux_name,
        cfl=cfl,
        method=method,
        M=M,
        t=round(t)
    ), temp_matrix, fmt="%1.15e", delimiter=',')


def ostream_uzero_file(M, uzero, init, method, cfl, flux_name):
    np.savetxt('{directory}{myname}_{init}_{fluxfunc}_{cfl}_{method}_{M}_uzero.csv'.format(
        directory=desired_directory,
        myname=myname,
        init=init,
        fluxfunc=flux_name,
        cfl=cfl,
        method=method,
        M=M
    ), uzero, fmt="%1.15e", delimiter=',')


# Initial values

def gaussian_pulse(x):
    return np.exp(-50 * x ** 2)


initials = {
    "GAUSSIAN_PULSE": gaussian_pulse
}


# Computing module

def Mac22_FB(u, h, dt, flux_func):
    x = np.arange(x_left, x_right, h)
    f = flux_func(x, u)

    u_hat = u + dt / h * (np.roll(f, -1) - f)

    f = flux_func(x, u_hat)

    return (u_hat + u + dt / h * (f - np.roll(f, 1))) / 2


def Mac22_BF(u, h, dt, flux_func):
    x = np.arange(x_left, x_right, h)
    f = flux_func(x, u)

    u_hat = u + dt / h * (f - np.roll(f, 1))

    f = flux_func(x, u_hat)

    return (u_hat + u + dt / h * (np.roll(f, -1) - f)) / 2


def Mac24_FB(u, h, dt, flux_func):
    x = np.arange(x_left, x_right, h)

    f = flux_func(x, u)
    u_hat = u + dt / h / 6 * (-np.roll(f, -2) + 8 * np.roll(f, -1) - 7 * f)

    f = flux_func(x, u_hat)
    return (u_hat + u + dt / h / 6 * (7 * f - 8 * np.roll(f, 1) + np.roll(f, 2))) / 2


def Mac24_BF(u, h, dt, flux_func):
    x = np.arange(x_left, x_right, h)

    f = flux_func(x, u)
    u_hat = u + dt / h / 6 * (7 * f - 8 * np.roll(f, 1) + np.roll(f, 2))

    f = flux_func(x, u_hat)
    return (u_hat + u + dt / h / 6 * (-7 * f + 8 * np.roll(f, -1) - np.roll(f, -2))) / 2


# Alternating method

def alternating_method(u_init, N, h, cfl, method, init, flux_func, flux_name, evolution1, evolution2):
    max_wavespeed = flux_func(1, 1, "maxspeed")  # Reset wavespeed
    print(max_wavespeed)

    max_dt = cfl * h / max_wavespeed
    num_solu = np.array(u_init, copy=True)

    t_zero = np.array([])
    u_zero = np.array([])
    nzero = round((N - 1) / 2)

    t = 0.0
    while (t < final_time):
        dt = min(max_dt, final_time - t)
        t_zero = np.append(t_zero, t)
        u_zero = np.append(u_zero, num_solu[nzero])
        num_solu = evolution1(num_solu, h, dt, flux_func)
        t += dt
        print(t)

        for tt in t_list:
            if abs(t - tt) < dt / 2:
                ostream_state_file(N, h, t, init, method, num_solu, cfl, flux_name)

        dt = min(max_dt, final_time - t)
        t_zero = np.append(t_zero, t)
        u_zero = np.append(u_zero, num_solu[nzero])
        num_solu = evolution2(num_solu, h, dt, flux_func)
        t += dt
        print(t)

        for tt in t_list:
            if abs(t - tt) < dt / 2:
                ostream_state_file(N, h, t, init, method, num_solu, cfl, flux_name)

    uzero = np.full((len(t_zero),), np.nan, dtype=zero_dtype)
    uzero['t_i'] = t_zero
    uzero['u_i'] = u_zero
    ostream_uzero_file(N, uzero, init, method, cfl, flux_name)


methods = {
    "MacCormack22": partial(alternating_method, evolution1=Mac22_BF, evolution2=Mac22_FB),
    #    "MacCormack24": partial(alternating_method, evolution1=Mac24_BF, evolution2=Mac24_FB)
}  # method in methods is actually the function "alternating_method" specifying some parameters


def run_method(N, method, init, flux_name, flux, cfl):
    h = (x_right - x_left) / (N - 1)
    mesh = np.arange(x_left, x_right, h)

    u_init = initials[init](mesh)

    methods[method](u_init, N, h, cfl, method, init, fluxes[flux], flux_name)


def all_data_preject01():
    for N in n_list:
        for cfl in cfl_list:
            for method in methods:
                for flux in fluxes:
                    for initial in initials:
                        flux_name = str(flux)
                        run_method(N, method, initial, flux, flux_name, cfl)


if __name__ == "__main__":
    all_data_preject01()
