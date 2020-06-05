# ADI

import numpy as np
from os import curdir
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

myname = 'TAO'

import ast

with open('adiset.txt', "r") as data:
    setup = ast.literal_eval(data.read())

setup['dt'] = setup['final_time'] / setup['max_step']


def output_state(num_solu, N, Nx, Ny, d1, d2):
    # output state data, num_soln, at time t and with the grid setting Nx adn Ny
    np.savetxt('{directory}/data/{myname}_dt4_Gaussian_ADI_{Nx}_{Ny}_{d1}_{d2}_{N}_state.csv'.format(
        directory=curdir,
        myname=myname,
        Nx=Nx,
        Ny=Ny,
        d1=d1,
        d2=d2,
        N=N
    ), num_solu, delimiter=',')


def thomas_solver(N, a, b, c, d):
    a = np.array(a, copy=True, dtype=np.float64)
    b = np.array(b, copy=True, dtype=np.float64)
    c = np.array(c, copy=True, dtype=np.float64)
    d = np.array(d, copy=True, dtype=np.float64)
    for i in range(1, N):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.full(N, np.nan, dtype=np.float64)
    x[N - 1] = d[N - 1] / b[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def sinuisoidal_initial(XL, XR, YBOT, YTOP, Nx, Ny):
    initial = np.full((Nx, Ny), 0, dtype=np.float64)
    step_x = (XR - XL) / (Nx - 1)
    step_y = (YBOT - YTOP) / (Ny - 1)
    mesh_x = np.arange(XL, XR + step_x / 2, step_x)
    mesh_y = np.arange(YTOP, YBOT + step_y / 2, step_y)
    if setup['d2'] == 0:
        for i in range(Nx):
            for j in range(Ny):
                initial[j][i] = np.sin(setup['k1'] * np.pi * mesh_x[i])
    else:
        for i in range(Nx):
            for j in range(Ny):
                initial[j][i] = np.sin(setup['k1'] * np.pi * mesh_x[i]) * np.sin(setup['k2'] * np.pi * mesh_y[j])
    return initial

# need changed immediately!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def sinuisoidal_exact(t,XL, XR, YBOT, YTOP, Nx, Ny):
    initial = np.full((Nx, Ny), 0, dtype=np.float64)
    step_x = (XR - XL) / (Nx - 1)
    step_y = (YBOT - YTOP) / (Ny - 1)
    mesh_x = np.arange(XL, XR + step_x / 2, step_x)
    mesh_y = np.arange(YTOP, YBOT + step_y / 2, step_y)
    if setup['d2'] == 0:
        for i in range(Nx):
            for j in range(Ny):
                initial[j][i] = np.sin(setup['k1'] * np.pi * mesh_x[i])
    else:
        for i in range(Nx):
            for j in range(Ny):
                initial[j][i] = np.sin(setup['k1'] * np.pi * mesh_x[i]) * np.sin(setup['k2'] * np.pi * mesh_y[j])
    return initial


def gaussian_initial(XL, XR, YBOT, YTOP, Nx, Ny):
    initial = np.full((Nx, Ny), 0, dtype=np.float64)
    step_x = (XR - XL) / (Nx - 1)
    step_y = (YBOT - YTOP) / (Ny - 1)
    mesh_x = np.arange(XL, XR + step_x / 2, step_x)
    mesh_y = np.arange(YTOP, YBOT + step_y / 2, step_y)
    for i in range(Nx):
        for j in range(Ny):
            initial[j][i] = np.exp(-((mesh_x[i] - 0.5) ** 2 + (mesh_y[j] - 0.5) ** 2) / (2 * 0.02 ** 2)) / (
                    2 * np.pi * 0.02 ** 2)
    return initial


def back_Euler_update(num_soln, Nx, Ny, gamma_x, gamma_y):
    ax = np.full((Nx - 3), -gamma_x, dtype=np.float64)
    bx = np.full((Nx - 2), 1 + 2 * gamma_x, dtype=np.float64)
    cx = np.full((Nx - 3), -gamma_x, dtype=np.float64)
    for k in range(Ny):
        num_soln[k, 1:Nx - 1] = thomas_solver(Nx - 2, ax, bx, cx, num_soln[k, 1:Nx - 1])
        num_soln[k, 0] = 0
        num_soln[k, Nx - 1] = 0


def ADI_dirichlet_update(num_soln, Nx, Ny, gamma_x, gamma_y):
    old_soln = np.array(num_soln, copy=True)  # Will be the UNEW
    old_soln[0, :] = 0
    old_soln[Ny - 1, :] = 0

    half_soln = np.array(num_soln, copy=True)  # UHALF in the projectadi.pdf

    # Specify 3 diagonals of coefficient matrices. They are shorter than Nx&Ny because we are sovling a Dirichlet problem

    ay = np.full((Ny - 3), -gamma_y, dtype=np.float64)
    by = np.full((Ny - 2), 1 + 2 * gamma_y, dtype=np.float64)
    cy = np.full((Ny - 3), -gamma_y, dtype=np.float64)

    ax = np.full((Nx - 3), -gamma_x, dtype=np.float64)
    bx = np.full((Nx - 2), 1 + 2 * gamma_x, dtype=np.float64)
    cx = np.full((Nx - 3), -gamma_x, dtype=np.float64)

    for k in range(1, Ny - 1):
        # Scan each row: explicit in y and implicit in x
        RHS = (1 - 2 * gamma_y) * old_soln[k, 1:Nx - 1] + gamma_y * old_soln[k - 1, 1:Nx - 1] + gamma_y * \
              old_soln[k + 1, 1:Nx - 1]
        half_soln[k, 1:Nx - 1] = thomas_solver(Nx - 2, ax, bx, cx,
                                               RHS)  # In Dirichlet BC, we are only interested in the interior points
        half_soln[k, 0] = 0  # Homogeneous Dirichlet BC
        half_soln[k, Nx - 1] = 0  # Homogeneous Dirichlet BC

    for j in range(1, Nx - 1):
        # Scan each column: explicit in x and implicit in y
        RHS = (1 - 2 * gamma_x) * half_soln[1:Ny - 1, j] + gamma_x * half_soln[1:Ny - 1, j - 1] + gamma_x * \
              half_soln[1:Ny - 1, j + 1]
        num_soln[1:Ny - 1, j] = thomas_solver(Ny - 2, ay, by, cy,
                                              RHS)  # In Dirichlet BC, we are only interested in the interior points
        num_soln[0, j] = num_soln[1, j]  # Homogeneous Dirichlet BC
        num_soln[Ny - 1, j] = num_soln[Ny - 2, j]  # Homogeneous Dirichlet BC

    num_soln[:, 0] = 0
    num_soln[:, Nx - 1] = 0
    num_soln[0, :] = 0
    num_soln[Ny - 1, :] = 0


def ADI_scheme(u_init, Nx, Ny, dx, dy, dt):
    # This functions as driver.m in the projectadi.pdf
    num_soln = np.array(u_init, copy=True)
    t = 0
    dth = dt / 2
    time_step = 0
    count = 0
    for k in range(setup['max_step']):
        time_step += 1
        print(time_step)
        tplot = setup['tplot'][count]



        ADI_dirichlet_update(num_soln, Nx, Ny, gamma_x=setup['d1'] * dth / dx ** 2,
                             gamma_y=setup['d2'] * dth / dy ** 2)
        t += dt
        if abs(t - tplot) < dth / 2:
            count += 1
            output_state(num_soln, count, Nx, Ny, d1=setup['d1'], d2=setup['d2'])


def main():
    u_init = gaussian_initial(setup['XL'], setup['XR'], setup['YBOT'], setup['YTOP'], setup['Nx'], setup['Ny'])
    print('go')
    ADI_scheme(u_init, setup['Nx'], setup['Ny'], dx=(setup['XR'] - setup['XL']) / (setup['Nx'] - 1),
               dy=(setup['YTOP'] - setup['YBOT']) / (setup['Ny'] - 1), dt=setup['dt'])

    if not True:
        N = 1
        for t in setup['tplot']:
            np.savetxt('{directory}/data/{myname}_sin_ADI_{Nx}_{Ny}_{d1}_{d2}_{N}_exact_state.csv'.format(
                directory=curdir,
                myname=myname,
                Nx=setup['Nx'],
                Ny=setup['Ny'],
                d1=setup['d1'],
                d2=setup['d2'],
                N=N
            ), np.exp(
                -setup['d1'] * (setup['k1'] * np.pi) ** 2 * t - setup['d2'] * (setup['k2'] * np.pi) ** 2 * t) * u_init,
                delimiter=',')
            N += 1


def unit_test_initial():
    u_init = gaussian_initial(setup['XL'], setup['XR'], setup['YBOT'], setup['YTOP'], setup['Nx'], setup['Ny'])
    X = np.full((setup['Ny'], setup['Nx']), 0, dtype=np.float64)
    Y = np.full((setup['Ny'], setup['Nx']), 0, dtype=np.float64)
    step_x = (setup['XR'] - setup['XL']) / (setup['Nx'] - 1)
    step_y = (setup['YBOT'] - setup['YTOP']) / (setup['Ny'] - 1)

    X = np.arange(setup['XL'], setup['XR'] + step_x / 2, step_x)
    Y = np.arange(setup['YTOP'], setup['YBOT'] + step_y / 2, step_y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z=u_init)
    plt.show()


if __name__ == '__main__':
    main()
    print(setup)
