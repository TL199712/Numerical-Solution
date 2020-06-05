import numpy as np
from functools import partial

myname = 'litao'
# Solve the PDE of form: u_t + u * u_x = - u * z_x with periodic conditions

##
# Basic functions
## 

##################
# Initialization #
##################

state_dtype = [
    ('x_i',np.float64),
    ('u_i',np.float64)
]

end_time = 0.1
CFL_number = 0.4
N_list = [10,20,40,80,160,320,640,1280,2560,5120,10240,20480]
N_list = [40]

def zz(x):
    return  (np.cos(np.pi*x))**6/10

def smooth(x):
    return 2 * np.ones(len(x))-zz(x)

def sinusoidal(x):
    return 1 - np.cos(np.pi*x)



##
# output
######

def output_data_file(N,num_solu,method):
    temp = np.full((N,),np.nan,dtype=state_dtype)
    x = np.arange(1.0/N,2,2.0/N)
    temp['x_i'] = x
    temp['u_i'] = num_solu.T
    np.savetxt('{myname}_{method}_{N}_state.csv'.format(
        myname = myname,
        method = method,
        N = N
    ),temp, fmt = "%1.30e", delimiter=',')
 
#################
# Fluxes Method #
#################

def approx_right(u,dx):
    return (np.roll(u,2) - 8 * np.roll(u,1) + 37 * u +37 * np.roll(u,-1) -8 * np.roll(u,-2) + np.roll(u,-3))/60
    #return (-np.roll(u,1) + 9 * u +9 * np.roll(u,-1) - np.roll(u,-2))/16

def derivative_1_right(u,dx):
    return (-2 * np.roll(u,2) + 25 * np.roll(u,1) - 245 * u + 245 * np.roll(u,-1) - 25 * np.roll(u,-2) + 2 * np.roll(u,-3))/(180*dx)
    #return (np.roll(u,1) - 15 * u +15 * np.roll(u,-1) - np.roll(u,-2))/(12 * dx)

def derivative_2_right(u,dx):
    return (-np.roll(u,2) + 7 * np.roll(u,1) - 6 * u - 6 * np.roll(u,-1) + 7 * np.roll(u,-2) - np.roll(u,-3))/(8*dx**2)
    #return (-np.roll(u,2) +6 * np.roll(u,1) - 8 * u +2 * np.roll(u,-1) + np.roll(u,-2))/(4 * dx**2)

def derivative_3_right(u,dx):
    return (np.roll(u,2) - 11*np.roll(u,1) + 28 * u - 28 * np.roll(u,-1) + 11*np.roll(u,-2) - np.roll(u,-3))/(6 * dx**3)

def approx_mid(u,dx):
    return (9/1920*np.roll(u,2) -116/1920*np.roll(u,1) + 2134/1920 * u - 116/1920 * np.roll(u,-1) +9/1920 * np.roll(u,-2))

def approx_right_3(u,dx):
    return (np.roll(u,2) - 8 * np.roll(u,1) + 37 * u +37 * np.roll(u,-1) -8 * np.roll(u,-2) + np.roll(u,-3))/60

def derivative_1_right_3(u,dx):
    return (np.roll(u,1) - 27 * u + 27 * np.roll(u,-1) - np.roll(u,-2))/(24 * dx)

def derivative_2_right_3(u,dx):
    return (-5*np.roll(u,2) + 39 * np.roll(u,1) - 34 * u - 34 * np.roll(u,-1) + 39 * np.roll(u,-2) - 5* np.roll(u,-3))/(48 * dx**2)

def derivative_3_right_3(u,dx):
    return (np.roll(u,2) - 13*np.roll(u,1) + 34 * u - 34 * np.roll(u,-1) + 13*np.roll(u,-2) - np.roll(u,-3))/(8 * dx**3)
    
    
def ADER_3_Flux(u,e,dx,dt):
    u_right = approx_right(u,dx)
    ux_right = derivative_1_right(u,dx)
    uxx_right = derivative_2_right(u,dx)
    ex_right = derivative_1_right(e,dx)
    exx_right = derivative_2_right(e,dx)
    exxx_right = derivative_3_right(e,dx)
    R_1 = - u_right**2 * ex_right
    R_2 = 2 * u_right**2 * ex_right**2 + u_right**2 * ux_right * ex_right + u_right**3 * exx_right
    R_3 = (- u_right**4 * exxx_right 
        - 3 * u_right**3 * ux_right * exx_right 
        - u_right**3 * ex_right * uxx_right
        - 8 * u_right**3 * ex_right * exx_right
        - 7 * u_right**2 * ux_right * ex_right**2
        - u_right**2 * ux_right**2 * ex_right
        - 4 * u_right**2 * ex_right**3)
    return u_right**2/2 + dt/2 * R_1 + dt**2/6 * R_2 + dt**3/24 * R_3

def ADER_3_Source(z,u,e,dx,dt):
    mesh = np.arange(dx/2 , 2 , dx)
    dz = -6 * np.pi*np.cos(np.pi*mesh)**5*np.sin(np.pi*mesh)
    mesh = np.arange (dx, 2+dx ,dx)
    dz_right =-6 * np.pi*np.cos(np.pi*mesh)**5*np.sin(np.pi*mesh)

    u_right = approx_right(u,dx)
    ux_right = derivative_1_right(u,dx)
    uxx_right = derivative_2_right(u,dx)
    e_right = approx_right(e,dx)
    ex_right = derivative_1_right(e,dx)
    exx_right = derivative_2_right(e,dx)
    exxx_right = derivative_3_right(e,dx)

    '''u = approx_mid(u,dx)
    ux_mid = derivative_1_mid_3(e,dx)
    uxx_mid = derivative_2_mid_3(u,dx)
    ex_mid = derivative_1_mid_3(e,dx)
    exx_mid = derivative_2_mid_3(e,dx)
    exxx_mid = derivative_3_mid_3(e,dx)'''

    u=approx_mid(u,dx)
    ux_mid = derivative_1_right_3(np.roll(u_right,1),dx)
    uxx_mid = derivative_2_right_3(np.roll(u_right,1),dx)
    ex_mid = derivative_1_right_3(np.roll(e_right,1),dx)
    exx_mid = derivative_2_right_3(np.roll(e_right,1),dx)
    exxx_mid = derivative_3_right_3(np.roll(e_right,1),dx)

    '''ux_mid = approx_right_3(np.roll(ux_right,1),dx)
    uxx_mid = approx_right_3(np.roll(uxx_right,1),dx)
    ex_mid = approx_right_3(np.roll(ex_right,1),dx)
    exx_mid = approx_right_3(np.roll(exx_right,1),dx)
    exxx_mid = approx_right_3(np.roll(exxx_right,1),dx)'''

    '''ux_mid = derivative_1_mid_3(u,dx)
    uxx_mid = derivative_2_mid_3(u,dx)
    ex_mid = derivative_1_mid_3(e,dx)
    exx_mid = derivative_2_mid_3(e,dx)
    exxx_mid = derivative_3_mid_3(e,dx)'''


    s_0_right = - u_right * dz_right
    s_1_right = u_right * ex_right * dz_right
    s_2_right = (-u_right * ex_right**2 - u_right * ux_right * ex_right - u_right**2 * exx_right) * dz_right
    s_3_right = ( u_right**3 * exxx_right + 3 * u_right**2 * ux_right * exx_right
        + 5 * u_right**2 * ex_right * exx_right + u_right**2 * ex_right * uxx_right
        + u_right * ex_right**3 + 4 * u_right * ex_right**2 * ux_right
        + u_right * ux_right**2 * ex_right) * dz_right

    s_0_mid = - u * dz
    s_1_mid = u * ex_mid * dz
    s_2_mid = (-u * ex_mid**2 - u * ux_mid * ex_mid - u**2 * exx_mid ) * dz
    s_3_mid = ( u**3 * exxx_mid + 3 * u**2 * ux_mid * exx_mid 
        + 5 * u**2 * ex_mid * exx_mid + u**2 * ex_mid * uxx_mid
        + u * ex_mid**3 + 4 * u * ex_mid**2 * ux_mid
        + u * ux_mid**2 * ex_mid) * dz

    '''s_0_lm = - 1/16 * np.roll(s_0_mid,1) + 9/16 * np.roll (s_0_right,1) + 9/16 * np.roll(s_0_mid, 0) - 1/16 * np.roll(s_0_right, 0)
    s_0_lr = - 1/16 * np.roll(s_0_right,1) + 9/16 * np.roll (s_0_mid,0) + 9/16 * np.roll(s_0_right, 0) - 1/16 * np.roll(s_0_mid, -1)
    s_1_lm = - 1/16 * np.roll(s_1_mid,1) + 9/16 * np.roll (s_1_right,1) + 9/16 * np.roll(s_1_mid, 0) - 1/16 * np.roll(s_1_right, 0)
    s_1_lr = - 1/16 * np.roll(s_1_right,1) + 9/16 * np.roll (s_1_mid,0) + 9/16 * np.roll(s_1_right, 0) - 1/16 * np.roll(s_1_mid, -1)
    s_2_lm = - 1/16 * np.roll(s_2_mid,1) + 9/16 * np.roll (s_2_right,1) + 9/16 * np.roll(s_2_mid, 0) - 1/16 * np.roll(s_2_right, 0)
    s_2_lr = - 1/16 * np.roll(s_2_right,1) + 9/16 * np.roll (s_2_mid,0) + 9/16 * np.roll(s_2_right, 0) - 1/16 * np.roll(s_2_mid, -1)
    s_3_lm = - 1/16 * np.roll(s_3_mid,1) + 9/16 * np.roll (s_3_right,1) + 9/16 * np.roll(s_3_mid, 0) - 1/16 * np.roll(s_3_right, 0)
    s_3_lr = - 1/16 * np.roll(s_3_right,1) + 9/16 * np.roll (s_3_mid,0) + 9/16 * np.roll(s_3_right, 0) - 1/16 * np.roll(s_3_mid, -1)'''

    
    return  (dx * (1/6 * s_0_right + 1/6 * np.roll(s_0_right,1) + 2/3 * s_0_mid)
        + dx * (1/6 * s_1_right + 1/6 * np.roll(s_1_right,1) + 2/3 * s_1_mid) * dt / 2 
        + dx * (1/6 * s_2_right + 1/6 * np.roll(s_2_right,1) + 2/3 * s_2_mid) * dt**2 /6)
    
def mean_approx_right_4(u,dx):
    return (-3 * np.roll(u,3) + 29 * np.roll(u,2) -139 * np.roll(u,1) +533 * u + 533 * np.roll(u,-1) -139 * np.roll(u,-2) + 29 * np.roll(u,-3) -3 * np.roll(u,-4))/840

def mean_derivative_1_right_4(u,dx):
    return (9 * np.roll(u,3) - 119 * np.roll(u,2) + 889 * np.roll(u,1) - 7175 * u + 7175 * np.roll(u,-1) - 889 * np.roll(u,-2) + 119 * np.roll(u,-3) - 9 * np.roll(u,-4))/(5040 * dx)

def mean_derivative_2_right_4(u,dx):
    return (7 * np.roll(u,3) - 65 * np.roll(u,2) + 273 * np.roll(u,1) - 215 * u - 215 * np.roll(u,-1) + 273 * np.roll(u,-2) - 65 * np.roll(u,-3) + 7 * np.roll(u,-4))/(240 * dx**2)

def mean_derivative_3_right_4(u,dx):
    return (-7 * np.roll(u,3) + 89 * np.roll(u,2) - 587 * np.roll(u,1) + 1365 * u - 1365 * np.roll(u,-1) + 587 * np.roll(u,-2) - 89 * np.roll(u,-3) + 7 * np.roll(u,-4))/(240 * dx**3)

def mean_apporx_mid_4(u,dx):
    return (-75 * np.roll(u,3) + 954 * np.roll(u,2) - 7621 * np.roll(u,1) + 121004 * u - 7621 * np.roll(u,-1) + 954 * np.roll(u,-2) -75 * np.roll(u,-3))/(107520)

def derivative_1_right_4(u,dx):
    return (-3/640 * np.roll(u,2) + 25/384 * np.roll(u,1) - 75/64 * u + 75/64 * np.roll(u,-1) - 25/384 * np.roll(u,-2) + 3/640 * np.roll(u,-3))/dx

def derivative_2_right_4(u,dx):
    return (-5 * np.roll(u,2) +39 * np.roll(u,1) - 34 * u - 34 * np.roll(u,-1) + 39 * np.roll(u,-2) - 5 * np.roll(u,-3) )/(48 * dx**2)

def derivative_3_right_4(u,dx):
    return (-37* np.roll(u,3) + 499 * np.roll(u,2) - 3897 * np.roll(u,1) + 9455 * u - 9455 * np.roll(u,-1) + 3897 * np.roll(u,-2) - 499*np.roll(u,-3) + 37 * np.roll(u,-4))/(1920*dx**3)

def ADER_4_Flux(u,e,dx,dt):
    u_right = mean_approx_right_4(u,dx)
    ux_right = mean_derivative_1_right_4(u,dx)
    uxx_right = mean_derivative_2_right_4(u,dx)
    ex_right = mean_derivative_1_right_4(e,dx)
    exx_right = mean_derivative_2_right_4(e,dx)
    exxx_right = mean_derivative_3_right_4(e,dx)
    R_1 = - u_right**2 * ex_right
    R_2 = 2 * u_right**2 * ex_right**2 + u_right**2 * ux_right * ex_right + u_right**3 * exx_right
    R_3 = (- u_right**4 * exxx_right 
        - 3 * u_right**3 * ux_right * exx_right 
        - u_right**3 * ex_right * uxx_right
        - 8 * u_right**3 * ex_right * exx_right
        - 7 * u_right**2 * ux_right * ex_right**2
        - u_right**2 * ux_right**2 * ex_right
        - 4 * u_right**2 * ex_right**3)
    return u_right**2/2 + dt/2 * R_1 + dt**2/6 * R_2 + dt**3/24 * R_3

def ADER_4_Source(z,u,e,dx,dt):
    mesh = np.arange(dx/2 , 2 , dx)
    dz = 2/5* np.pi * np.sin(np.pi*mesh)**3*np.cos(np.pi*mesh)
    mesh = np.arange (dx, 2+dx ,dx)
    dz_right = 2/5 * np.pi * np.sin(np.pi*mesh)**3*np.cos(np.pi*mesh)

    u_right = mean_approx_right_4(u,dx)
    ux_right = mean_derivative_1_right_4(u,dx)
    uxx_right = mean_derivative_2_right_4(u,dx)
    e_right = mean_approx_right_4(e,dx)
    ex_right = mean_derivative_1_right_4(e,dx)
    exx_right = mean_derivative_2_right_4(e,dx)
    exxx_right = mean_derivative_3_right_4(e,dx)

    u=mean_apporx_mid_4(u,dx)
    ux_mid = derivative_1_right_4(np.roll(u_right,1),dx)
    uxx_mid = derivative_2_right_4(np.roll(u_right,1),dx)
    ex_mid = derivative_1_right_4(np.roll(e_right,1),dx)
    exx_mid = derivative_2_right_4(np.roll(e_right,1),dx)
    exxx_mid = derivative_3_right_4(np.roll(e_right,1),dx)

    s_0_right = - u_right * dz_right
    s_1_right = u_right * ex_right * dz_right
    s_2_right = (-u_right * ex_right**2 - u_right * ux_right * ex_right - u_right**2 * exx_right) * dz_right
    s_3_right = ( u_right**3 * exxx_right + 3 * u_right**2 * ux_right * exx_right
        + 5 * u_right**2 * ex_right * exx_right + u_right**2 * ex_right * uxx_right
        + u_right * ex_right**3 + 4 * u_right * ex_right**2 * ux_right
        + u_right * ux_right**2 * ex_right) * dz_right
    s_0_mid = - u * dz
    s_1_mid = u * ex_mid * dz
    s_2_mid = (-u * ex_mid**2 - u * ux_mid * ex_mid - u**2 * exx_mid ) * dz
    s_3_mid = ( u**3 * exxx_mid + 3 * u**2 * ux_mid * exx_mid 
        + 5 * u**2 * ex_mid * exx_mid + u**2 * ex_mid * uxx_mid
        + u * ex_mid**3 + 4 * u * ex_mid**2 * ux_mid
        + u * ux_mid**2 * ex_mid) * dz

    s_0_lm =( 3/256 * np.roll(s_0_right,2) - 25/256 * np.roll(s_0_mid,1) + 75/128 * np.roll(s_0_right,1)
         + 75/128 * s_0_mid - 25/256 * s_0_right + 3/256 * np.roll(s_0_mid,-1) )
    s_0_rm = (3/256 * np.roll(s_0_mid,1) - 25/256 * np.roll(s_0_right,1) + 75/128 * s_0_mid
         + 75/128 * s_0_right - 25/256 * np.roll(s_0_mid,-1) + 3/256 * np.roll(s_0_right,-1))
    s_1_lm =( 3/256 * np.roll(s_1_right,2) - 25/256 * np.roll(s_1_mid,1) + 75/128 * np.roll(s_1_right,1)
         + 75/128 * s_1_mid - 25/256 * s_1_right + 3/256 * np.roll(s_1_mid,-1) )
    s_1_rm =( 3/256 * np.roll(s_1_mid,1) - 25/256 * np.roll(s_1_right,1) + 75/128 * s_1_mid
         + 75/128 * s_1_right - 25/256 * np.roll(s_1_mid,-1) + 3/256 * np.roll(s_1_right,-1))
    s_2_lm =( 3/256 * np.roll(s_2_right,2) - 25/256 * np.roll(s_2_mid,1) + 75/128 * np.roll(s_2_right,1)
         + 75/128 * s_2_mid - 25/256 * s_2_right + 3/256 * np.roll(s_2_mid,-1) )
    s_2_rm =( 3/256 * np.roll(s_2_mid,1) - 25/256 * np.roll(s_2_right,1) + 75/128 * s_2_mid
         + 75/128 * s_2_right - 25/256 * np.roll(s_2_mid,-1) + 3/256 * np.roll(s_2_right,-1))
    s_3_lm =( 3/256 * np.roll(s_3_right,2) - 25/256 * np.roll(s_3_mid,1) + 75/128 * np.roll(s_3_right,1)
         + 75/128 * s_3_mid - 25/256 * s_3_right + 3/256 * np.roll(s_3_mid,-1) )
    s_3_rm =( 3/256 * np.roll(s_3_mid,1) - 25/256 * np.roll(s_3_right,1) + 75/128 * s_3_mid
         + 75/128 * s_3_right - 25/256 * np.roll(s_3_mid,-1) + 3/256 * np.roll(s_3_right,-1) )
    
    return (dx * (7/90 * np.roll(s_0_right,1) + 7/90 * s_0_right + 12/90 * s_0_mid + 32/90 * s_0_lm + 32/90 * s_0_rm )
    + dx * (7/90 * np.roll(s_1_right,1) + 7/90 * s_1_right + 12/90 * s_1_mid + 32/90 * s_1_lm + 32/90 * s_1_rm ) * dt/2
    + dx * (7/90 * np.roll(s_2_right,1) + 7/90 * s_2_right + 12/90 * s_2_mid + 32/90 * s_2_lm + 32/90 * s_2_rm ) * dt**2/6
    + dx * (7/90 * np.roll(s_3_right,1) + 7/90 * s_3_right + 12/90 * s_3_mid + 32/90 * s_3_lm + 32/90 * s_3_rm ) * dt**3 / 24 )
    
    '''(dx * (1/6 * s_0_right + 1/6 * np.roll(s_0_right,1) + 2/3 * s_0_mid)
    + dx * (1/6 * s_1_right + 1/6 * np.roll(s_1_right,1) + 2/3 * s_1_mid) * dt / 2 
    + dx * (1/6 * s_2_right + 1/6 * np.roll(s_2_right,1) + 2/3 * s_2_mid) * dt**2 /6)'''
    

def flux_method(u_init,dx,z_init,flux_func,source_func):
    t = 0 
    z = np.array(z_init,copy=True)
    num_solu_u = np.array(u_init,copy=True)
    num_solu_e = num_solu_u + z
    max_dt = dx * CFL_number
    while t <  end_time:
        dt = min(max_dt,end_time - t)
        flux = flux_func(num_solu_u,num_solu_e,dx,dt)
        source = source_func(z,num_solu_u,num_solu_e,dx,dt)
        num_solu_u += (dt/dx * (np.roll(flux,1) - flux) + dt/dx * source)
        num_solu_e = num_solu_u + z
        t += dt
    return num_solu_u

methods = {
    "ADER3":partial(flux_method,flux_func = ADER_3_Flux,source_func=ADER_3_Source),
#   "ADER4":partial(flux_method,flux_func = ADER_4_Flux,source_func=ADER_4_Source)
}

def run_method(N,method):
    dx = 2.0/N
    mesh = np.arange(1/N , 2 , 2/N)
    u_init = smooth(mesh)
    z_init = zz(mesh)
    num_solu = methods[method](u_init,dx,z_init)
    output_data_file(N,num_solu,method)

def all_data():
    for method in methods:
        for n in N_list:
            run_method(n,method)

all_data()