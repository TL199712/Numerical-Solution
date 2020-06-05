import numpy as np # Use package NumPy and rename it to np
from functools import partial 
import time

# Purpose: Implement different ... (see arguments of run_method). You can change 

# Parameter

BeginResolu = 6
EndResolu = 12
sigma = 0.9
Wave_Speed = 3.0
End_Time = 9.0
STATERANGE = np.array([4096]) # Determine the state data interesting you

# Username

myname = "LMNLI"

# Data type of matrix error_data[norm]

error_dtype = [
    ('k',np.int64),
    ('h',np.float64),
    ('Fromm',np.float64),
    ('FvL',np.float64)
]

# Data type of matrix error_data[norm]

state_dtype = [
    ('x_i',np.float64),
    ('u_i',np.float64)
]

# Norm Calculation Functions

def Norm_L1(h,u):
    # Purpose: Calculate the L1 Norm of u, with stepsize h
    return h * np.sum(np.abs(u))

def Norm_L2(h,u):
    # Purpose: Calculate the L2 Norm of u, with stepsize h
    return np.sqrt(h * np.sum(u*u))

def Norm_Linf(h,u):
    # Purpose: Calculate the Linf Norm of u, with stepsize h
    return np.amax(np.abs(u))

# Initial Data

def semicircle(x):
    # Purpose: semicircle initial data
    return np.sqrt(0.25-(x-0.5)*(x-0.5))

def square_wave(x):
    # Purpose: square wave intiial data
    return np.where((x>=0.25)&(x<=0.75),1.0,0.0)

def gaussian(x):
    # Purpose: Gaussian-like intial data
    return np.exp(-256.0*(x-0.5)**2)

def sinusoidal(x):
    # Purpose: sinusoidal intial data
    return 0.5 * (1.0 - np.cos(2 * np.pi * x))

# Methods

def Fromm_Flux(u,h):
    # Purpose: calculate the flux for fromm's method
    global sigma
    return Wave_Speed * ( u + (1.0-sigma) * 0.25 * ( np.roll(u,-1) - np.roll(u,1) ) )

def FvL_Flux(u,h):
    # Purpose: calculate the flux for FvL method
    global sigma
    # Calculate \Theta  
    temp_A = 2 * np.abs(u-np.roll(u,1))
    temp_B = np.abs(np.roll(u,-1)-np.roll(u,1)) / 2
    temp_C = 2 * np.abs(np.roll(u,-1)-u)
    temp_B = np.where(temp_B<temp_C,temp_B,temp_C)
    Theta = np.sign(np.roll(u,-1) - np.roll(u,1)) * np.where(temp_A<temp_B,temp_A,temp_B)
    # Decide by \phi
    phi = (np.roll(u,-1)-u) * (u-np.roll(u,1))
    return Wave_Speed * (u + ( 1.0 - sigma ) * 0.5 * np.where(phi > 0, Theta, 0))

def flux_method(u_init, h, MAXSTEP, flux_func):
    # Purpose: implement all flux-like method
    # MAXSTEP is involved in order for different choices
    global sigma, End_Time, Wave_Speed
    max_dt = sigma * h / Wave_Speed
    num_solu = np.array(u_init,copy=True)
    # Initialize t
    t = 0.0
    time_step = 0
    while( t < End_Time ):
        dt = min(max_dt , End_Time - t)
        t += dt
        time_step += 1
        flux = flux_func(num_solu,h)
        num_solu += dt/h * ( np.roll(flux,1) - np.roll(flux,0) )
        #print(np.sum(flux))
    return num_solu

# Settings for specific task

methods = {
    "Fromm": partial(flux_method, flux_func = Fromm_Flux),
    "FvL": partial(flux_method, flux_func = FvL_Flux)
}
initials = {
#    "SINUSOIDAL": sinusoidal,
    "GAUSSIAN": gaussian,
    "SEMICIRCLE": semicircle,
    "SQUARE_WAVE": square_wave
}
norms = {
    "L1": Norm_L1,
    "L2": Norm_L2,
    "Linf": Norm_Linf
}

# Periodic grid
def peri_gird(mesh):
    # Purpose: avoid using ceiling function in the 'initials'. Turn to modify the grid.
    # Note: this function is useless for this assignment. It is used for checking if the flux has a correct direction.
    mesh -= Wave_Speed * End_Time
    return mesh - np.floor(mesh)

# Ostream file

def ostream_error_file(init, norm, Matrix):
    # Purpose: output error file as desired.
    fmt = ["%d", "%12.6e", "%12.6e", "%12.6e"] # fmt is desired output format *** Make sure Matrix is an n*4 matrix.
    np.savetxt('{myname}_{init}_{norm}_ERROR.csv'.format(
        myname=myname,
        init=init,
        norm=norm
    ), Matrix, fmt=fmt, delimiter=',')

def ostream_state_file(M,init,method,num_solu):
    # Purpose: output state file as desired. *** Make sure ___ is an n*1 matrix
    # Decide whether or not out put the data
    if M in STATERANGE :
        temp_matrix=np.zeros((M,),dtype=state_dtype)
        temp_matrix['x_i']=np.arange(1.0/M/2,1,1.0/M)
        temp_matrix['u_i']=num_solu.T
        np.savetxt('{myname}_{init}_{method}_{M}_STATE.csv'.format(
            myname = myname,
            init = init,
            method = method,
            M =M
        ),temp_matrix, fmt = "%1.15e", delimiter=',')
    

# Implement different ... (see arguments of run_method)

def run_method(k,method,init,error_data):
    # Purpose: implement a particular method with a particular initial data and a particular stepsize
    # error_data changes when this function is used. 
    # ***Please keep error_data here the same as that in all_data_CA02.
    # Mesh
    M = 2**k
    MAXSTEP = End_Time / sigma * Wave_Speed * M
    h = 1.0 / M
    mesh = np.arange(h/2, 1, h)
    # Exact solutions for linear conservation laws
    
    u_init = initials[init](mesh)
    num_solu = methods[method](u_init, h, MAXSTEP) # Wrong Syntax: num_solu = flux_method(u_init,h,method) since method has been reloaded.
    print(init+method)
    diff = num_solu - u_init
    
    for norm in norms:
    #    print(str(norms[norm](h,diff))+norm+method+str(k))
        ostream_state_file(M,init,method,num_solu)
        error_data[norm]["h"][error_data[norm]["k"] == k] = h
        error_data[norm][method][error_data[norm]["k"] == k] = norms[norm](h,diff)
    #    print(error_data[norm])

time_dtype =[
    ('k',np.int64),
    ('Fromm',np.float64),
    ('FvL',np.float64)
]

def all_data_CA02():
    k_list = range(BeginResolu,EndResolu+1)


    error_data={}
    for norm in norms:
        error_data[norm] = np.ones((len(k_list),),dtype=error_dtype)
        error_data[norm]["k"] = k_list
    
    for init in initials:
        time_tick = np.full((len(k_list),),np.nan,dtype = time_dtype)
        time_tick['k'] = k_list
        for method in methods:
            for k in k_list:
                start = time.clock()
                run_method(k,method,init,error_data)
                time_tick[method][time_tick['k']==k] = time.clock() - start
        np.savetxt('{init}_s{sigma}_runtime.csv'.format(
            init = init,
            sigma = str(10*sigma)
        ),time_tick,fmt="%d,%.5f,%.5f",delimiter=',')
        for norm in error_data:
            ostream_error_file(init,norm,error_data[norm])
    