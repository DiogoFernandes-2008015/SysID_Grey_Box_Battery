# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:14:57 2025

@author: Lenovo
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat
from tqdm import tqdm # For the progress bar
import os
import scipy as sc
import scipy.io
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, LinearInterpolation
try:
    print(f"JAX is running on: {jax.devices()[0].platform.upper()}")
except IndexError:
    print("No JAX devices found.")

jax.config.update("jax_enable_x64", True)

# Loading training dataset
DATA = loadmat('data_train.mat')
u = DATA['i']
y = DATA['v']
time = DATA['t']


fig, axs = plt.subplots(2, 1, sharex=True) # sharex makes sense for time series

# Plot 1: Input u
axs[0].plot(time, u, color='b') # Added color for clarity
axs[0].set_title('Input Signal (u) vs. Time')
axs[0].set_ylabel('u (Input)')
axs[0].grid(True)

# Plot 2: Output y and Reference yref
axs[1].plot(time, y, 'k', label='y (Output)')
axs[1].set_title('Output (y) vs. Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout() # Adjusts subplot params for a tight layout
plt.show()

time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)
decimate = 1
u = u[::decimate]
y = y[::decimate]
time = time[::decimate]
# Signal generation parameters
# N = 2048  # Number of samples (power of 2 is efficient for FFT)
N = time.shape[0]
Ts = time[1]-time[0]
fs = 1/Ts
T = time[-1]  # Total time in seconds

print(N, fs, T, Ts)

n_shots = 10 # 8150 / 163 = 50 data points per shot.
n_timesteps_per_shot = N // n_shots
n_states = 4 # number of states of the model
n_pars = 7  #number of parameters

# Reshape data into batches for multiple shooting
t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)

# The JAX-compatible model evaluates the interpolated input u at any time t
def battery_pngv_jax(t, x, args):
    R0, C0,R1, C1, R2,C2, n,  u_interp = args
    u = u_interp.evaluate(t)
    dx  = [-n*u/3440.05372,
           u/C0,
           -1/R1/C1*x[2]+1/C1*u,
           -1/R2/C2*x[3]+1/C2*u]
    dx = jnp.array(dx)
    return dx

term = ODETerm(battery_pngv_jax)
solver = Dopri5()

def create_loss_and_constraint_fns(t_shots, y_data, u_interp):
    """A factory to create the objective and constraint functions."""
    
    @jax.jit
    def objective_jax(decision_vars):
        R0, C0,R1, C1,R2, C2, n = decision_vars[:n_pars]
        x_initial_shots = decision_vars[n_pars:].reshape(n_shots, n_states) 
        args = (R0,C1,R1,C1,R2,C2,n, u_interp)
        
        def simulate_shot(t_shot, x0):
            saveat = SaveAt(ts=t_shot)
            sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=x0, saveat=saveat, args=args)
            return sol.ys 
        
        # x_pred shape: (N_shots, N_steps, N_states)
        x_pred = jax.vmap(simulate_shot)(t_shots, x_initial_shots)
        
        # Função que calcula a saída para UM PASSO de tempo
        def model_output_step(t, x_step, R0_scalar, u_interp_obj):
            # Obtemos a corrente (u) usando o objeto interpolador
            u = u_interp_obj.evaluate(t)
            
            p = jnp.array([1.02726610e+03,
                           -5.13266541e+03,
                           1.09109051e+04,
                           -1.28481333e+04,
                           9.13851696e+03,
                           -4.01608666e+03,
                           1.07265101e+03,
                           -1.65017255e+02,
                           1.36600705e+01,
                           3.10715139e+00]) # Polinômio
            OCV = jnp.polyval(p, x_step[0])
            # A saída é: OCV + R0*u + Vc (x[1])
            y_pred_step = OCV + R0_scalar * u + x_step[1]+x_step[2]+x_step[3]
            return y_pred_step

        def process_shot_output(t_shot, x_shot, R0_scalar, u_interp_obj):
            # Mapeia sobre o tempo (N_steps). t, x_step variam (0), R0 e u_interp são constantes (None).
            return jax.vmap(model_output_step, in_axes=(0, 0, None, None))(t_shot, x_shot, R0_scalar, u_interp_obj)
        
        # Mapeia sobre os shots (N_shots). t_shot e x_shot variam (0), R0 e u_interp são constantes (None).
        y_pred = jax.vmap(process_shot_output, in_axes=(0, 0, None, None))(t_shots, x_pred, R0, u_interp)
        
        # Perda: y_pred e y_data agora têm shape (N_shots, N_steps)
        return jnp.sum((y_pred - y_data)**2)

    @jax.jit
    def continuity_constraints_jax(decision_vars):
        R0,C0,R1,C1,R2,C2,n = decision_vars[:n_pars]
        x_initial_shots = decision_vars[n_pars:].reshape(n_shots, n_states) 
        args = (R0,C0,R1,C1,R2,C2,n, u_interp)

        def get_end_state(t_shot, x0):
            # CORREÇÃO: y0 = x0
            sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=x0, args=args)
            return sol.ys[-1]

        x_end_of_shots = jax.vmap(get_end_state)(t_shots[:-1], x_initial_shots[:-1])
        # CORREÇÃO: X_end_of_shots -> x_end_of_shots
        return (x_end_of_shots - x_initial_shots[1:]).flatten()

    return objective_jax, continuity_constraints_jax
# Create the specific functions for our data
objective_jax, continuity_constraints_jax = create_loss_and_constraint_fns(t_shots, y_data, u_interpolation)

# Create JIT-compiled gradient and Jacobian functions
objective_grad_func = jax.jit(jax.value_and_grad(objective_jax))
# We use jacrev (reverse-mode AD) as it's compatible with the adjoint-based diffrax solver
constraints_jac_func = jax.jit(jax.jacrev(continuity_constraints_jax))

# Wrapper functions to interface between SciPy (NumPy) and JAX
def obj_for_scipy(dv_np):
    val, grad = objective_grad_func(jnp.array(dv_np))
    return np.array(val), np.array(grad)

def cons_for_scipy(dv_np):
    return np.array(continuity_constraints_jax(jnp.array(dv_np)))

def cons_jac_for_scipy(dv_np):
    return np.array(constraints_jac_func(jnp.array(dv_np)))

# Set up the optimization problem
R0_guess = 0.0268 
C0_guess = 1000
R1_guess = 56.323 
C1_guess = 3620.4 
R2_guess = 3000
C2_guess = 1000
n_guess  = 1e-4
param_guess = jnp.array([R0_guess,C0_guess, R1_guess, C1_guess, R2_guess, C2_guess,n_guess])

# 2. Palpite para o PRIMEIRO Estado Inicial (x_0,1)
# Estado inicial: [SOC, Vc]
x_initial_first_shot = jnp.array([0.98, 0,0,0]) # SOC em 98%, Vc em 0V

# 3. Palpite para os Estados dos Shots Intermediários
x_initial_shots_repeated = jnp.tile(x_initial_first_shot, n_shots) 
# Agora x_initial_shots_repeated tem 326 elementos (163 * 2)

# 4. Concatenar para o vetor de decisão final
initial_guess_np = jnp.concatenate([param_guess, x_initial_shots_repeated])
cons = ({'type': 'eq', 'fun': cons_for_scipy, 'jac': cons_jac_for_scipy})

# ---  Limites para os 4 Parâmetros (R0, R1, C1, n) ---

# [R0_min, R0_max] (Exemplo: R0 deve ser pequeno e positivo)
b_R0 = (1e-6, 1) 
# [R1_min, R1_max]
b_R1 = (1e-6, 5000.0) 
# [C1_min, C1_max] (Exemplo: Capacitância geralmente é grande)
b_C1 = (1.0, 50000.0) 
# [n_min, n_max] (Exemplo: Taxa de reação/perda)
b_n = (1e-6, 1)

param_bounds = [b_R0, b_C1,b_R1, b_C1,b_R1, b_C1, b_n]

# ---  Limites para os Estados Iniciais (Todos sem limites) ---

# Definimos 'None' para um único estado.
b_no_limit = (None, None) 

# Os estados iniciais são (SOC, Vc). Repetimos (None, None) duas vezes por shot.
state_bounds_one_shot = [b_no_limit, b_no_limit,b_no_limit,b_no_limit] 

# Repetimos essa dupla (None, None) para todos os n_shots
state_bounds_all_shots = state_bounds_one_shot * n_shots 

# Combina os limites dos parâmetros com as restrições livres dos estados
all_bounds = param_bounds + state_bounds_all_shots


# Run the optimization with a progress bar
max_iterations = 10000
with tqdm(total=max_iterations, desc="Optimizing Parameters") as pbar:
    def callback(xk):
        pbar.update(1)

    print("\n--- Running Optimization with Automatic Differentiation ---")
    result = minimize(obj_for_scipy,
                      initial_guess_np,
                      method='SLSQP',
                      jac=True, # Tells SciPy that our objective function returns value and gradient
                      constraints=cons,
                      bounds=all_bounds,
                      options={'maxiter': max_iterations, 'disp': False}, # Set disp=False for cleaner output with tqdm
                      callback=callback,
                      tol=1e-10
                      )

print("\nOptimization finished with status:", result.message)

# Extract and display results
R0,C0,R1,C1,R2,C2,n = result.x[:n_pars]
x_initial_estimated = jnp.array(result.x[n_pars:n_pars+n_states])

print("\n--- Identification Results ---")
print(f"Estimated parameters: R0 = {R0:.4f}, C0 = {C0:.4f}, R1 = {R1:.4f}, C1 = {C1:.4f},R2 = {R2:.4f}, C2 = {C2:.4f} n = {n:.4f}")

# Simulate the final model prediction

def model_output_step(t, x_step, R0_scalar, u_interp_obj):
    # Obtemos a corrente (u) usando o objeto interpolador
    u = u_interp_obj.evaluate(t)
    
    p = jnp.array([1.02726610e+03,
                   -5.13266541e+03,
                   1.09109051e+04,
                   -1.28481333e+04,
                   9.13851696e+03,
                   -4.01608666e+03,
                   1.07265101e+03,
                   -1.65017255e+02,
                   1.36600705e+01,
                   3.10715139e+00]) # Polinômio
    OCV = jnp.polyval(p, x_step[0])
    # A saída é: OCV + R0*u + Vc (x[1])
    y_pred_step = OCV + R0_scalar * u + x_step[1]+x_step[2]+x_step[3]
    return y_pred_step

final_args = (R0,C0, R1,C1,R2,C2,n, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=x_initial_estimated, saveat=SaveAt(ts=jnp.array(time)), args=final_args,max_steps=100000)
# final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args_
yhat = final_sol.ys.flatten()
y_hat = jax.vmap(model_output_step, in_axes=(0, 0, None, None))(time,final_sol.ys, R0, u_interpolation)

MSE = np.mean((y-y_hat)**2)
print(MSE)

# Média dos dados reais (y)
y_mean = jnp.mean(y)

# 1. Soma dos Quadrados dos Resíduos (RSS - Residual Sum of Squares)
# O numerador da fórmula do R²
RSS = jnp.sum((y - y_hat)**2)

# 2. Soma Total dos Quadrados (TSS - Total Sum of Squares)
# O denominador da fórmula do R² (Variância total dos dados)
TSS = jnp.sum((y - y_mean)**2)

# 3. Cálculo final do R²
r2 = 1.0 - (RSS / TSS)
print(r2)

print(f"R²: {r2:.4f}")

# Plot final results
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='True state', alpha=0.4)
plt.plot(time,  y_hat, 'b--', label='Identified Model Prediction', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result with Multisine Input')
plt.legend()
plt.grid(True)
plt.show()

DATA = loadmat('data_val.mat')
u = DATA['i']
y = DATA['v']
time = DATA['t']
time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)

# Signal generation parameters
# N = 2048  # Number of samples (power of 2 is efficient for FFT)
N = time.shape[0]
Ts = time[1]-time[0]
fs = 1/Ts
T = time[-1]  # Total time in seconds

print(N, fs, T, Ts)


# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)

# Simulate the final model prediction
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=x_initial_estimated, saveat=SaveAt(ts=jnp.array(time)), args=final_args,max_steps=100000)
# final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args_
yhat = final_sol.ys.flatten()
y_hat = jax.vmap(model_output_step, in_axes=(0, 0, None, None))(time,final_sol.ys, R0, u_interpolation)

# # Plot final results
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='True state', alpha=0.4)
plt.plot(time, y_hat, 'b--', label='Identified Model Prediction', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result with Multisine Input')
plt.legend()
plt.grid(True)
plt.show()


MSE = np.mean((y-y_hat)**2)
print(MSE)

# Média dos dados reais (y)
y_mean = jnp.mean(y)

# 1. Soma dos Quadrados dos Resíduos (RSS - Residual Sum of Squares)
# O numerador da fórmula do R²
RSS = jnp.sum((y - y_hat)**2)

# 2. Soma Total dos Quadrados (TSS - Total Sum of Squares)
# O denominador da fórmula do R² (Variância total dos dados)
TSS = jnp.sum((y - y_mean)**2)

# 3. Cálculo final do R²
r2 = 1.0 - (RSS / TSS)
print(r2)

print(f"R²: {r2:.4f}")