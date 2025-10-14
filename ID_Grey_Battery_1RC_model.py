from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.io import loadmat
from tqdm import tqdm # For the progress bar
import os
import scipy.io
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, LinearInterpolation
#Cheking where is JAX running
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

# Signal generation parameters
N = time.shape[0]
Ts = time[1]-time[0]
fs = 1/Ts
T = time[-1]  # Total time in seconds

print(f"N ={N:.4f}\nfs={fs:.4f}\nT = {T:.4f}\nTs = {Ts:.4f}")


n_shots = 100 # number of shots
n_timesteps_per_shot = N // n_shots
n_states = 2 # number of states of the model
n_params = 4 # number of parameters

# Reshape data into batches for multiple shooting
t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)

# The JAX-compatible model evaluates the interpolated input u at any time t
def battery_1rc_jax(t, x, args):
    R0, R1, C1, n,  u_interp = args
    u = u_interp.evaluate(t)
    dx  = [-n*u/3440.05372, -1/R1/C1*x[1]+1/C1*u]
    dx = jnp.array(dx)
    return dx

term = ODETerm(battery_1rc_jax)
solver = Dopri5()

def create_loss_and_constraint_fns(t_shots, y_data, u_interp):
    """A factory to create the objective and constraint functions."""
    
    @jax.jit
    def objective_jax(decision_vars):
        R0, R1, C1, n = decision_vars[:n_params]
        x_initial_shots = decision_vars[n_params:].reshape(n_shots, n_states)
        args = (R0,R1,C1,n, u_interp)
        
        def simulate_shot(t_shot, x0):
            saveat = SaveAt(ts=t_shot)
            sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=x0, saveat=saveat, args=args)
            return sol.ys 
        
        # x_pred shape: (N_shots, N_steps, N_states)
        x_pred = jax.vmap(simulate_shot)(t_shots, x_initial_shots)
        
        # Função que calcula a saída para UM PASSO de tempo
        def model_output_step(t, x_step, R0_scalar, u_interp_obj):
            u = u_interp_obj.evaluate(t)
            #OCV = p(SOC)
            p = jnp.array([1.02726610e+03,
                           -5.13266541e+03,
                           1.09109051e+04,
                           -1.28481333e+04,
                           9.13851696e+03,
                           -4.01608666e+03,
                           1.07265101e+03,
                           -1.65017255e+02,
                           1.36600705e+01,
                           3.10715139e+00])
            OCV = jnp.polyval(p, x_step[0])
            # Output equation: OCV + R0*u + Vc (x[1])
            y_pred_step = OCV + R0_scalar * u + x_step[1]
            return y_pred_step

        def process_shot_output(t_shot, x_shot, R0_scalar, u_interp_obj):
            return jax.vmap(model_output_step, in_axes=(0, 0, None, None))(t_shot, x_shot, R0_scalar, u_interp_obj)
        
        y_pred = jax.vmap(process_shot_output, in_axes=(0, 0, None, None))(t_shots, x_pred, R0, u_interp)
        
        # Loss function
        return jnp.sum((y_pred - y_data)**2)

    @jax.jit
    def continuity_constraints_jax(decision_vars):
        R0,R1,C1,n = decision_vars[:n_params]
        x_initial_shots = decision_vars[n_params:].reshape(n_shots, n_states)
        args = (R0,R1,C1,n, u_interp)

        def get_end_state(t_shot, x0):

            sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=x0, args=args)
            return sol.ys[-1]

        x_end_of_shots = jax.vmap(get_end_state)(t_shots[:-1], x_initial_shots[:-1])

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

# Set up the optimization problem - initial guess
R0_guess = 0.0268
R1_guess = 56.323
C1_guess = 3620.4
n_guess  = 1e-4
param_guess = jnp.array([R0_guess, R1_guess, C1_guess, n_guess])

# Guess for the states (x_0,1)
x_initial_first_shot = jnp.array([0.98, 0.0]) # SOC em 98%, Vc em 0V

x_initial_shots_repeated = jnp.tile(x_initial_first_shot, n_shots)

# Concatenate the initial guess for the parameters and states
initial_guess_np = jnp.concatenate([param_guess, x_initial_shots_repeated])
cons = ({'type': 'eq', 'fun': cons_for_scipy, 'jac': cons_jac_for_scipy})

# Setting bounds for parameters and states
b_R0 = (1e-6, 1)
b_R1 = (1e-6, 1e10)
b_C1 = (1.0, 50000.0)
b_n = (1e-6, 1)
param_bounds = [b_R0, b_R1, b_C1, b_n]

b_no_limit = (None, None)
state_bounds_one_shot = [b_no_limit, b_no_limit]
state_bounds_all_shots = state_bounds_one_shot * n_shots
all_bounds = param_bounds + state_bounds_all_shots

# Run the optimization with a progress bar
max_iterations = 10000
with tqdm(total=max_iterations, desc="Optimizing Parameters") as pbar:
    def callback(xk):
        pbar.update(1)

    print("--- Running Optimization with Automatic Differentiation ---")
    result = minimize(obj_for_scipy,
                      initial_guess_np,
                      method='SLSQP',
                      jac=True, # Tells SciPy that our objective function returns value and gradient
                      constraints=cons,
                      bounds=all_bounds,
                      options={'maxiter': max_iterations, 'disp': False}, # Set disp=False for cleaner output with tqdm
                      callback=callback,
                      tol=1e-6
                      )

print("\nOptimization finished with status:", result.message)

# Extract and display results
R0,R1,C1,n = result.x[:n_params]
x_initial_estimated = jnp.array(result.x[n_params:n_params+n_states])

print("\n--- Identification Results ---")
print(f"Estimated parameters: R0 = {R0:.4f}, R1 = {R1:.4f}, C1 = {C1:.4f}, n = {n:.4f}")

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
    y_pred_step = OCV + R0_scalar * u + x_step[1]
    return y_pred_step

final_args = (R0, R1,C1,n, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=x_initial_estimated, saveat=SaveAt(ts=jnp.array(time)), args=final_args,max_steps=100000)
# final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args_
yhat = final_sol.ys.flatten()
y_hat = jax.vmap(model_output_step, in_axes=(0, 0, None, None))(time,final_sol.ys, R0, u_interpolation)

#Metrics
MSE = np.mean((y-y_hat)**2)
y_mean = jnp.mean(y)
RSS = jnp.sum((y - y_hat)**2)
TSS = jnp.sum((y - y_mean)**2)
r2 = 1.0 - (RSS / TSS)

print(f"R²: {r2:.4f}, MSE = {MSE:.4f}")

# Plot final results
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='True state', alpha=0.4)
plt.plot(time,  y_hat, 'b--', label='Identified Model Prediction', linewidth=2)
plt.plot(time, y-y_hat, 'r', label='Residue', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result')
plt.legend()
plt.grid(True)
plt.show()

#Loading validation dataset
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

print(f"N ={N:.4f}\nfs={fs:.4f}\nT = {T:.4f}\nTs = {Ts:.4f}")

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
plt.plot(time, y-y_hat, 'r', label='Residue', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result')
plt.legend()
plt.grid(True)
plt.show()

#Metrics
MSE = np.mean((y-y_hat)**2)
y_mean = jnp.mean(y)
RSS = jnp.sum((y - y_hat)**2)
TSS = jnp.sum((y - y_mean)**2)
r2 = 1.0 - (RSS / TSS)
print(f"R²: {r2:.4f}, MSE = {MSE:.4f}")
