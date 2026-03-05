import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import os

# ======================================================================
# 1. CONFIGURAÇÕES E CARREGAMENTO DE DADOS
# ======================================================================
# Defina os nomes dos arquivos gerados/utilizados no script anterior
arquivo_dados = 'data_val.mat'  # Ou 'data_train.mat'
arquivo_resultados = 'results_1RC_v0artigo_2025-10-26_18-45-45.mat'  # Atualize este nome!

print("--- Iniciando Script de Estimação RLS Standalone ---")

# Carregando dados de medição (Tempo, Corrente, Tensão)
try:
    DATA = loadmat(arquivo_dados)
    u = DATA['i'].reshape(-1)
    y = DATA['v'].reshape(-1)
    time = DATA['t'].reshape(-1)

    Ts = time[1] - time[0]
    N_samples = len(time)
    print(f"Dados carregados de '{arquivo_dados}': N = {N_samples}, Ts = {Ts:.4f}s")
except FileNotFoundError:
    print(f"ERRO: Arquivo de dados '{arquivo_dados}' não encontrado.")
    exit()

# Carregando parâmetros de referência (Offline)
try:
    resultados = loadmat(arquivo_resultados)
    # Acessa a chave 'Params' e transforma em um vetor 1D
    parametros_offline = resultados['Params'].flatten()
    R0_ref, R1_ref, C1_ref, n_est = parametros_offline
    print(f"Parâmetros carregados de '{arquivo_resultados}':")
    print(f"R0={R0_ref:.4f}, R1={R1_ref:.4f}, C1={C1_ref:.4f}, n={n_est:.4f}")
except FileNotFoundError:
    print(f"AVISO: Arquivo '{arquivo_resultados}' não encontrado.")
    print("Usando parâmetros de fallback (padrão)...")
    parametros_offline = np.array([0.0268, 56.323, 3620.4, 0.98])
    R0_ref, R1_ref, C1_ref, n_est = parametros_offline

# ======================================================================
# 2. INICIALIZAÇÃO DO RLS
# ======================================================================
# Polinômio da OCV
p_ocv = np.array([1.02726610e+03, -5.13266541e+03, 1.09109051e+04, -1.28481333e+04,
                  9.13851696e+03, -4.01608666e+03, 1.07265101e+03, -1.65017255e+02,
                  1.36600705e+01, 3.10715139e+00])

# Configurações do Filtro
lambda_rls = 0.995  # Fator de esquecimento
P = np.eye(3) * 1e2  # Matriz de covariância inicial
theta = np.zeros((3, 1))  # Vetor de parâmetros discretos [a1, b0, b1]^T

# Arrays para histórico
R0_rls = np.zeros(N_samples)
R1_rls = np.zeros(N_samples)
C1_rls = np.zeros(N_samples)

# Estados Iniciais
soc_k = 0.98  # SOC inicial
V_dyn_prev = 0.0  # V_dyn(k-1)
u_prev = 0.0  # u(k-1)

# ======================================================================
# 3. LOOP PRINCIPAL DO RLS
# ======================================================================
for k in tqdm(range(N_samples), desc="Rodando Filtro RLS"):

    # Atualiza o SOC via Coulomb Counting
    if k > 0:
        soc_k = soc_k - (n_est * u[k] * Ts) / 3440.05372

    # Calcula OCV
    ocv_k = np.polyval(p_ocv, soc_k)

    # Tensão dinâmica medida V_dyn(k)
    V_dyn_k = y[k] - ocv_k

    if k > 0:
        # Vetor de Regressão Phi(k)
        phi_k = np.array([[V_dyn_prev], [u[k]], [u_prev]])

        # Erro de predição a priori
        e_k = V_dyn_k - (phi_k.T @ theta)[0, 0]

        # Ganho de Kalman (K)
        K = (P @ phi_k) / (lambda_rls + phi_k.T @ P @ phi_k)

        # Atualização (theta e P)
        theta = theta + K * e_k
        P = (P - K @ phi_k.T @ P) / lambda_rls

    # Conversão Discreto (ARX) -> Físico (RC)
    a1, b0, b1 = theta.flatten()

    if 0 < a1 < 1 and (1 - a1) != 0:
        tau_est = -Ts / np.log(a1)
        r0_est = b0
        r1_est = (b1 + r0_est * a1) / (1 - a1)
        c1_est = tau_est / r1_est if r1_est != 0 else 0
    else:
        r0_est, r1_est, c1_est = 0, 0, 0

    R0_rls[k] = r0_est
    R1_rls[k] = r1_est
    C1_rls[k] = np.clip(c1_est, 1.0, 50000.0)

    # Prepara próximo passo
    V_dyn_prev = V_dyn_k
    u_prev = u[k]

# ======================================================================
# 4. PLOTAGEM DOS RESULTADOS
# ======================================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot R0
axs[0].plot(time, R0_rls, 'b', label='R0 Estimado (RLS)', alpha=0.8)
axs[0].axhline(R0_ref, color='r', linestyle='--', label='R0 Referência Offline')
axs[0].set_ylabel('R0 (Ohms)')
axs[0].legend()
axs[0].grid(True)

# Plot R1
axs[1].plot(time, R1_rls, 'g', label='R1 Estimado (RLS)', alpha=0.8)
axs[1].axhline(R1_ref, color='r', linestyle='--', label='R1 Referência Offline')
axs[1].set_ylabel('R1 (Ohms)')
axs[1].legend()
axs[1].grid(True)

# Plot C1
axs[2].plot(time, C1_rls, 'k', label='C1 Estimado (RLS)', alpha=0.8)
axs[2].axhline(C1_ref, color='r', linestyle='--', label='C1 Referência Offline')
axs[2].set_ylabel('C1 (Farads)')
axs[2].set_xlabel('Tempo (s)')
axs[2].legend()
axs[2].grid(True)

plt.suptitle('Estimação Online (RLS) vs Identificação Offline')
plt.tight_layout()
plt.show()