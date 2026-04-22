import numpy as np
import matplotlib.pyplot as plt

 
R = 8.314   

 
D0 = 1e-7      
EL = 40000      

 
T_min = 300      
phi = 0.1       

 
L = 1e-3         
Nx = 50
dx = L / Nx

 
dt = 0.01
Nt = 2000

 
C0 = 1.0
 
traps = [
    {"E": 60000, "density": 0.5},
    {"E": 90000, "density": 0.3}
]

 
def temperature(t):
    return T_min + phi * t

def diffusion_coefficient(T):
    return D0 * np.exp(-EL / (R * T))

def trap_equilibrium(theta_L, T, traps):
    total_trap = 0
    for trap in traps:
        K = np.exp(-trap["E"] / (R * T))
        theta_T = (theta_L * K) / (1 + (K - 1) * theta_L)
        total_trap += trap["density"] * theta_T
    return total_trap

 
x = np.linspace(0, L, Nx)
C_L = np.ones(Nx) * C0

temperatures = []
fluxes = []

 
for step in range(Nt):

    t = step * dt
    T = temperature(t)
    D = diffusion_coefficient(T)

    C_new = C_L.copy()

    for i in range(1, Nx-1):

 
        diffusion = D * (C_L[i+1] - 2*C_L[i] + C_L[i-1]) / dx**2

       
        theta_L = C_L[i]
        trap_effect = trap_equilibrium(theta_L, T, traps)

 
        C_new[i] = C_L[i] + dt * (diffusion - trap_effect * 0.01)

    C_new[0] = 0
    C_new[-1] = 0

    C_L = C_new

    flux = -D * (C_L[1] - C_L[0]) / dx

    temperatures.append(T)
    fluxes.append(flux)

plt.figure(figsize=(8,5))
plt.plot(temperatures, fluxes)
plt.xlabel("Temperatura (K)")
plt.ylabel("Fluxo de Hidrogênio")
plt.title("Simulação TDS (simplificada)")
plt.grid()
plt.show()