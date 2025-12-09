import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Parameters ---
R = 4.0      # Resistance (Ohms) (Using > 2*sqrt(L/C) prevents oscillation, < allows it)
L = 1.0      # Inductance (Henrys)
C = 0.25     # Capacitance (Farads)
A = 10.0     # Step input amplitude (Volts)

# --- 2. Simulation Settings ---
# Note: RLC circuits can be oscillatory. As per source [33], 
# we need a small time step 'h' for stability.
tau = np.sqrt(L*C)  # Natural period approximation for scaling
T_end = 20.0        # Total simulation time
h = 0.01            # Step size (try making this smaller if the plot looks jagged)

# Time grid
t_values = np.arange(0, T_end, h)
N = len(t_values)

# --- 3. Define the State Derivatives ---
# Based on: dot_x1 = (1/C)*x2, dot_x2 = (-R*x2 - x1 + vs)/L
def get_derivatives(t, current_state, u_val):
    v_c = current_state[0]  # x1
    i_l = current_state[1]  # x2
    
    # State Space Equations
    dv_c_dt = (1/C) * i_l
    di_l_dt = (1/L) * (u_val - R * i_l - v_c)
    
    return np.array([dv_c_dt, di_l_dt])

# --- 4. The Euler Loop ---
# Initialize state vector x = [v_C, i_L]
# Assuming zero initial conditions: capacitor uncharged, no current.
x = np.zeros((N, 2)) 

for k in range(N - 1):
    # Define Input Source u(t)
    # Here we use a Step Input: u(t) = A for t >= 0
    u_k = A 
    
    # Current state at step k
    x_k = x[k]
    
    # Calculate derivatives (slope)
    f = get_derivatives(t_values[k], x_k, u_k)
    
    # Update state: x[k+1] = x[k] + h * f
    # (Forward Euler Method )
    x[k+1] = x_k + h * f

# --- 5. Plotting ---
v_c_result = x[:, 0]
i_l_result = x[:, 1]

plt.figure(figsize=(10, 6))

# Plot Capacitor Voltage
plt.subplot(2, 1, 1)
plt.plot(t_values, v_c_result, label=r'$v_C(t)$ (Voltage)', color='blue')
plt.plot(t_values, [A]*N, 'r--', label='Input Step (10V)', alpha=0.5)
plt.title('RLC Circuit Simulation (Forward Euler)')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Inductor Current
plt.subplot(2, 1, 2)
plt.plot(t_values, i_l_result, label=r'$i_L(t)$ (Current)', color='orange')
plt.ylabel('Current (A)')
plt.xlabel('Time (s)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()