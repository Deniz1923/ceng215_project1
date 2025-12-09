import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Parameters [cite: 98] ---
R = 50e3          # 50 kOhm
C = 1e-6          # 1 uF
v_c_0 = 3.0       # Initial Voltage (V)

# Simulation Settings
t_start = 0.0
t_end = 2.0       # Simulate for 2 seconds (roughly 3 periods of 10rad/s)
h = 1e-5          # Step size (10us) for stability

t = np.arange(t_start, t_end, h)
v_out = np.zeros_like(t)
v_out[0] = v_c_0

# --- 2. Define Component Models ---

def V_source(t):
    """Input Voltage: 10 sin(10t) [cite: 98]"""
    return 10 * np.sin(10 * t)

def get_diode_current(v_d):
    """
    X-Diode Piecewise Characteristic.
    Input: Diode Voltage v_d (Volts)
    Output: Diode Current (Amps) -> NOTE: Formulas are in mA!
    """
    i_mA = 0.0
    
    if v_d < 0:
        # Region 1: i_D = 0.1 * v_D
        i_mA = 0.1 * v_d
        
    elif 0 <= v_d <= 3:
        # Region 2: i_D = (2/3) * v_D
        i_mA = (2.0/3.0) * v_d
        
    else: # v_d > 3
        # Region 3: i_D = (v_D - 3)^2 + 2
        i_mA = (v_d - 3.0)**2 + 2.0
        
    return i_mA * 1e-3  # Convert mA to Amps

# --- 3. Euler Simulation Loop ---

for k in range(len(t) - 1):
    # Current state
    v_o_k = v_out[k]
    t_k = t[k]
    
    # Calculate diode voltage drop: v_D = V_source - V_output
    v_source_val = V_source(t_k)
    v_diode_drop = v_source_val - v_o_k
    
    # Calculate currents (in Amps)
    i_d = get_diode_current(v_diode_drop)
    i_r = v_o_k / R
    
    # KCL at output node: i_C = i_D - i_R
    i_c = i_d - i_r
    
    # Derivative: dVo/dt = i_C / C
    dv_dt = i_c / C
    
    # Euler Update
    v_out[k+1] = v_o_k + h * dv_dt

# --- 4. Plotting [cite: 100] ---

plt.figure(figsize=(10, 6))

# Plot Vs and Vo on the same axes
plt.plot(t, V_source(t), label='$V_s(t)$ Input', linestyle='--', color='gray', alpha=0.7)
plt.plot(t, v_out, label='$V_o(t)$ Output', color='blue', linewidth=2)

plt.title('Project #1: X-Diode with Parallel RC Load')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()