import numpy as np
import matplotlib.pyplot as plt


# It is for series RL circuit with an X-Diode.

# --- 1. System Parameters ---
R = 50.0          # Resistance in Ohms
L = 10e-3         # Inductance in Henries (10mH)
i0 = 0.0          # Initial current (Amps)

# Simulation settings
t_start = 0.0
t_end = 0.1       # Simulate for 100ms
h = 1e-5          # Step size (choose small for stability per [cite: 73])

# Time grid
t = np.arange(t_start, t_end, h)
i_out = np.zeros_like(t) # State variable array
i_out[0] = i0

# --- 2. Define Helper Functions ---

def V_source(t):
    """Sinusoidal Input Voltage: 10 sin(100t)"""
    return 10 * np.sin(100 * t)

def get_v_diode(i_val_amps):
    """
    Inverted X-Diode Model: Returns Voltage given Current.
    Note: The project defines current in mA and Voltage in V[cite: 16, 23].
    We must convert Amps -> mA for calculation, then return Volts.
    """
    i_mA = i_val_amps * 1000.0  # Convert to mA for the formula
    
    v_d = 0.0
    
    # Apply inverted piecewise logic 
    if i_mA < 0:
        # Region 1: i = 0.1 * v  -> v = 10 * i
        v_d = 10.0 * i_mA
        
    elif 0 <= i_mA <= 2.0:
        # Region 2: i = (2/3) * v -> v = 1.5 * i
        v_d = 1.5 * i_mA
        
    else: # i_mA > 2.0
        # Region 3: i = (v - 3)^2 + 2 -> v = 3 + sqrt(i - 2)
        # We take positive root because voltage increases with current
        v_d = 3.0 + np.sqrt(i_mA - 2.0)
        
    return v_d

# --- 3. Euler Simulation Loop ---

for k in range(len(t) - 1):
    # Current state
    i_k = i_out[k]
    t_k = t[k]
    
    # 1. Calculate voltages across components
    v_s = V_source(t_k)
    v_d = get_v_diode(i_k)
    v_r = R * i_k
    
    # 2. Compute Derivative di/dt = (Vs - Vr - Vd) / L
    # Based on KVL: Vs = Vd + Vr + V_L  ->  V_L = L * di/dt
    di_dt = (v_s - v_r - v_d) / L
    
    # 3. Euler Update: x[k+1] = x[k] + h * f(x,t) [cite: 69]
    i_out[k+1] = i_k + h * di_dt

# --- 4. Plotting Results ---

# Calculate voltages for plotting (post-processing)
v_source_arr = V_source(t)
v_diode_arr = np.array([get_v_diode(i) for i in i_out])
v_resistor_arr = R * i_out

plt.figure(figsize=(10, 6))

# Plot 1: Source vs Diode Voltage
plt.subplot(2, 1, 1)
plt.plot(t, v_source_arr, label='$V_s(t)$ (Source)', linestyle='--')
plt.plot(t, v_diode_arr, label='$V_{Diode}(t)$', linewidth=2)
plt.title('Series RL with X-Diode: Voltages')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Output Current (State Variable)
plt.subplot(2, 1, 2)
plt.plot(t, i_out * 1000, label='$i_L(t)$', color='orange', linewidth=2) # Plot in mA
plt.xlabel('Time (s)')
plt.ylabel('Current (mA)')
plt.title('Inductor Current (State Variable)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()