import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters  ---
R = 50e3       # Resistance in Ohms (50 kOhm)
C = 1e-6       # Capacitance in Farads (1 uF)
V_c_init = 3.0 # Initial capacitor voltage in Volts

# --- Simulation Settings ---
t_start = 0
t_end = 2.0    # Simulate for 2 seconds (enough to see behaviors)
dt = 1e-5      # Time step (must be small for stability: dt << RC = 0.05s)

# Create time array
t = np.arange(t_start, t_end, dt)
n_steps = len(t)

# Initialize arrays to store voltages
Vs = np.zeros(n_steps) # Source Voltage
Vo = np.zeros(n_steps) # Output Voltage

# Set Initial Condition 
Vo[0] = V_c_init

# --- Task (a): Define the X-diode Function [cite: 32, 33, 36] ---
def get_diode_current(v_d):
    """
    Calculates diode current based on voltage drop v_d.
    Returns current in AMPS (converts from mA defined in text).
    """
    # Initialize current (scalar or array)
    i_d_mA = 0.0
    
    if v_d < 0:
        # i_D = 0.1 * V (Assuming linear leakage based on notation)
        i_d_mA = 0.1 * v_d
    elif 0 <= v_d <= 3:
        # i_D = (2/3) * v_D
        i_d_mA = (2/3) * v_d
    else: # v_d > 3
        # i_D = (v_D - 3)^2 + 2
        i_d_mA = (v_d - 3)**2 + 2
        
    return i_d_mA * 1e-3 # Convert mA to Amps

# --- Task (b): Simulation Loop using Euler Method  ---
for k in range(n_steps - 1):
    # 1. Current time and state
    current_time = t[k]
    
    # Calculate Source Voltage: Vs(t) = 10 sin(10t) 
    Vs[k] = 10 * np.sin(10 * current_time)
    
    # 2. Calculate Diode Voltage Drop
    v_diode_drop = Vs[k] - Vo[k]
    
    # 3. Calculate Diode Current (in Amps)
    i_diode = get_diode_current(v_diode_drop)
    
    # 4. Calculate Resistor Current (Ohm's law)
    i_resistor = Vo[k] / R
    
    # 5. KCL: Current into Capacitor = Diode Current - Resistor Current
    i_cap = i_diode - i_resistor
    
    # 6. State Equation: dVo/dt = i_cap / C
    dVo_dt = i_cap / C
    
    # 7. Euler Update: V(next) = V(current) + derivative * dt
    Vo[k+1] = Vo[k] + dVo_dt * dt

# Fill in the last Vs value for plotting completeness
Vs[-1] = 10 * np.sin(10 * t[-1])

# --- Task (c): Plotting [cite: 39] ---
plt.figure(figsize=(10, 6))
plt.plot(t, Vs, label='$V_s(t)$ (Source)', linestyle='--', alpha=0.7)
plt.plot(t, Vo, label='$V_o(t)$ (Output)', linewidth=2)

plt.title('Simulation of X-Diode Circuit Response')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()