import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Circuit Parameters ---
R = 50e3      # Resistance = 50 kOhms 
C = 1e-6      # Capacitance = 1 uF 
dt = 1e-5     # Time step (small enough for accuracy)
t_end = 1.0   # Simulate for 1 second

# Initial Conditions
# Vc(0) = 3 Volts 
initial_voltage = 3.0 

# --- 2. Define Helper Functions ---

def get_source_voltage(t):
    # Vs(t) = 10 * sin(10t) 
    return 10 * np.sin(10 * t)

def get_diode_current(v_d):
    """
    Returns diode current in Amps based on the piecewise model.
    Note: The problem specifies values in mA, so we multiply by 1e-3.
    """
    if v_d < 0:
        i_mA = 0.1 * v_d  # 
    elif 0 <= v_d <= 3:
        i_mA = (2/3) * v_d # 
    else: # v_d > 3
        i_mA = (v_d - 3)**2 + 2 # [cite: 33]
        
    return i_mA * 1e-3 # Convert mA to Amps

# --- 3. Run Simulation (Euler Method) ---

# Create time array
time = np.arange(0, t_end, dt)
n_steps = len(time)

# Initialize arrays to store results
Vo = np.zeros(n_steps)
Vs = np.zeros(n_steps)

# Set initial condition
Vo[0] = initial_voltage

for k in range(n_steps - 1):
    t_curr = time[k]
    
    # 1. Get current source voltage
    Vs[k] = get_source_voltage(t_curr)
    
    # 2. Determine voltage across the diode (Vs - Vo)
    v_diode = Vs[k] - Vo[k]
    
    # 3. Calculate currents
    i_d = get_diode_current(v_diode) # Diode current
    i_r = Vo[k] / R                  # Resistor current
    i_c = i_d - i_r                  # Capacitor current (KCL)
    
    # 4. Calculate derivative dVo/dt = iC / C
    dVo_dt = i_c / C
    
    # 5. Update Vo using Euler's formula: y_new = y_old + slope * step
    Vo[k+1] = Vo[k] + dVo_dt * dt

# Fill in the last Vs value for plotting consistency
Vs[-1] = get_source_voltage(time[-1])

# --- 4. Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(time, Vs, label='$V_s(t)$ Source Voltage', linestyle='--', color='blue', alpha=0.6)
plt.plot(time, Vo, label='$V_o(t)$ Output Voltage', color='red', linewidth=2)

plt.title('Transient Response of X-Diode Circuit')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(loc='upper right')
plt.grid(True)

# Mark the start voltage to verify IC
plt.plot(0, initial_voltage, 'ko', label='Start (3V)') 

plt.show()