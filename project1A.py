import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS
# ==========================================
# WARNING: The PDF does not specify L and C values.
# These specific values (10mH, 100uF) are standard placeholders used
# to make the simulation runnable. 
# *** IF YOU HAVE SPECIFIC VALUES FROM CLASS, CHANGE THEM HERE. ***
L = 0.010   # 10 mH
C = 0.0001  # 100 uF

# Time setup
dt = 1e-6       # Time step (1 microsecond)
t_final = 0.015 # Duration (15 milliseconds)

# We make the simulation space with numpy's linspace function   
time_steps = int(t_final / dt)
t = np.linspace(0, t_final, time_steps)

# Initialize State Arrays
Vc = np.zeros(time_steps)
iL = np.zeros(time_steps)
Vl = np.zeros(time_steps) # This is voltage across inductor

# Initial Conditions [cite: 36, 37]
Vc[0] = 3.0  # Volts
iL[0] = 0.0  # Amps

# ==========================================
# 2. X-DIODE MODEL (Piecewise)
# ==========================================
def get_diode_voltage(current_amps):
    """
    Returns diode voltage based on current using the text descriptions
    and graph from the PDF.
    """
    # Convert to mA for easier mapping to Figure 2
    i_mA = current_amps * 1000.0
    
    # If current is zero or negative, the diode is effectively open.
    # We return 0V drop here for the calculation, but the logic in the
    # loop handles the "blocking" of negative current.
    if i_mA <= 0:
        return 0.25 # Return 'turn on' voltage or 0.
    
    # MODEL CONSTRUCTION [cite: 16, 30]
    # The text explicitly gives us regions to ignore the non-uniform axis:
    # 1. "vD = 0.25V when 0 < iD < 5mA"
    # 2. "vD = 0.60V when 15 < iD < 17.5mA"
    # 3. We assume linear connections between these known regions.
    # 4. Graph ends at 25mA, 0.9V [cite: 26, 18]
    
    current_points = [0,    5,    15,   17.5, 25]  # x-axis (mA)
    voltage_points = [0.25, 0.25, 0.60, 0.60, 0.90] # y-axis (V)
    
    # Linear interpolation between these "truth" points
    v_d = np.interp(i_mA, current_points, voltage_points)
    return v_d

# ==========================================
# 3. SIMULATION LOOP (Euler Method)
# ==========================================
for n in range(time_steps - 1):
    
    # A. Get current Diode Voltage Drop
    v_diode = get_diode_voltage(iL[n])
    
    # B. Calculate Derivatives (State Equations)
    # dVc/dt = -iL / C
    dVc = (-iL[n] / C)
    
    # diL/dt = (Vc - Vdiode) / L
    diL = (Vc[n] - v_diode) / L
    
    # C. Update States (Euler Integration) 
    Vc[n+1] = Vc[n] + dVc * dt
    iL[n+1] = iL[n] + diL * dt
    
    # D. Enforce Diode Physical Constraint [cite: 17]
    # "Diode does not allow negative currents"
    if iL[n+1] < 0:
        iL[n+1] = 0
        
    # E. Calculate V_l(t) for plotting
    # V_l is the node between diode and inductor.
    # Since V_l is just the voltage across the inductor:
    # V_l = Vc - V_diode
    Vl[n] = Vc[n] - v_diode

# Fix last point for Vl
Vl[-1] = Vc[-1] - get_diode_voltage(iL[-1])

# ==========================================
# 4. PLOTS & RESULTS
# ==========================================
plt.figure(figsize=(10, 8))

# Plot Vc(t)
plt.subplot(2, 1, 1)
plt.plot(t * 1000, Vc, 'b', label='$V_c(t)$')
plt.title('Capacitor Voltage $V_c(t)$')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Plot Vl(t)
plt.subplot(2, 1, 2)
plt.plot(t * 1000, Vl, 'r', label='$V_l(t)$') # [cite: 39]
plt.title('Inductor Node Voltage $V_l(t)$')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Final Value Result [cite: 40]
print(f"--- RESULTS ---")
print(f"Initial Capacitor Voltage: {Vc[0]} V")
print(f"Final Capacitor Voltage:   {Vc[-1]:.4f} V")
print(f"(Note: The final voltage depends on L/C values used)")