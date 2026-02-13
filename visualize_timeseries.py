import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Directory containing the parquet files
dir_path = 'data/2024-06-01'

# Get all parquet files in the directory
parquet_files = glob.glob(os.path.join(dir_path, '*.parquet'))

# Create results directory
results_dir = 'results/data_plot'
os.makedirs(results_dir, exist_ok=True)

# Create three figures for the three variables
fig_freq, ax_freq = plt.subplots(figsize=(15, 8))
fig_angle, ax_angle = plt.subplots(figsize=(15, 8))
fig_magnitude, ax_magnitude = plt.subplots(figsize=(15, 8))

# Loop through each file and plot the data
for file_path in parquet_files:
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        # Extract sensor ID from filename (e.g., '620-UsIlChicago620-20240101.parquet' -> '620')
        sensor_id = os.path.basename(file_path).split('-')[0]
        
        # Plot Frequency
        ax_freq.plot(df.index, df['Frequency'], alpha=0.5, label=sensor_id)
        
        # Plot VoltageAngle
        ax_angle.plot(df.index, df['VoltageAngle'], alpha=0.5, label=sensor_id)
        
        # Plot VoltageMagnitude
        ax_magnitude.plot(df.index, df['VoltageMagnitude'], alpha=0.5, label=sensor_id)
        
        print(f"Plotted data from {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Set titles and labels
ax_freq.set_title('Frequency Time Series for All Sensors')
ax_freq.set_xlabel('Time')
ax_freq.set_ylabel('Frequency')
ax_freq.set_ylim(59.8, 60.2)  # Focus on meaningful range around 60 Hz
ax_freq.legend(loc='upper right', fontsize='small', ncol=5)

ax_angle.set_title('Voltage Angle Time Series for All Sensors')
ax_angle.set_xlabel('Time')
ax_angle.set_ylabel('Voltage Angle')
ax_angle.set_ylim(-10, 10)  # Reasonable range for voltage angle
ax_angle.legend(loc='upper right', fontsize='small', ncol=5)

ax_magnitude.set_title('Voltage Magnitude Time Series for All Sensors')
ax_magnitude.set_xlabel('Time')
ax_magnitude.set_ylabel('Voltage Magnitude')
ax_magnitude.set_ylim(110, 130)  # Reasonable range for voltage magnitude
ax_magnitude.legend(loc='upper right', fontsize='small', ncol=5)

# Save the figures
fig_freq.savefig(os.path.join(results_dir, 'frequency_timeseries.png'), dpi=300, bbox_inches='tight')
fig_angle.savefig(os.path.join(results_dir, 'voltage_angle_timeseries.png'), dpi=300, bbox_inches='tight')
fig_magnitude.savefig(os.path.join(results_dir, 'voltage_magnitude_timeseries.png'), dpi=300, bbox_inches='tight')

print("Plots saved in results/data_plot directory.")
print("Finished plotting all data.")