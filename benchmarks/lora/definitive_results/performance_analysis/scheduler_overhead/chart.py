import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (13, 4)  # Adjusted for balanced horizontal layout
    })

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

# Load your data
csv_file = 'metrics_data_aggregated_rate.csv'  # Update with your file path
df = pd.read_csv(csv_file)

# Calculate scheduling time per request
# df['scheduling_time_per_request'] = df['total_scheduling_time'] / df['completed_requests']
df['scheduling_time_per_step'] = df['total_scheduling_time'] / df['steps']
# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
df.head()

csv_file = 'metrics_data_adapter_rates.csv'
adapter_rate_df = pd.read_csv(csv_file)

# Calculate scheduling time per request
adapter_rate_df['scheduling_time_per_step'] = adapter_rate_df['total_scheduling_time'] / adapter_rate_df['steps']
combined = pd.concat([df, adapter_rate_df], ignore_index=True)
df=combined.copy()

# Calculate the proportion of scheduling time per step to ITL
df['scheduling_proportion'] = ((df['scheduling_time_per_step'] * 1000) / df['mean_itl_ms']) * 100

# Calculate the mean scheduling proportion across all rates for each GPU LoRA and CPU LoRA combination
df_avg = df.groupby(['gpu_loras', 'cpu_loras'])['scheduling_proportion'].mean().reset_index()
df_avg['scheduling_proportion'] = df_avg['scheduling_proportion'].round(2)
# Sort the DataFrame by cpu_loras for proper line plotting
df_avg = df_avg.sort_values(by=['gpu_loras', 'cpu_loras'])

# Create the figure
fig, ax = plt.subplots()

# Get unique GPU LoRAs values
gpu_loras_values = sorted(df_avg['gpu_loras'].unique())
gpu_loras_values = [8, 32, 64, 128, 256, 448]

# Plot a line for each GPU LoRA value
for i, gpu_lora in enumerate(gpu_loras_values):
    # Filter data for current GPU LoRA
    gpu_data = df_avg[df_avg['gpu_loras'] == gpu_lora]

    # Sort by CPU LoRAs to ensure proper line connection
    gpu_data = gpu_data.sort_values('cpu_loras')

    # Plot the line - let matplotlib assign colors automatically from default cycle
    plt.plot(gpu_data['cpu_loras'], gpu_data['scheduling_proportion'],
             'o-',
             label=fr"$A_{{\mathrm{{max}}}} = {gpu_lora}$")

    # Add data labels (optional)
    '''for _, row in gpu_data.iterrows():
        plt.annotate(f"{row['scheduling_proportion']:.2f}",
                     (row['cpu_loras'], row['scheduling_proportion']),
                     textcoords="offset points", xytext=(0, 7),
                     ha='center', fontsize=9)'''

# Configure the plot
plt.xlabel('adapters (#)')
# plt.ylabel('scheduler time (%)')
plt.title('Relative scheduler time (%)')
ax.set_xlim(50, 550)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.45))

# Tight layout
plt.tight_layout()

output_path = ''
plot_path = os.path.join(output_path, f'scheduling_proportion_by_gpu_cpu_loras.pdf')
plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=400)