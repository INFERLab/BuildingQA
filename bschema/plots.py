import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('bschema/full/stats.csv')

# Add the threshold values

thresholds = [0,0,0,0,0,0, 30,30,30,30,30,30, 50,50,50,50,50,50, 70,70,70,70,70,70, 100,100,100,100,100,100]
df['threshold'] = thresholds

# Get unique file names
files = df['file_name'].unique()

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: bschema_length vs threshold
ax1 = axes[0]
for file in files:
    file_data = df[df['file_name'] == file]
    ax1.plot(file_data['threshold'], file_data['bschema_length'], 
             marker='o', label=file, linewidth=2)

ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('BSchema Length', fontsize=12)
ax1.set_title('BSchema Length by Threshold', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: iterations vs threshold
ax2 = axes[1]
for file in files:
    file_data = df[df['file_name'] == file]
    ax2.plot(file_data['threshold'], file_data['iterations'], 
             marker='s', label=file, linewidth=2)

ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('Iterations', fontsize=12)
ax2.set_title('Iterations by Threshold', fontsize=14, fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: runtime vs threshold
ax3 = axes[2]
for file in files:
    file_data = df[df['file_name'] == file]
    ax3.plot(file_data['threshold'], file_data['runtime'], 
             marker='^', label=file, linewidth=2)

ax3.set_xlabel('Threshold', fontsize=12)
ax3.set_ylabel('Runtime (seconds)', fontsize=12)
ax3.set_title('Runtime by Threshold', fontsize=14, fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')

# for metric, ylabel, title in [
#     ('bschema_length', 'BSchema Length', 'BSchema Length by Threshold'),
#     ('iterations', 'Iterations', 'Iterations by Threshold'),
#     ('runtime', 'Runtime (seconds)', 'Runtime by Threshold')
# ]:
#     plt.figure(figsize=(10, 6))
#     for file in files:
#         file_data = df[df['file_name'] == file]
#         plt.plot(file_data['threshold'], file_data[metric], 
#                 marker='o', label=file, linewidth=2, markersize=8)
    
#     plt.xlabel('Threshold', fontsize=12)
#     plt.ylabel(ylabel, fontsize=12)
#     plt.title(title, fontsize=14, fontweight='bold')
#     plt.legend(fontsize=9)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'{metric}_by_threshold.png', dpi=300, bbox_inches='tight')