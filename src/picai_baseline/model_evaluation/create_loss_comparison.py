import matplotlib.pyplot as plt
import numpy as np

# Data for epsilon=3 (with DP)
epsilon_3_rounds = list(range(1, 41))
epsilon_3_losses = [659199.92, 483229.64, 396499.39, 353337.89, 281478.39, 229222.60, 198675.05, 159644.99,
                   132996.99, 122190.27, 104612.81, 95236.17, 82764.78, 70453.97, 61968.95, 60601.66,
                   53919.11, 53777.78, 49602.87, 41775.70, 39944.98, 38570.83, 34862.08, 36628.49,
                   26842.59, 36512.28, 29786.93, 23624.42, 23163.22, 23513.80, 24629.21, 28885.76,
                   19360.37, 31811.58, 23601.03, 24823.02, 23602.06, 26641.22, 17912.43, 27628.48]

# Data for no-DP
no_dp_rounds = list(range(1, 41))
no_dp_losses = [44521.59, 17405.60, 17271.18, 16068.81, 16576.60, 14800.46, 16625.16, 13739.49,
               11247.19, 10910.61, 12837.88, 10524.55, 9206.30, 8482.80, 7560.35, 6039.88,
               7413.37, 5540.50, 6479.56, 6643.26, 6044.38, 5591.84, 5142.19, 5925.60,
               5244.84, 5561.54, 5125.18, 4907.27, 4485.08, 5155.49, 5154.33, 5371.27,
               6109.18, 5010.37, 4863.21, 3927.22, 4171.48, 4067.29, 4016.18, 4546.30]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot both lines
plt.plot(epsilon_3_rounds, epsilon_3_losses, 'o-', linewidth=3, markersize=6,
         color='#ff7f0e', label='ε = 3 (With DP)', alpha=0.8)
plt.plot(no_dp_rounds, no_dp_losses, 'o-', linewidth=3, markersize=6,
         color='#1f77b4', label='No-DP (Without DP)', alpha=0.8)

# Formatting
plt.xlabel('Training Round', fontsize=14, fontweight='bold')
plt.ylabel('Training Loss', fontsize=14, fontweight='bold')
plt.title('Training Loss Comparison: ε=3 vs No Differential Privacy',
          fontsize=16, fontweight='bold', pad=20)

# Grid and styling
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Format y-axis with commas
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Set limits and ticks
plt.xlim(1, 40)
plt.xticks(range(0, 41, 5), fontsize=11)
plt.yticks(fontsize=11)

# Tight layout
plt.tight_layout()

# Save as high-quality PNG
plt.savefig('training_loss_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Display the plot
plt.show()

print("Plot saved as 'training_loss_comparison.png'")
print("File saved with 300 DPI for publication quality")