import matplotlib.pyplot as plt
import numpy as np

# Define the categories (metrics)
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
N = len(categories)

# Dummy data for models (make them realistic)
signsei = [0.97, 0.96, 0.95, 0.96]
mediapipe = [0.89, 0.88, 0.87, 0.88]
cnn = [0.91, 0.90, 0.89, 0.90]
random_forest = [0.88, 0.87, 0.85, 0.86]

# Repeat the first value to close the radar chart
values_signsei = signsei + [signsei[0]]
values_mediapipe = mediapipe + [mediapipe[0]]
values_cnn = cnn + [cnn[0]]
values_random_forest = random_forest + [random_forest[0]]

# Compute angle for each category
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Initialize radar chart
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Plot each model
ax.plot(angles, values_signsei, label='SignSei', linewidth=2, linestyle='solid')
ax.fill(angles, values_signsei, alpha=0.25)

ax.plot(angles, values_mediapipe, label='MediaPipe', linewidth=2, linestyle='solid')
ax.fill(angles, values_mediapipe, alpha=0.25)

ax.plot(angles, values_cnn, label='CNN', linewidth=2, linestyle='solid')
ax.fill(angles, values_cnn, alpha=0.25)

ax.plot(angles, values_random_forest, label='Random Forest', linewidth=2, linestyle='solid')
ax.fill(angles, values_random_forest, alpha=0.25)

# Add labels for each category
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

# Add title
plt.title('Model Comparison Radar Chart', size=15, y=1.08)

# Save and show plot
plt.tight_layout()
plt.savefig('model_comparison_radar_chart.png', dpi=300)
plt.show()
