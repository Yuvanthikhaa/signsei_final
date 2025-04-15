import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Dummy ground truth labels and predictions
true_labels = np.random.choice(['A', 'B', 'C', 'D', 'E', 'Thank You', '0', '1', '2', '3'], 100)
predicted_labels = np.random.choice(['A', 'B', 'C', 'D', 'E', 'Thank You', '0', '1', '2', '3'], 100)

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=['A', 'B', 'C', 'D', 'E', 'Thank You', '0', '1', '2', '3'])

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B', 'C', 'D', 'E', 'Thank You', '0', '1', '2', '3'], yticklabels=['A', 'B', 'C', 'D', 'E', 'Thank You', '0', '1', '2', '3'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png', dpi=300)
plt.show()
