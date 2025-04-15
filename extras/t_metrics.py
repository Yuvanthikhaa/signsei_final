import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data
epochs = np.arange(1, 21)  # 20 epochs
train_accuracy = np.linspace(0.6, 0.95, 20) + np.random.normal(0, 0.02, 20)
val_accuracy = np.linspace(0.55, 0.92, 20) + np.random.normal(0, 0.02, 20)
train_loss = np.linspace(1.0, 0.1, 20) + np.random.normal(0, 0.05, 20)
val_loss = np.linspace(1.2, 0.15, 20) + np.random.normal(0, 0.05, 20)

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)
plt.show()

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)
plt.show()
