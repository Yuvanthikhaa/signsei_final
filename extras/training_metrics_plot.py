import pickle
import matplotlib.pyplot as plt

with open('training_history.pickle', 'rb') as file:
    history = pickle.load(file)

plt.figure(figsize=(10, 5))
plt.plot(history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Training Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='s', color='red')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Training metrics plots saved as 'training_accuracy_plot.png' and 'training_loss_plot.png'")
