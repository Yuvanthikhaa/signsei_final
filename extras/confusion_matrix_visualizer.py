import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
with open('./model.p', 'rb') as file:
    model_dict = pickle.load(file)
    model = model_dict['model']
with open('./data.pickle', 'rb') as file:
    data = pickle.load(file)

X = np.array(data['data'])
y = np.array(data['labels'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y)))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Sign Language Recognition')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved as 'confusion_matrix.png'")
