import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Datos de tu último test (7 audios)
# Reales: 3 gases, 4 sueno
# Predichos: Gases acertó 3. Sueno acertó 2 (otros 2 se fueron a gases)
y_true = ["gases", "gases", "gases", "sueno", "sueno", "sueno", "sueno"]
y_pred = ["gases", "gases", "gases", "gases", "gases", "sueno", "sueno"]

labels = ["gases", "sueno"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusión: Modelo Avanzado (71.4% Accuracy)')
plt.ylabel('Clase Real')
plt.xlabel('Predicción de la IA')
plt.show()