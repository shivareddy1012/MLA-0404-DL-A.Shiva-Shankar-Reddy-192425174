from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load only 2 classes
wine = load_wine()
X = wine.data[wine.target != 2]
y = wine.target[wine.target != 2]

# Train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
y_pred = DecisionTreeClassifier().fit(X_train, y_train).predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='PuBuGn')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
