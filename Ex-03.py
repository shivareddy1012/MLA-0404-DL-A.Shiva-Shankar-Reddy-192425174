from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

y_pred = RandomForestClassifier().fit(X_train, y_train).predict(X_test)

sns.heatmap(confusion_matrix(y_test, y_pred))
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
