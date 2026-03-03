from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data[:, 0].reshape(-1,1)   
y = iris.data[:, 1]                
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()
