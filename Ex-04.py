import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(0)
X = np.sort(np.random.rand(30))
y = np.cos(1.5 * np.pi * X) + np.random.randn(30) * 0.1

degrees = [1, 4, 15]

plt.figure(figsize=(12,4))

for i, d in enumerate(degrees):
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X.reshape(-1,1))
    model = LinearRegression().fit(X_poly, y)

    X_test = np.linspace(0,1,100)
    X_test_poly = poly.transform(X_test.reshape(-1,1))

    plt.subplot(1,3,i+1)
    plt.scatter(X,y)
    plt.plot(X_test, model.predict(X_test_poly))
    plt.title(f"Degree {d}")

plt.show()
