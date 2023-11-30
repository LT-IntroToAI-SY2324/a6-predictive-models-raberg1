import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = datasets.load_diabetes(as_frame=True)
x = data.frame["bmi"].values
# print(x)
y = data.target.values
# print(y)
# x = x[:,np.newaxis, 3]
x = x.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)
# print(f"xtrain {xtrain}")
# print(f"xtest {xtest}")
# print(f"ytrain {ytrain}")
# print(f"ytest {ytest}")

model = linear_model.LinearRegression().fit(xtrain, ytrain)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)

print(coef, intercept)

prediction = model.predict(xtest)

print(f"Coefficient: {coef}")
print(f"Mean squared error: {mean_squared_error(ytest, prediction)}")
print(f"Coefficient of determiniation: {r2_score(ytest, prediction)}")

# Plot the points 
plt.scatter(xtest, ytest, c="red")
plt.scatter(xtrain, ytrain, c="purple")
# plt.plot(xtrain, ytrain, c="blue", linewidth=3)
plt.plot(xtest, coef*xtest + intercept, c="r", label="Line of Best Fit")

plt.xlabel("bmi")
plt.ylabel("quantitative measure of disease progression one year after baseline")
plt.title("quantitative measure of disease progression one year after baseline by bmi")

plt.legend()
plt.xticks()
plt.yticks()
plt.show()