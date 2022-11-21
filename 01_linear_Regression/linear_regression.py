import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tf_utils.dummyData import regression_data


def main():
    x,y = regression_data()
    print(x.shape) # (100,)
    x = x.reshape(-1, 1)
    print(x.shape) # (100,1)

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

    lin_regression_model = LinearRegression()
    lin_regression_model.fit(x_train, y_train)

    score = lin_regression_model.score(x_test, y_test)
    print(f"Linear Regression Score: {score}")
    print(f"Coefs: {lin_regression_model.coef_}")
    print(f"Intersept: {lin_regression_model.intercept_}")

    y_pred = lin_regression_model.predict(x_test)

    plt.scatter(x,y)
    plt.plot(x_test, y_pred)
    plt.show()

if __name__ == "__main__":
    main()
