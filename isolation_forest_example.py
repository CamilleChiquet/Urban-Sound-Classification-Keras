import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(behaviour='new', max_samples=X.shape[0], contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()


def outlier_prediction(x_train, y_train):
    # Use built-in isolation forest or use predicted vs. actual
    # Compute squared residuals of every point
    # Make a threshold criteria for inclusion

    # The prediction returns 1 if sample point is inlier. If outlier prediction returns -1
    clf_all_features = IsolationForest(max_samples=100)
    clf_all_features.fit(x_train)

    # Predict if a particular sample is an outlier using all features for higher dimensional data set.
    y_pred_train = clf_all_features.predict(x_train)

    # Exclude suggested outlier samples for improvement of prediction power/score
    outlier_map_out_train = np.array(map(lambda x: x == 1, y_pred_train))
    x_train_modified = x_train[outlier_map_out_train, ]
    y_train_modified = y_train[outlier_map_out_train, ]

    return x_train_modified, y_train_modified
