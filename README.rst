KS Metric
=========

Kolmogorov-Smirnov metric (ks metric) is derived from K-S test. K-S test
measures the distance between two plotted cumulative distribution
functions (CDF). To use it as a metric for classification machine
learning problem we see the distance of plotted CDF of target and
non-target. The model that produces the greatest amount of separability
between target and non-target distribution would be considered the
better model.

Installation
------------

The package requires: ``pandas`` and ``numpy``.

To install the package, execute:

.. code:: shell

   $ python setup.py install

or

.. code:: shell

   pip install ks_metric

Usage
-----

To get the KS score :

.. code:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

   from ks_metric import ks_score

   data = load_breast_cancer()
   X, y = data['data'], data['target']
   X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

   clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
   ks_score(y_train, clf.predict_proba(X_train)[:,1])

KS table :

.. code:: python

   from ks_metric import ks_table

   ks_table(y_train, clf.predict_proba(X_train)[:,1])

KS scorer (for hyperparameter search) :

.. code:: python

   from sklearn.model_selection import GridSearchCV
   from ks_metric import ks_scorer

   clf = GridSearchCV(estimator=LogisticRegression(), param_grid={'C':[0.01,0.1,1]}, scoring=ks_scorer)

see the example notebook for detailed usage.
