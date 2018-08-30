import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBRegressor, XGBClassifier


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        
        logger.info('\nTimings: %r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed           


class OneHotEncoding(BaseEstimator, TransformerMixin):
    """Takes in dataframe and give one hot encoding for categorical features """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def transform(self, df, y=None):
        """transform a categorical feature into one-hot-encoding"""
        return pd.get_dummies(df, columns=self.column_names)

    def fit(self, df, y=None):
        """Pass"""
        return self


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop the columns in a dataframe """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def transform(self, df, y=None):
        """drop the columns present in self.columns"""
        return df.drop(self.column_names, axis=1)

    def fit(self, df, y=None):
        """Pass"""
        return self


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a columns as feture """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def transform(self, df, y=None):
        """Return the columns"""
        return df.loc[:, self.column_names]

    def fit(self, df, y=None):
        """Pass"""
        return self


class SexBinarizer(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a column as feature """

    def __init__(self, column_names=[]):
        pass

    def transform(self, df, y=None):
        """female maps to 0 and male maps to 1"""
        df.loc[:, "Sex"] = df.loc[:, "Sex"].map({"male": 0, "female": 1})
        return df

    def fit(self, df, y=None):
        """pass"""
        return self


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a columns as feture """

    def __init__(self, column_names=[]):
        self.column_names = column_names
        self.min_max_scalar = MinMaxScaler()

    def transform(self, df, y=None):
        """Min Max Scalar"""
        df.loc[:, self.column_names] = self.min_max_scalar.transform(df[self.column_names].as_matrix())
        return df

    def fit(self, df, y=None):
        """FItting Min Max Scalar"""
        self.min_max_scalar.fit(df[self.column_names].as_matrix())
        return self


class FillNa2(BaseEstimator, TransformerMixin):
    """Takes in dataframe, fill NaN values in a given columns """

    def __init__(self, method="mean"):
        self.method = method
        self.X = pd.DataFrame()

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        if isinstance(df, np.ndarray):
            self.X = pd.DataFrame(df)
        if self.method == "zeros":
            self.X.fillna(0)
        elif self.method == "mean":
            self.X.fillna(self.X.mean(), inplace=True)
        else:
            raise ValueError("Method should be 'mean' or 'zeros'")
        return self.X

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class FillNa(BaseEstimator, TransformerMixin):
    """Takes in dataframe, fill NaN values in a given columns """

    def __init__(self, method="mean"):
        self.method = method

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        if self.method == "zeros":
            df.fillna(0)
        elif self.method == "mean":
            df.fillna(df.mean(), inplace=True)
        else:
            raise ValueError("Method should be 'mean' or 'zeros'")
        return df

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    


class AddTwoCategoricalVariables(BaseEstimator, TransformerMixin):
    def __init__(self, column_1, column_2):
        self.column_1 = column_1
        self.column_2 = column_2
    
    def transform(self, df):
        df[self.column_1 + "_" + self.column_2] = (df[self.column_1].astype(float) + 
                                                (len(df[self.column_1].unique()) * 
                                                (df[self.column_2].astype(float))))
        return df
    
    def fit(self, df, y=None):
        return self


class Numerical2Categorical(BaseEstimator, TransformerMixin):
    def __init__(self, column, ranges, labels):
        self.column = column
        self.ranges = ranges
        self.labels = labels
        
    def transform(self, df):
        df.loc[:, self.column + "_" + "cat"] = (pd
                                                .cut(df.loc[:, self.column], 
                                                     self.ranges, labels=self.labels))
        return df
    
    def fit(self, df, y=None):
        return self