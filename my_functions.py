import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        # X.drop(["VentasMismoDíaU1","VentasMismoDíaU2","VentasMismoDíaU3","VentasMismoDíaU4",
        # "VentasMismoDíaU5","VentasMismoDíaU6","VentasMismoDíaU7","VentasMismoDíaU8",
        # "VentasMismoDíaU9","VentasMismoDíaU10","VentasU1","VentasU2","VentasU3","VentasU4","VentasU5"
        # ,"VentasU6","VentasU7","VentasU8","VentasU9","VentasU10","Date","DiaMes","Objetivo","Tmin","Tmax","EsDiaLaboral"],axis=1, inplace = True)
        X.drop(["Date","DiaMes","Objetivo","Tmin","Tmax","EsDiaLaboral"],axis=1, inplace = True)
        return X
class ConvertNegativeToCero(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        for col in X.columns:
            X.loc[X[col] < 0] = 0
        return X
class ConvertNegativeToString(BaseEstimator, TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        columnas = self.columns
        for col in columnas:
            X[col] = X[col].apply(str)
            X[col] = X[col].replace('-1','unk')
        return X
    
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_encoder = OneHotEncoder()):
        self.ohe = one_hot_encoder
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        columnas = X.columns
        for col in columnas:
            X[col] = X[col].apply(str)
            X[col] = X[col].replace('-1','unk')        
        oencoder = self.ohe
        X = oencoder.transform(X)
        return X
def get_categories(X, attribs = []):
    result = []
    return result

        