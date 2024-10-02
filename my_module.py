from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
class SparseToDenseTransformer():
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray()
class Processor_s():
    def Transformer(x,y):
        y=float(y)
        if x=='snapdragon':
            if (y) < 10:
                return round(round(y)*115+(y-round(y))*500)
            elif y>4000:
                return(y/9)
            else :
                return y
        else:
            return y
class LogStandardScaler(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, y):
        y_log = np.log1p(y)
        self.scaler.fit(y_log)
        return self

    def transform(self, y):
        y_log = np.log1p(y)
        return self.scaler.transform(y_log)

    def inverse_transform(self, y):
        y_scaled = self.scaler.inverse_transform(y)
        return np.expm1(y_scaled)
class confidence_():
    def __init__(self):
        self.model=pickle.load((open('smart_phone_price_with_PLSR.pkl','rb')))
        self.pipe=pickle.load(open('pipe.pkl','rb'))
        self.pls_x=self.model.named_steps['reg'].regressor_.x_scores_
        self.x_weights_=self.model.named_steps['reg'].regressor_.x_weights_
    def interval(self,x):
        Y=self.model.predict(x)
        XTX_inv=np.linalg.inv(self.pls_x.T.dot(self.pls_x))
        X_conponet=self.pipe.transform(x).dot(self.x_weights_).reshape(-1,1)
        lower=Y-1.6*np.sqrt(((X_conponet.T.dot(XTX_inv)).dot(X_conponet)+1)*0.0229)
        upper=Y+1.6*np.sqrt(((X_conponet.T.dot(XTX_inv)).dot(X_conponet)+1)*0.0229)
        return (lower[0,0],upper[0,0])