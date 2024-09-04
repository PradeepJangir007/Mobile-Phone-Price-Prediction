class SparseToDenseTransformer():
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray()