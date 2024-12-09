import pandas as pd
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor

class FeatSelector():
    def __init__(self, method, topk):
        self.method = method
        self.topk = topk
        
    def topk_feats(self, input_vect, output):
    
        if self.method == "RF-R":
            selector = RandomForestRegressor()
        elif self.method == "RF-C":
            selector = RandomForestClassifier()
        elif self.method == "ET-R":
            selector = ExtraTreesRegressor()
        elif self.method == "XGB-R":
            selector = xgb.XGBRegressor()
        else:
            raise ValueError("Method {self.method}, is not included.")

        selector.fit(input_vect, output)

        importances = pd.DataFrame({'feature': input_vect.columns, 'importance': selector.feature_importances_})
        importances = importances.sort_values('importance', ascending=False)

        return list(importances["feature"][:self.topk].values)
    
        
