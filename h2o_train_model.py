
import h2o
h2o.init(min_mem_size=8)
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.model.metrics_base import H2OAutoEncoderModelMetrics
from h2o.model.metrics_base import MetricsBase
from h2o.model.metrics_base import H2OBinomialModelMetrics   
from h2o.model.metrics_base import H2OClusteringModelMetrics
from h2o.model.metrics_base import H2ODimReductionModelMetrics 
from h2o.model.metrics_base import H2OMultinomialModelMetrics
import pandas as pd
import h2o
from h2o.job import H2OJob
from h2o.frame import H2OFrame
from h2o.exceptions import H2OValueError
from h2o.estimators.estimator_base import H2OEstimator
from h2o.two_dim_table import H2OTwoDimTable
from h2o.display import H2ODisplay
from h2o.grid.metrics import *  # NOQA
from h2o.utils.backward_compatibility import backwards_compatible
from h2o.utils.shared_utils import deprecated, quoted
from h2o.utils.compatibility import *  # NOQA
from h2o.utils.typechecks import assert_is_type, is_type



df = h2o.import_file('var_class_Imp_only.csv')
#df= h2o.H2OFrame(result1)
#var = pd.read_csv('abc.csv')
df[df==0]=None

split= df.split_frame(ratios=[0.7], seed=-1)

df_train = split[0]
df_valid = split[1]

y= 'response'
x= df.col_names
x.remove(y)

variables= list(range(1,df_train.shape[1]))

param = {
      "sample_rate" : 0.5
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 1
    , "seed": -1
    , "score_tree_interval": 100
    ,"stopping_metric": "AUTO"
    ,"stopping_tolerance":0.001
    ,"max_bins":256
    ,"min_sum_hessian_in_leaf":100.00
    , "distribution": "Multinomial"
    
}


hyper_parameters = {'ntrees':[800], 'max_depth':[12], 'learn_rate':[0.03]}
grid_search = H2OGridSearch(model= H2OXGBoostEstimator(**param), hyper_params=hyper_parameters)
grid_search.train(x=list(variables), y=0, training_frame=df_train, validation_frame= df_valid,**param)
grid_search.get_grid(sort_by='r2', decreasing= True)
model= grid_search[0]
print(grid_search.auc)

h2o.save_model(model=model, path="", force=True)



model_path = h2o.save_model(model=model, path="", force=True)




# =============================================================================
# saved_model = h2o.load_model("/home/sahan/Downloads/RandomForest/Grid_XGBoost_py_15_sid_ab71_model_python_1546583108429_5_model_0")
# 
# df_test= h2o.import_file('input.csv')
# preds= saved_model.predict(df_test)
# dff = preds.as_data_frame();dff
# d= dff.to_json()
# print(d)
# 
# =============================================================================
