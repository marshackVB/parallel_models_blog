# Databricks notebook source
# MAGIC %md #Training machine learning models in parallel using PandasUDFs on Databricks  
# MAGIC 
# MAGIC Data Scientists often need to fit models to different groups of data. A data scientist in real estate industry may find it more effective to create separate models per geographic area due to regional difference that impact model performance.  
# MAGIC 
# MAGIC PandasUDFs on Databricks provide a mechanism for fitting machine leaning models on different groups of data in parallel. Models can be tuned using Hyperopt, an optimization framework built into the Machine Learning Runtime. Groups of fitted models can be saved to an MLflow Tracking Server instance and promoted to the Model Registry for inference.

# COMMAND ----------

from collections import OrderedDict
import datetime
from pickle import dump
from typing import List, Callable
import csv

from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.datasets import make_classification

from pyspark import TaskContext
from pyspark.sql.functions import col
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType, MapType

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
from hyperopt.early_stop import no_progress_loss

import mlflow
from mlflow.tracking import MlflowClient

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md ## Environment setup
# MAGIC We want to fit each model in parrallel using separate Spark tasks. When working with smaller groups of data, Adaptive Query Execution (AQE) can combine these smaller model fitting tasks into a single, larger task where models are fit sequentially. Since we want to avoid this behavior in this example, we will disable Adaptive Query Execution. Generally, AQE should be left enabled.

# COMMAND ----------

spark.conf.set('spark.sql.adaptive.enabled', 'false')

# COMMAND ----------

# MAGIC %md Also, since we are using Python libraries that can benefit from multiple cores, we can instruct Spark to provide more than one CPU core per tasks by setting **spark.task.cpus** in the Advanced options of the Clusters UI. In the Spark config section under the Spark tab, we set **spark.task.cpus 8**. In our example, we will fit 10 models in parrallel, so we need 80 cores in total to fit all models at the same time. We have also chosen compute optimized instances due to the computational intensity of our UDFs.

# COMMAND ----------

# MAGIC %md ## Generate sample data
# MAGIC We can use a PandasUDF to create synthetic, binary classification training datasets for each group. We'll create a Spark DataFrame containing 10 groups, then use a PandasUDF to generate data for each group.

# COMMAND ----------

groups = [[f'group_{str(n+1).zfill(2)}'] for n in range(10)]

schema = StructType()
schema.add('group_name', StringType())

df = spark.createDataFrame(groups, schema=schema)
display(df)

# COMMAND ----------

def create_group_data(group_data: pd.DataFrame) -> pd.DataFrame:
  """
  Generate a synthetic classification dataset
  """
  
  n_samples = 10000
  n_features = 20

  X, y = make_classification(n_samples=      n_samples, 
                             n_features=     n_features, 
                             n_informative=  10, 
                             n_redundant=    0, 
                             n_classes=      2, 
                             flip_y=         0.4,
                             random_state=   np.random.randint(1,999))
  
  numeric_feature_names = [f'numeric_feature_{str(n+1).zfill(2)}' for n in range(n_features)]
  categorical_feature_names = []

  df = pd.DataFrame(X, columns=numeric_feature_names)
  
  num_categorical_features = 1
  
  # Convert numeric column to categorical based on quartiles
  for numeric_feature_name in numeric_feature_names[:num_categorical_features]:
  
    numeric_feature_names.remove(numeric_feature_name)

    categorical_name = numeric_feature_name.replace("numeric", "categorical")

    categorical_feature_names.append(categorical_name)

    df[categorical_name] = pd.qcut(df['numeric_feature_01'],
                                        q = [0, .25, .5, .75, 1],
                                       labels=False,
                                       precision = 0)
    
  df = df[categorical_feature_names + numeric_feature_names]

  # Convert a proportion of values to missing
  percent_missing_values = 0.05
  mask = np.random.choice([True, False], size=df.shape, p=[percent_missing_values, 1 - percent_missing_values])
  df = df.mask(mask, other=np.nan)

  df['label'] = y
  df['group_name'] = group_data["group_name"].loc[0]

  col_ordering = ['group_name', 'label'] + categorical_feature_names + numeric_feature_names

  return df[col_ordering]

# COMMAND ----------

# MAGIC %md Our synthetic dataset contains the group name column, label, distinct id of each row, a single categorical feature, and 20 numerical features.

# COMMAND ----------

schema = StructType()
schema.add('group_name', StringType())
schema.add('label', IntegerType())
schema.add('categorical_feature_01', FloatType())

num_categorical_features = 1
total_features = 20

for column_name in [f'numeric_feature_{str(n+1).zfill(2)}' for n in range(num_categorical_features, total_features)]:
  schema.add(column_name, FloatType())
  
features = (df.groupby('group_name').applyInPandas(create_group_data, schema=schema)
              .withColumn('id', func.monotonically_increasing_id()))

features.write.mode('overwrite').format('delta').partitionBy('group_name').saveAsTable('default.synthetic_group_features')

features = spark.table('default.synthetic_group_features')
display(features)

# COMMAND ----------

# MAGIC %md ## Feature encoding
# MAGIC We will fit [XGBoost classification models](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier), which requires that our features are encoded. We will use scikit-learn's Pipeline and ColumnTransformer to apply different transformations based on column name.

# COMMAND ----------

def create_preprocessing_transform(categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
  
  categorical_pipe = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

  numerical_pipe_quantile = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

  preprocessing = ColumnTransformer(
        [
            ("categorical", categorical_pipe,        categorical_features),
            ("numeric",     numerical_pipe_quantile, numerical_features)
        ],
        remainder='drop'
    )

  return preprocessing

# COMMAND ----------

# MAGIC %md ## Fitting mutltiple XGBoost models using a PandasUDF
# MAGIC For a first example, we will create a PandasUDF that will fit XGBoost models separately on the each groups's features. Then, we will incorporate more functionality related to hyper-parameter tuning and MLflow.

# COMMAND ----------

# MAGIC %md ### Define the PandasUDF  
# MAGIC 
# MAGIC  - The **configure_model** wrapper function allows us to make several parameters available to our PandasUDF, **train_model**, which receives a Pandas DataFrame and returns a Pandas DataFrame.  
# MAGIC  
# MAGIC  - The data for each ship will be passed into a separate instance of our PandasUDF and each instance will be executed in parallel in different tasks on our cluster. We will capture information about where each model is trained to confirm this behavior.
# MAGIC  
# MAGIC  - Since XGBoost fits tree-based models sequentially, we will leverage the built-in early stopping functionality and continue to fit additional trees until the model's predictive capability stops improving. In our example, if a model's performance does not improve after building 25 consecutive trees, we will stop training.

# COMMAND ----------

def configure_model_udf(label_col: str, grouping_col:str, pipeline:ColumnTransformer, test_size:float=0.33, 
                        xgb_early_stopping_rounds:int=25, eval_metric:str="auc", random_state:int=123) -> Callable[[pd.DataFrame], pd.DataFrame]:
  
  """
  Configure a PandasUDF function and that trains and XGBoost model on a group of data. The UDF is applied
  using the groupBy.applyInPandas method.
  """
    
  def train_model_udf(group_training_data):
    
    # Measure the training time of each model
    start = datetime.datetime.now()
    
    # Capture the name of the group to be modeled
    group_name = group_training_data[grouping_col].loc[0]
    
    x_train, x_test, y_train, y_test = train_test_split(group_training_data, 
                                                        group_training_data[label_col], 
                                                        test_size=test_size, 
                                                        random_state=random_state)
     
    # We must pass the testing dataset to the model to leverage early stopping,
    # and the training dataset must be transformed.
    x_train_transformed = pipeline.fit_transform(x_train)
    x_test_transformed = pipeline.transform(x_test)
    
    # Create a scikit-learning pipeline that transforms the features and applies the
    # model
    model = XGBClassifier(n_estimators=1000)
    
    # Fit the model with early stopping
    # Note: Early stopping returns the model from the last iteration (not the best one). If there’s more 
    # than one item in eval_set, the last entry will be used for early stopping.
    model.fit(x_train_transformed, y_train.values.ravel(),
              eval_set = [(x_test_transformed, y_test.values.ravel())],
              eval_metric=eval_metric,
              early_stopping_rounds=xgb_early_stopping_rounds,
              verbose=True)
    
    # Capture statistics on the best model run
    best_score = model.best_score
    
    # The best performing number of XGBoost trees
    best_iteration = model.best_iteration
    best_xgboost_rounds = (0, best_iteration + 1)
    
    # Predict using only the boosters leading up to and including the best boosting 
    # round. This accounts for the fact that the model retained by xgboost is the last
    # model fit before early stopping rounds were triggered
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, 
                                                                                 model.predict(x_train_transformed, 
                                                                                               iteration_range=best_xgboost_rounds), 
                                                                                 average='weighted')
    
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, 
                                                                              model.predict(x_test_transformed, 
                                                                                            iteration_range=best_xgboost_rounds), 
                                                                              average='weighted')
    
    train_auc = roc_auc_score(y_train, 
                              model.predict_proba(x_train_transformed, 
                                                  iteration_range=best_xgboost_rounds)[:,1],
                              average="weighted")
    
    end = datetime.datetime.now()
    elapsed = end-start
    seconds = round(elapsed.total_seconds(), 1)
    
    # Capture data about our the model
    digits = 3
    metrics = OrderedDict()
    metrics["train_precision"]= round(precision_train, digits)
    metrics["train_recall"] =   round(recall_train, digits)
    metrics["train_f1"] =       round(f1_train, digits)
    metrics["train_auc"] =      round(train_auc, digits)
    metrics["test_precision"] = round(precision_test, digits)
    metrics["test_recall"] =    round(recall_test, digits)
    metrics["test_f1"] =        round(f1_test, digits)
    metrics["test_auc"] =       round(best_score, digits)
    metrics["best_iteration"] = round(best_iteration, digits)
    
    other_meta = OrderedDict()
    other_meta['group'] =           group_name
    other_meta['stage_id'] =        TaskContext().stageId()
    other_meta['task_attempt_id'] = task_attempt_id = TaskContext().taskAttemptId()
    other_meta['start_time'] =      start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    other_meta['end_time'] =        end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    other_meta['elapsed_seconds'] = seconds
    
    other_meta.update(metrics)
    
    return pd.DataFrame(other_meta, index=[0])

  return train_model_udf

# COMMAND ----------

# MAGIC %md ### Apply the PandasUDF  
# MAGIC The PandasUDF returns a Pandas DataFrame; we must specify a Spark DataFrame schema that maps to the column names and Python data types returned by the UDF. 

# COMMAND ----------

# Specify Spark DataFrame Schema
spark_types = [('group',             StringType()),
                ('stage_id',         IntegerType()),
                ('task_attempt_id',  IntegerType()),
                ('start_time',       StringType()),
                ('end_time',         StringType()),
                ('elapsed_seconds',  FloatType()),
                ('train_precision',  FloatType()),
                ('train_recall',     FloatType()),
                ('train_f1',         FloatType()),
                ('train_auc',        FloatType()),
                ('test_precision',   FloatType()),
                ('test_recall',      FloatType()),
                ('test_f1',          FloatType()),
                ('test_auc',         FloatType()),
                ('best_iteration',   IntegerType())]

spark_schema = StructType()
for col_name, spark_type in spark_types:
  spark_schema.add(col_name, spark_type)
  
  
categorical_features = [col for col in features.columns if 'categorical' in col]
numerical_features =   [col for col in features.columns if 'numeric' in col]
label_col =            ['label']
grouping_col =         'group_name'

# COMMAND ----------

# MAGIC %md ### View models' output
# MAGIC We will now apply our PandasUDF to a Spark Dataframe, returning a new Spark Dataframe. Our returned DataFrame contains one row per group in our data. We can see that models were fit seprately for each group on different, independent Spark tasks and these tasks were all executed at the same time, and therefore, in the same stage. We also retrieved several performance statistics for each model.

# COMMAND ----------

# Create a pre-processing pipeline instance
pipeline = create_preprocessing_transform(categorical_features, numerical_features)

# Configure the PandasUDF
train_model_udf = configure_model_udf(label_col, 
                                      grouping_col, 
                                      pipeline)

best_model_stats = features.groupBy('group_name').applyInPandas(train_model_udf, schema=spark_schema)

best_model_stats.write.mode('overwrite').format('delta').saveAsTable('default.best_model_stats')
display(spark.table('default.best_model_stats'))

# COMMAND ----------

# MAGIC %md ## Tuning our models using Hyperopt
# MAGIC Now that we can fit models in parallel on different groups of data, we shift toward model tuning. Compared to arbitrarily choosing different model parameters to test, the Hyperopt optimization library, which is built into the Databricks Machine Learning runtime,  provides a more robust mechanism for intelligently searching a broader hyperparameter space, potentially leading to better models.
# MAGIC 
# MAGIC We can incorporate a hyper-parameter search using Hyperopt into our Pandas UDF. Let's first fit a simple example on a single group for illustration purposes.

# COMMAND ----------

# MAGIC %md ### Specify a search space  
# MAGIC We focus on four parameters to tune using Hyperopt that can help reduce overfitting by [adjusting different behaviors of our XGBoost models](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster) during training.
# MAGIC  - **max_depth**: the maximum depth of each XGBoost tree.
# MAGIC  - **lambda**: a regularization parameter that reduces the model's sensitivity to the training data.
# MAGIC  - **subsample**: the percent of training data rows that will be sampled before fitting a single tree.
# MAGIC  - **colsample_bytree**: the percent of columns that will be sampled before fitting a single tree.

# COMMAND ----------

parameter_search_space = {'n_estimators':      1000,
                          'max_depth':         hp.quniform('max_depth', 3, 18, 1),
                          'lambda':            hp.uniform('lambda', 1, 15),
                          'subsample':         hp.uniform('subsample', 0.5, 1.0),
                          'colsample_bytree':  hp.uniform('colsample_bytree', 0.5, 1.0),
                          'eval_metric':       'auc',
                          'use_label_encoder': False,
                          'random_state':      1}

# COMMAND ----------

# MAGIC %md ### Specify a Hyperopt Objective Function  
# MAGIC We create a function that can receive combinations of hyper-parameter values from Hyperopt, fit an XGBoost model using those parameters, and return information to Hyperopt. This information, specifically the models 'loss', will be used by Hyperot to influence which hyper-parameter combinations should be tested next. Our loss will be calculated as 1 - the area under the curve (auc). Thus, we will find the model with the highest auc value, where 1 is the highest possible value.
# MAGIC 
# MAGIC  Will will again add a parent function, **configure_object_fn**, to pass additional information to our Hyperopt objective function.

# COMMAND ----------

def configure_object_fn(x_train_transformed, y_train, x_test_transformed, y_test, xgb_early_stopping_rounds=25, 
                        eval_metric="auc") -> Callable[[dict], dict]:
  """
  Configure a Hyperopt objective function
  """

  def hyperopt_objective_fn(params):

    # Some model parameters require integeger values; change the type in these cases
    params['max_depth'] =        int(params['max_depth'])

    model = XGBClassifier(**params)

    model.fit(x_train_transformed, y_train.values.ravel(),
              eval_set = [(x_test_transformed, y_test.values.ravel())],
              # See options here: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
              eval_metric=eval_metric,
              early_stopping_rounds=xgb_early_stopping_rounds,
              verbose=True)

    best_score = model.best_score
    best_iteration = model.best_iteration
    best_xgboost_rounds = (0, best_iteration + 1)
    
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, 
                                                                                 model.predict(x_train_transformed, 
                                                                                               iteration_range=best_xgboost_rounds), 
                                                                                 average='weighted')
    
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, 
                                                                              model.predict(x_test_transformed, 
                                                                                            iteration_range=best_xgboost_rounds), 
                                                                              average='weighted')
    
    train_auc = roc_auc_score(y_train, 
                              model.predict_proba(x_train_transformed, 
                                                  iteration_range=best_xgboost_rounds)[:,1],
                              average="weighted")

    digits = 3
    metrics = OrderedDict()
    metrics["train_precision"]= round(precision_train, digits)
    metrics["train_recall"] =   round(recall_train, digits)
    metrics["train_f1"] =       round(f1_train, digits)
    metrics["train_auc"] =      round(train_auc, digits)
    metrics["test_precision"] = round(precision_test, digits)
    metrics["test_recall"] =    round(recall_test, digits)
    metrics["test_f1"] =        round(f1_test, digits)
    metrics["test_auc"] =       round(best_score, digits)
    metrics["best_iteration"] = round(best_iteration, digits)

    return {'status': STATUS_OK, 'loss': 1- best_score, 'metrics': metrics}
      
  return hyperopt_objective_fn

# COMMAND ----------

# MAGIC %md ### Encode input datasets and configure Hyperopt Objective Function  

# COMMAND ----------

features_df = features.filter(col('group_name') == 'group_01').toPandas()

x_train, x_test, y_train, y_test = train_test_split(features_df, 
                                                    features_df['label'], 
                                                    test_size=0.33, 
                                                    random_state=123)

pipeline = create_preprocessing_transform(categorical_features, numerical_features)

x_train_transformed = pipeline.fit_transform(x_train)
x_test_transformed = pipeline.transform(x_test)


objective_fn = configure_object_fn(x_train_transformed, y_train, x_test_transformed, y_test)

# COMMAND ----------

# MAGIC %md ### Launch the Hyperopt tuning workflow

# COMMAND ----------

trials = Trials()

best_params = fmin(fn=objective_fn, 
                   space=parameter_search_space, 
                   algo=tpe.suggest,
                   max_evals=25,
                   trials=trials, 
                   rstate=np.random.default_rng(50))

# COMMAND ----------

# MAGIC %md ### View Hyperopt findings  
# MAGIC Hyperopt provides the hyper-parameter combination of the best model as well as validation statistics generated using this model. We can use this information to fit a final model on our full datasets.

# COMMAND ----------

print("Best model parameters \n")
for param, value in best_params.items():
  print(param, value)

# COMMAND ----------

print("Best model statistics \n")
for metric, value in trials.best_trial['result']['metrics'].items():
  print(metric, value)

# COMMAND ----------

print(f"Best Hyperopt trial: {trials.best_trial['tid']}")

# COMMAND ----------

# MAGIC %md ### Fitting a final model  
# MAGIC We will construct a final model using the hyperprameter values returned by Hyperopt. There can be differences in the data types returned by Hyperopt and those required by libraries such as XGBoost. We will do some type conversion to account for this.  
# MAGIC   
# MAGIC The **n_estimators** parameter was found using XGBoosts early stopping functionality. Trees built beyond this number did not improve the model's predictive performance on the test dataset.  
# MAGIC   
# MAGIC In this example, we will combine the preprocessing pipeline and XGBoost model into a single Pipeline object so that feature transformation and model fitting/inference can occure via a single function call.

# COMMAND ----------

# Collect the best model parameters
final_model_parameters = {}
final_model_parameters['n_estimators'] = trials.best_trial['result']['metrics']['best_iteration']

for parameter, value in best_params.items():
      if parameter in ['max_depth']:
        final_model_parameters[parameter] = int(value)
      else:
        final_model_parameters[parameter] = value
        
# Specify model        
model = XGBClassifier(**final_model_parameters)

# Define pre-processing pipeline
pipeline = create_preprocessing_transform(categorical_features, numerical_features)

# Combine pre-processing pipeline and model
model_pipeline = Pipeline([("preprocess", pipeline), ("classifier", model)])

# fit pre-processor and model
model_pipeline.fit(features_df, features_df['label'])

# COMMAND ----------

# MAGIC %md ### Performing inference  
# MAGIC Notice we are able to call our fitted model Pipeline on raw input data. Our Pipeline handles both feature transformations/encoding and prediction.

# COMMAND ----------

# Probability that passenger survived
predictions = pd.DataFrame(model_pipeline.predict_proba(x_test)[:, 1] , columns=['label_probability'])
predictions = pd.concat([predictions, x_test.reset_index(drop=True)], axis=1)
predictions.head()

# COMMAND ----------

# MAGIC %md ### Incorporating Hyperopt into our PandasUDF
# MAGIC With minor alterations, we can include Hyperopt into our UDF.

# COMMAND ----------

def configure_model_hyperopt_udf(label_col:str, grouping_col:str, pipeline:ColumnTransformer, parameter_search_space, 
                                 xgb_early_stopping_rounds:str=25, max_hyperopt_evals:int=25, eval_metric:str="auc", 
                                 test_size:float=0.33, random_state:int=123) -> Callable[[pd.DataFrame], pd.DataFrame]:
  
  """
  Configure a PandasUDF that train models using Hyperopt for hyperparameter tuning
  """
    
    
  def train_model_hyperopt_udf(group_training_data):
    
    start = datetime.datetime.now()
    
    group_name = group_training_data[grouping_col].loc[0]
    
    x_train, x_test, y_train, y_test = train_test_split(group_training_data, 
                                                        group_training_data[label_col], 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    # Transforming features outside of the iterative Hyperopt workflow
    x_train_transformed = pipeline.fit_transform(x_train)
    x_test_transformed = pipeline.transform(x_test)
    
    
    def hyperopt_objective_fn(params):
      
      # Some model parameters require integeger values; change the time in these cases
      params['max_depth'] =        int(params['max_depth'])
      
      model = XGBClassifier(**params)
        
      model.fit(x_train_transformed, y_train.values.ravel(),
                eval_set = [(x_test_transformed, y_test.values.ravel())],
                # See options here: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
                eval_metric=eval_metric,
                early_stopping_rounds=xgb_early_stopping_rounds,
                verbose=True)
      
      best_score = model.best_score
      best_iteration = model.best_iteration
      best_xgboost_rounds = (0, best_iteration + 1)
      
      precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, 
                                                                                   model.predict(x_train_transformed, 
                                                                                                 iteration_range=best_xgboost_rounds), 
                                                                                   average='weighted')
    
      precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, 
                                                                                model.predict(x_test_transformed, 
                                                                                              iteration_range=best_xgboost_rounds), 
                                                                                average='weighted')
      train_auc = roc_auc_score(y_train, 
                                model.predict_proba(x_train_transformed, 
                                                    iteration_range=best_xgboost_rounds)[:,1],
                                average="weighted")
      
      digits = 3
      metrics = OrderedDict()
      metrics["train_precision"]= round(precision_train, digits)
      metrics["train_recall"] =   round(recall_train, digits)
      metrics["train_f1"] =       round(f1_train, digits)
      metrics["train_auc"] =      round(train_auc, digits)
      metrics["test_precision"] = round(precision_test, digits)
      metrics["test_recall"] =    round(recall_test, digits)
      metrics["test_f1"] =        round(f1_test, digits)
      metrics["test_auc"] =     round(best_score, digits)
      metrics["best_iteration"] = round(best_iteration, digits)
    
      return {'status': STATUS_OK, 'loss': 1- best_score, 'metrics': metrics}
    
  
    trials = Trials()

    best_params = fmin(fn=hyperopt_objective_fn, 
                       space=parameter_search_space, 
                       algo=tpe.suggest,
                       max_evals=max_hyperopt_evals, 
                       trials=trials, 
                       rstate=np.random.default_rng(50))
        

    # Fit final model with best parameters on full dataset
    final_model_parameters = {}
    final_model_parameters['n_estimators'] = int(trials.best_trial['result']['metrics']['best_iteration'])

    # Adjust parameter data types to meet xgboost requirements
    for parameter, value in best_params.items():
          if parameter in ['max_depth']:
            final_model_parameters[parameter] = int(value)
          else:
            final_model_parameters[parameter] = value

    # Fit the pipeline and model on full dataset
    final_model = XGBClassifier(**final_model_parameters)
    final_pipeline = Pipeline([("preprocess", pipeline), ("classifier", final_model)])

    final_pipeline.fit(group_training_data, group_training_data[label_col])
    
    end = datetime.datetime.now()
    elapsed = end-start
    seconds = round(elapsed.total_seconds(), 1)
    
    # Construct final output
    output =                    OrderedDict()
    output['group'] =           group_name
    output['stage_id'] =        TaskContext().stageId()
    output['task_attempt_id'] = task_attempt_id = TaskContext().taskAttemptId()
    output['start_time'] =      start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    output['end_time'] =        end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    output['elapsed_seconds'] = seconds
  
    output.update(trials.best_trial['result']['metrics'])
    
    return pd.DataFrame(output, index=[0])
  
  return train_model_hyperopt_udf

# COMMAND ----------

# Create a pre-processing pipeline instance
pipeline = create_preprocessing_transform(categorical_features, numerical_features)

# Configure the PandasUDF
train_model_hyperopt_udf = configure_model_hyperopt_udf(label_col, 
                                                        grouping_col, 
                                                        pipeline,
                                                        parameter_search_space)
  
best_model_stats = features.groupBy('group_name').applyInPandas(train_model_hyperopt_udf, schema=spark_schema)

best_model_stats.write.mode('overwrite').format('delta').saveAsTable('default.best_model_stats')

display(spark.table('default.best_model_stats'))

# COMMAND ----------

# MAGIC %md ## Tracking model runs and artifacts with MLflow
# MAGIC We can leverage an MLflow Tracking Server to record information about our model runs, as well as artifacts, like fitted models. By leveraging more advanced MLflow capabilities we can also create a PandasUDF for model inference, which will allow us to score different groups of data with the model trained on each group.

# COMMAND ----------

# MAGIC %md ### Create an MLflow Tracking Server instance  
# MAGIC You will see an entry in the MLflow Experiments UI for "pandas_udf_models". Our models will be logged to that location.

# COMMAND ----------

def get_or_create_experiment(experiment_location: str) -> None:
 
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)


experiment_location = '/Shared/pandas_udf_models'
get_or_create_experiment(experiment_location)

# COMMAND ----------

def get_new_run(experiment_location: str, run_name: str) -> str:
  """
  Given an MLflow experiment location and a run name, create an 
  MLflow experiment run to which artifacts can be logged.
  """
  
  mlflow.set_experiment(experiment_location)
  run = mlflow.start_run(run_name=run_name)
  run_id = run.to_dictionary()['info']['run_id']
  mlflow.end_run()
  
  return run_id

# COMMAND ----------

# MAGIC %md ### Working with custom MLflow models  
# MAGIC Custom MLflow models provide a way to store special transformations as an MLflow model flavor that can be easily managed. You may want to alter the behavior of a modeling framework's built-in predict method or store a transformation that is not part of a supported ml framework. Both of these use cases are possible with Custom MLflow models.  
# MAGIC 
# MAGIC See a simple example below that receive and input and then multiplies that input by a number. We can store this custom model in MLflow, load it, and apply it to a Spark DataFrame.

# COMMAND ----------

class CustomPythonModel(mlflow.pyfunc.PythonModel):
  def __init__(self, multiply_by):
    super().__init__()
    self.multiply_by = multiply_by
    
  def predict(self, context, model_input):
    prediction = model_input * self.multiply_by
    return prediction
  
  
with mlflow.start_run() as run:
  
  run_id = run.info.run_id
  print(f"run_id: {run_id}")
  
  my_custom_model = CustomPythonModel(2)
  mlflow.pyfunc.log_model("my_model", python_model=my_custom_model)

# COMMAND ----------

# MAGIC %md Apply the custom model to the features DataFrame

# COMMAND ----------

# Load the model from MLflow
logged_model = f'runs:/{run_id}/my_model'
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Apply the model to a Spark Dataframe
my_predictions = features.select('numeric_feature_02').withColumn('numeric_feature_02_scaled', loaded_model('numeric_feature_02'))

display(my_predictions)

# COMMAND ----------

# MAGIC %md ### Creating a custom MLflow models to load each group's model
# MAGIC As we fit models on separate groups of data using our PandasUDF, we will log each fitted model to the same run in MLflow as an artifact. We will also create a "meta" model that given a group of data, will load the group's trained model from MLflow.

# COMMAND ----------

class GroupInferenceModel(mlflow.pyfunc.PythonModel):
  """
  A custom MLflow model designed to accept a group of data, 
  import that group's trained model from MLflow, and return 
  predictions for the group.
  
  Attributes:
    run_id: The MLflow run id to which all models will be logged
    group_name_col: The column name containing the grouping variable
    id_cols: The id columns that should be returned along with the 
             predictions
  """

  def __init__(self, run_id:str, group_name_col:str, id_cols:List[str]):
    super().__init__()
    self.run_id = run_id
    self.group_name_col = group_name_col    
    self.id_cols = id_cols
    
  def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
    
    # Determine the group of the data
    group_name = model_input[self.group_name_col].loc[0]
    
    # Load the group's trained model from MLflow
    model_artifact_location = f'runs:/{self.run_id}/models/{group_name}'
    
    model = mlflow.sklearn.load_model(model_artifact_location)
    
    predictions = model.predict_proba(model_input)
    
    output_df = model_input[[self.group_name_col] + self.id_cols]
    output_df['probabilities'] = predictions.tolist()
    
    return output_df

# COMMAND ----------

# MAGIC %md ### Adding the custom MLflow model to our PandasUDF and Logging to MLflow  
# MAGIC 
# MAGIC Model metrics and parameters are stored in csv files within each group's model artifact directory. We also include the best model parameters found by Hyperopt within our output Delta table as a Spark MapType.
# MAGIC 
# MAGIC In addition, we leverage Hyperopt's early stopping functionality. Similar to early stopping for XGBoost, we can end our Hyperopt training runs if performance does not improve. Since we want to find our best models efficiently and are now working with larger datasets, we will instruct Hyperopt to stop testing hyper-parameters if our loss functions does not decrease after 25 trials.
# MAGIC 
# MAGIC We set the **early_stop_fn** to **no_progress_loss**, which specifies the theshold beyond which the loss for a trial must improve. We will specify that the model loss must inprove by one half of a percentage point after 25 trials or else model training will stop. Details of the early stopping function [is available here](https://github.com/hyperopt/hyperopt/blob/master/hyperopt/early_stop.py), which is referenced in the [databricks documentation](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html#fmin).

# COMMAND ----------

def configure_model_hyperopt_mlflow_udf(label_col:str, grouping_col:str, id_cols:List[str], pipeline:ColumnTransformer, parameter_search_space, 
                                        experiment_location:str, run_id:str, xgb_early_stopping_rounds:str=25, max_hyperopt_evals:int=200,
                                        hyperopt_early_stopping_rounds:str=25, hyperopt_early_stopping_threshold:float=0.5, eval_metric:str="auc",
                                        test_size:float=0.33, random_state:int=123) -> Callable[[pd.DataFrame], pd.DataFrame]:
  
  
  # Log the meta model to the parent run
  with mlflow.start_run(run_id = run_id) as run:

      meta_model = GroupInferenceModel(run_id, grouping_col, id_cols)  
      mlflow.pyfunc.log_model("meta_model", python_model=meta_model)
    
    
  def train_model_hyperopt_mlflow_udf(group_training_data: pd.DataFrame) -> pd.DataFrame:
    """
    A PandasUDF that give a group of data will train an XGBoost model. Hyperparameter tuning
    is performed using Hyperopt. The best model parameters found by Hyperopt is used to train
    a final model. The model is logged to MLflow. Model fit statistics and parameters are logged
    as .csv files in the model artifact directory in MLflow
    """
    
    start = datetime.datetime.now()
    
    group_name = group_training_data[grouping_col].loc[0]
    
    x_train, x_test, y_train, y_test = train_test_split(group_training_data, 
                                                        group_training_data[label_col], 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    # Transforming features outside of the iterative Hyperopt workflow
    x_train_transformed = pipeline.fit_transform(x_train)
    x_test_transformed = pipeline.transform(x_test)
    
    
    def hyperopt_objective_fn(params):
      
      # Some model parameters require integeger values; change the type in these cases
      params['max_depth'] =        int(params['max_depth'])
      
      model = XGBClassifier(**params)
        
      model.fit(x_train_transformed, y_train.values.ravel(),
                eval_set = [(x_test_transformed, y_test.values.ravel())],
                # See options here: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
                eval_metric=eval_metric,
                early_stopping_rounds=xgb_early_stopping_rounds,
                verbose=True)
      
      best_score = model.best_score
      best_iteration = model.best_iteration
      best_xgboost_rounds = (0, best_iteration + 1)
      
      precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, 
                                                                                   model.predict(x_train_transformed, iteration_range=best_xgboost_rounds), 
                                                                                   average='weighted')
    
      precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, 
                                                                                model.predict(x_test_transformed, iteration_range=best_xgboost_rounds), 
                                                                                average='weighted')
    
      train_auc = roc_auc_score(y_train, 
                                model.predict_proba(x_train_transformed, iteration_range=best_xgboost_rounds)[:,1],
                                average="weighted")

      # Capture and return fit statistcs from the Hyperopt Trial
      digits = 3
      metrics = OrderedDict()
      metrics["train_precision"]= round(precision_train, digits)
      metrics["train_recall"] =   round(recall_train, digits)
      metrics["train_f1"] =       round(f1_train, digits)
      metrics["train_auc"] =      round(train_auc, digits)
      metrics["test_precision"] = round(precision_test, digits)
      metrics["test_recall"] =    round(recall_test, digits)
      metrics["test_f1"] =        round(f1_test, digits)
      metrics["test_auc"] =       round(best_score, digits)
      metrics["best_iteration"] = round(best_iteration, digits)
    
      return {'status': STATUS_OK, 'loss': 1- best_score, 'metrics': metrics}
    
  
    trials = Trials()

    best_params = fmin(fn=hyperopt_objective_fn, 
                       space=parameter_search_space, 
                       algo=tpe.suggest,
                       max_evals=max_hyperopt_evals, 
                       trials=trials, 
                       rstate=np.random.default_rng(50),
                       early_stop_fn=no_progress_loss(iteration_stop_count=hyperopt_early_stopping_rounds, percent_increase=hyperopt_early_stopping_threshold))
        
    final_model_parameters = {}
    final_model_parameters['n_estimators'] = int(trials.best_trial['result']['metrics']['best_iteration'])

    # Adjust parameter data types to meet xgboost requirements
    for parameter, value in best_params.items():
          if parameter in ['max_depth']:
            final_model_parameters[parameter] = int(value)
          else:
            final_model_parameters[parameter] = value

    # Fit the pipeline and model on full dataset
    final_model = XGBClassifier(**final_model_parameters)
    final_pipeline = Pipeline([("preprocess", pipeline), ("classifier", final_model)])

    final_pipeline.fit(group_training_data, group_training_data[label_col])
    
    end = datetime.datetime.now()
    elapsed = end-start
    seconds = round(elapsed.total_seconds(), 1)
    
    mlflow.set_experiment(experiment_location)
    with mlflow.start_run(run_id = run_id) as run:
      
      # Log group model
      artifact_path =f'models/{group_name}'
      mlflow.sklearn.log_model(final_pipeline, artifact_path=artifact_path)
      
      # Log group parameters as csv
      parameters_column_names = ['parameter', 'value']
      parameters_csv_formatted = [{"parameter": parameter, "value": value} 
                                    for parameter, value in final_model_parameters.items()]
      
      parameters_file_name = '/parameters.csv'
      with open(parameters_file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=parameters_column_names)
        writer.writeheader()
        writer.writerows(parameters_csv_formatted)
        
      mlflow.log_artifact(parameters_file_name, artifact_path=artifact_path)
      
      # Log group metrics as csv
      best_model_metrics = trials.best_trial['result']['metrics']
      
      metrics_column_names = ['metric', 'value']
      metrics_csv_formatted = [{'metric': metric, "value": value} 
                                    for metric, value in best_model_metrics.items()]
      
      metrics_file_name = f'/metrics.csv'
      with open(metrics_file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_column_names)
        writer.writeheader()
        writer.writerows(metrics_csv_formatted)
        
      mlflow.log_artifact(metrics_file_name, artifact_path=artifact_path)
      

      # Construct dataframe output
      output = OrderedDict()
      output['group'] =                group_name
      output['mlflow_run_id'] =        run_id
      output['stage_id'] =             TaskContext().stageId()
      output['task_attempt_id'] =      task_attempt_id = TaskContext().taskAttemptId()
      output['start_time'] =           start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
      output['end_time'] =             end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
      output['elapsed_seconds'] =      seconds
      output['best_hyperopt_trial'] =  trials.best_trial['tid']
      output['best_params'] =          [final_model_parameters]

      # Delete XGBoost best interation (number of trees) from the metrics dict
      del best_model_metrics['best_iteration']
      output.update(best_model_metrics)
    
    return pd.DataFrame(output, index=[0])
  
  return train_model_hyperopt_mlflow_udf

# COMMAND ----------

# MAGIC %md ### Fitting the models and log to MLflow

# COMMAND ----------

id_cols = ['id']

# Add additional columns to out Spark Schema
spark_types = [('group',                StringType()),
               ('mlflow_run_id',        StringType()),
               ('stage_id',             IntegerType()),
               ('task_attempt_id',      IntegerType()),
               ('start_time',           StringType()),
               ('end_time',             StringType()),
               ('elapsed_seconds',      FloatType()),
               ('train_precision',      FloatType()),
               ('train_recall',         FloatType()),
               ('train_f1',             FloatType()),
               ('train_auc',            FloatType()),
               ('test_precision',       FloatType()),
               ('test_recall',          FloatType()),
               ('test_f1',              FloatType()),
               ('test_auc',             FloatType()),
               ('best_hyperopt_trial',  IntegerType()),
               ('best_params',          MapType(StringType(), FloatType()))]

spark_schema = StructType()
for col_name, spark_type in spark_types:
  spark_schema.add(col_name, spark_type)

pipeline = create_preprocessing_transform(categorical_features, numerical_features)

# Create MLflow parent run
run_id = get_new_run(experiment_location, "group_model_run")

# Configure Pandas UDF            
train_model_hyperopt_mlflow_udf = configure_model_hyperopt_mlflow_udf(label_col, 
                                                                      grouping_col, 
                                                                      id_cols,
                                                                      pipeline,
                                                                      parameter_search_space,
                                                                      experiment_location,
                                                                      run_id)

# Fit models by applying UDF  
best_model_stats = features.groupBy('group_name').applyInPandas(train_model_hyperopt_mlflow_udf, schema=spark_schema)

best_model_stats.write.mode('overwrite').format('delta').saveAsTable('default.best_model_stats')
display(spark.table('default.best_model_stats'))

# COMMAND ----------

# MAGIC %md ### Promoting the run to the Model Registry  

# COMMAND ----------

# MAGIC %md Create an [MLflow Client](https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html) instance

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

# MAGIC %md Create a [Model Registry](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html) entry if one does not exist

# COMMAND ----------

model_registry_name = 'pandas_udf_models'
try:
  client.get_registered_model(model_registry_name)
  print(" Registered model already exists")
except:
  client.create_registered_model(model_registry_name)

# COMMAND ----------

# MAGIC %md Create an entry for the model in the registry

# COMMAND ----------

model_info = client.get_run(run_id).to_dictionary()
artifact_uri = model_info['info']['artifact_uri']


registered_model = client.create_model_version(
                     name = model_registry_name,
                     source = artifact_uri + "/meta_model",
                     run_id = run_id
                    )

# COMMAND ----------

# MAGIC %md Move the registered model to the "Production" stage

# COMMAND ----------

promote_to_prod = client.transition_model_version_stage(name=model_registry_name,
                                                        version = int(registered_model.version),
                                                        stage="Production",
                                                        archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md ### Create an inference UDF
# MAGIC Our PandasUDF for inference is reletively simple. We will configure the PandasUDF to load our Production meta model from the Model Registry. Our helper function will capture the model's unique identifier, which we will use to load the model into the notebook.
# MAGIC 
# MAGIC Similar to our model training UDF, our inference UDF will extract the group name of the data it receives. Then, the UDF will load the approriate group-level model and score the group's data.  
# MAGIC 
# MAGIC Our UDF will return the group name, unique id of the record, and the classification probabilities.

# COMMAND ----------

def get_model_info(model_name:str, stage:str) -> str:
  """
  Given a the name of a registered model and a Model Registry stage,
  return the model's unique run id
  """
  
  from mlflow.tracking import MlflowClient
  
  client = MlflowClient()
  
  run_info = [run for run in client.search_model_versions(f"name='{model_name}'") 
                  if run.current_stage == stage][0]
  
  return run_info.source

# COMMAND ----------

# MAGIC %md Define the PandasUDF for applying the meta_model to groups of data

# COMMAND ----------

def inference_model_config(model_name:str, registry_stage:str='Production') -> Callable[[pd.DataFrame], pd.DataFrame]:
  """
  Load a model from the Model Registry and return a PandasUDF function. The PandasUDF will apply 
  the models to different groups of data via the groupBy.applyInPandas method.
  """
  
  model_artifact_location = get_model_info(model_name, registry_stage)
  
  model = mlflow.pyfunc.load_model(model_artifact_location)
  
  def apply_models(model_input):
    
    predictions = model.predict(model_input)
    
    return predictions
  
  return apply_models

# COMMAND ----------

# MAGIC %md Specify the Spark Dataframe schema that maps to the UDF's output. Apply the inference UDF to generate predictions.

# COMMAND ----------

prediction_schema = StructType()
prediction_schema.add('group_name', StringType())
prediction_schema.add('id', IntegerType())
prediction_schema.add('probabilities', ArrayType(FloatType()))

inference_model = inference_model_config(model_name = model_registry_name)

predictions = features.groupBy('group_name').applyInPandas(inference_model, schema=prediction_schema)

display(predictions)
