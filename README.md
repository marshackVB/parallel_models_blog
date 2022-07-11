# Training many machine learning models in parallel using Databricks and PandasUDFs  

The Databricks Notebook within this repository provides a detailed, step-by-step example of training multiple machine learning models in parallel on different datasets. It includes the following steps.

 - Configuring the Databricks Cluster
 - Leveraging PandasUDFs to train machine learning models in parallel on different groups of a dataset.
 - Tuning model parameters using Hyperopt
 - Logging multiple models to a single MLflow Experiment Run
 - Applying multiple models for inference to different groups of data in parallel


 This repository can be cloned into a Databricks Repo; the code is self contained and can be run in any Databricks environment. The most recent testing of this notebook leveraged the Databricks ML Runtime version 10.5.