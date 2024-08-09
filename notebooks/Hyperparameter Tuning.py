# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/hyperopt_lab
# MAGIC  mkdir /dbfs/hyperopt_lab
# MAGIC  wget -O /dbfs/hyperopt_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
   
data = spark.read.format("csv").option("header", "true").load("/hyperopt_lab/penguins.csv")
data = data.dropna().select(col("Island").astype("string"),
                          col("CulmenLength").astype("float"),
                          col("CulmenDepth").astype("float"),
                          col("FlipperLength").astype("float"),
                          col("BodyMass").astype("float"),
                          col("Species").astype("int")
                          )
display(data.sample(0.2))
   
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
print ("Training Rows:", train.count(), " Testing Rows:", test.count())

# COMMAND ----------

from hyperopt import STATUS_OK
import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
def objective(params):
    # Train a model using the provided hyperparameter value
    catFeature = "Island"
    numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
    numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
    numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
    featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
    mlAlgo = DecisionTreeClassifier(labelCol="Species",    
                                    featuresCol="Features",
                                    maxDepth=params['MaxDepth'], maxBins=params['MaxBins'])
    pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, mlAlgo])
    model = pipeline.fit(train)
       
    # Evaluate the model to get the target metric
    prediction = model.transform(test)
    eval = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction", metricName="accuracy")
    accuracy = eval.evaluate(prediction)
       
    # Hyperopt tries to minimize the objective function, so you must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

from hyperopt import fmin, tpe, hp
   
# Define a search space for two hyperparameters (maxDepth and maxBins)
search_space = {
    'MaxDepth': hp.randint('MaxDepth', 10),
    'MaxBins': hp.choice('MaxBins', [10, 20, 30])
}
   
# Specify an algorithm for the hyperparameter optimization process
algo=tpe.suggest
   
# Call the training function iteratively to find the optimal hyperparameter values
argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=6)
   
print("Best param values: ", argmin)

# COMMAND ----------

from hyperopt import Trials
   
# Create a Trials object to track each run
trial_runs = Trials()
   
argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=3,
  trials=trial_runs)
   
print("Best param values: ", argmin)
   
# Get details from each trial run
print ("trials:")
for trial in trial_runs.trials:
    print ("\n", trial)
