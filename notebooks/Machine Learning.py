# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/ml_lab
# MAGIC  mkdir /dbfs/ml_lab
# MAGIC  wget -O /dbfs/ml_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load("/ml_lab/penguins.csv")
display(df)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
   
data = df.dropna().select(col("Island").astype("string"),
                           col("CulmenLength").astype("float"),
                          col("CulmenDepth").astype("float"),
                          col("FlipperLength").astype("float"),
                          col("BodyMass").astype("float"),
                          col("Species").astype("int")
                          )
display(data)

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
print ("Training Rows:", train.count(), " Testing Rows:", test.count())

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="Island", outputCol="IslandIdx")
indexedData = indexer.fit(train).transform(train).drop("Island")
display(indexedData)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# Create a vector column containing all numeric features
numericFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
numericColVector = VectorAssembler(inputCols=numericFeatures, outputCol="numericFeatures")
vectorizedData = numericColVector.transform(indexedData)
   
# Use a MinMax scaler to normalize the numeric values in the vector
minMax = MinMaxScaler(inputCol = numericColVector.getOutputCol(), outputCol="normalizedFeatures")
scaledData = minMax.fit(vectorizedData).transform(vectorizedData)
   
# Display the data with numeric feature vectors (before and after scaling)
compareNumerics = scaledData.select("numericFeatures", "normalizedFeatures")
display(compareNumerics)

# COMMAND ----------

featVect = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="featuresVector")
preppedData = featVect.transform(scaledData)[col("featuresVector").alias("features"), col("Species").alias("label")]
display(preppedData)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3)
model = lr.fit(preppedData)
print ("Model trained!")

# COMMAND ----------

# Prepare the test data
indexedTestData = indexer.fit(test).transform(test).drop("Island")
vectorizedTestData = numericColVector.transform(indexedTestData)
scaledTestData = minMax.fit(vectorizedTestData).transform(vectorizedTestData)
preppedTestData = featVect.transform(scaledTestData)[col("featuresVector").alias("features"), col("Species").alias("label")]
   
# Get predictions
prediction = model.transform(preppedTestData)
predicted = prediction.select("features", "probability", col("prediction").astype("Int"), col("label").alias("trueLabel"))
display(predicted)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
   
# Simple accuracy
accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
print("Accuracy:", accuracy)
   
# Individual class metrics
labels = [0,1,2]
print("\nIndividual class metrics:")
for label in sorted(labels):
    print ("Class %s" % (label))
   
    # Precision
    precision = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                evaluator.metricName:"precisionByLabel"})
    print("\tPrecision:", precision)
   
    # Recall
    recall = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                             evaluator.metricName:"recallByLabel"})
    print("\tRecall:", recall)
   
    # F1 score
    f1 = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                         evaluator.metricName:"fMeasureByLabel"})
    print("\tF1 Score:", f1)
   
# Weighted (overall) metrics
overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
print("Overall Precision:", overallPrecision)
overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
print("Overall Recall:", overallRecall)
overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
print("Overall F1 Score:", overallF1)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
   
catFeature = "Island"
numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   
# Define the feature engineering and model training algorithm steps
catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
algo = LogisticRegression(labelCol="Species", featuresCol="Features", maxIter=10, regParam=0.3)
   
# Chain the steps as stages in a pipeline
pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
# Use the pipeline to prepare data and fit the model algorithm
model = pipeline.fit(train)
print ("Model trained!")

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))
display(predicted)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import DecisionTreeClassifier
   
catFeature = "Island"
numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   
# Define the feature engineering and model steps
catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
algo = DecisionTreeClassifier(labelCol="Species", featuresCol="Features", maxDepth=10)
   
# Chain the steps as stages in a pipeline
pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
# Use the pipeline to prepare data and fit the model algorithm
model = pipeline.fit(train)
print ("Model trained!")

# COMMAND ----------

# Get predictions
prediction = model.transform(test)
predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))
   
# Generate evaluation metrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction")
   
# Simple accuracy
accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
print("Accuracy:", accuracy)
   
# Class metrics
labels = [0,1,2]
print("\nIndividual class metrics:")
for label in sorted(labels):
    print ("Class %s" % (label))
   
    # Precision
    precision = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                    evaluator.metricName:"precisionByLabel"})
    print("\tPrecision:", precision)
   
    # Recall
    recall = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                             evaluator.metricName:"recallByLabel"})
    print("\tRecall:", recall)
   
    # F1 score
    f1 = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                         evaluator.metricName:"fMeasureByLabel"})
    print("\tF1 Score:", f1)
   
# Weighed (overall) metrics
overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
print("Overall Precision:", overallPrecision)
overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
print("Overall Recall:", overallRecall)
overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
print("Overall F1 Score:", overallF1)

# COMMAND ----------

model.save("/models/penguin.model")

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel

persistedModel = PipelineModel.load("/models/penguin.model")
   
newData = spark.createDataFrame ([{"Island": "Biscoe",
                                  "CulmenLength": 47.6,
                                  "CulmenDepth": 14.5,
                                  "FlipperLength": 215,
                                  "BodyMass": 5400}])
   
   
predictions = persistedModel.transform(newData)
display(predictions.select("Island", "CulmenDepth", "CulmenLength", "FlipperLength", "BodyMass", col("prediction").alias("PredictedSpecies")))
