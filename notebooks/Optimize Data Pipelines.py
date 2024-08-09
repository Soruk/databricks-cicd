# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/nyc_taxi_trips
# MAGIC  mkdir /dbfs/nyc_taxi_trips
# MAGIC  wget -O /dbfs/nyc_taxi_trips/yellow_tripdata_2021-01.parquet https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/yellow_tripdata_2021-01.parquet

# COMMAND ----------

 # Load the dataset into a DataFrame
 df = spark.read.parquet("/nyc_taxi_trips/yellow_tripdata_2021-01.parquet")
 display(df)

# COMMAND ----------

  df = (spark.readStream
          .format("cloudFiles")
          .option("cloudFiles.format", "parquet")
          .option("cloudFiles.schemaLocation", "/stream_data/nyc_taxi_trips/schema")
          .load("/nyc_taxi_trips/"))
  df.writeStream.format("delta") \
      .option("checkpointLocation", "/stream_data/nyc_taxi_trips/checkpoints") \
      .option("mergeSchema", "true") \
      .start("/delta/nyc_taxi_trips")
  display(df)

# COMMAND ----------

# MAGIC  %sh
# MAGIC  rm -r /dbfs/nyc_taxi_trips
# MAGIC  mkdir /dbfs/nyc_taxi_trips
# MAGIC  wget -O /dbfs/nyc_taxi_trips/yellow_tripdata_2021-02_edited.parquet https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/yellow_tripdata_2021-02_edited.parquet

# COMMAND ----------

 from pyspark.sql.functions import lit, rand

 # Convert streaming DataFrame back to batch DataFrame
 df = spark.read.parquet("/nyc_taxi_trips/*.parquet")
     
 # Add a salt column
 df_salted = df.withColumn("salt", (rand() * 100).cast("int"))

 # Repartition based on the salted column
 df_salted.repartition("salt").write.format("delta").mode("overwrite").save("/delta/nyc_taxi_trips_salted")

 display(df_salted)

# COMMAND ----------

 from delta.tables import DeltaTable

 delta_table = DeltaTable.forPath(spark, "/delta/nyc_taxi_trips")
 delta_table.optimize().executeCompaction()b

# COMMAND ----------

 delta_table.optimize().executeZOrderBy("tpep_pickup_datetime")
