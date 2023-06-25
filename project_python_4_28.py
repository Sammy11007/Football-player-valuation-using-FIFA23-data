#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.cloud import storage
import pandas as pd
import io
from pyspark.sql import SparkSession

spark = SparkSession.builder     .appName('my-spark-job')     .config('spark.jars.packages', 'com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.22.1')     .config('spark.hadoop.fs.gs.auth.service.account.enable', 'true')     .config('spark.sql.catalog.spark_bigquery', 'com.google.cloud.spark.bigquery.v2.BigQueryCatalog')     .config('spark.sql.catalog.spark_bigquery.projectId', 'arched-glass-384816')     .getOrCreate()




spark



df = spark.read.format("csv")       .option("header", "true")       .load("gs://sammy-project-bucket/out.csv")




df.show()




from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as F
import pyspark.sql.types as T


# In[9]:


df.dropna()
df.show()


# In[10]:


df2 = df.where(df.player_positions!='GK')
df2.show()


# In[11]:


df3 = df2.drop(df2.preferred_foot)



# In[12]:


from pyspark.sql.functions import concat, col, lit
df3 = df3.withColumn('fixed_pos', concat(df3.player_positions.substr(1, 3)))
df3.show()


# In[13]:


df3.select('fixed_pos').distinct().collect()


# In[14]:


from pyspark.sql.functions import when
from pyspark.sql.functions import regexp_replace
df3 = df3.withColumn('fixed_pos', 
               when(col('fixed_pos') == 'CB,', 'CB')
               .when(col('fixed_pos') == 'LB,', 'LB')
               .when(col('fixed_pos') == 'RB,', 'RB')
               .when(col('fixed_pos') == 'CM,', 'CM')
               .when(col('fixed_pos') == 'LM,', 'LM')
               .when(col('fixed_pos') == 'RM,', 'RM')
               .when(col('fixed_pos') == 'LW,', 'LW')
               .when(col('fixed_pos') == 'RW,', 'RW')
               .when(col('fixed_pos') == 'CF,', 'CF')
               .when(col('fixed_pos') == 'ST,', 'ST')
               .otherwise(col('fixed_pos'))
              )



# In[15]:


df3.select('fixed_pos').distinct().collect()


# In[16]:


df3 =  df3.withColumn(
    "fixed_pos",
    when(col("fixed_pos").isin('CB', 'LB', 'RB', 'RWB', 'LWB'),"DEF")
    .when(col("fixed_pos").isin("CM", "CAM", "RM", "LM", "CDM"),"MID")
    .when(col("fixed_pos").isin("ST", "CF", "LW", "RW"),"FWD")
    .otherwise(col("fixed_pos"))
)
df3.show()


# In[17]:


df3 = df3.drop(df3.player_positions)
df3 = df3.drop(df3.short_name)

# In[18]:


df3 = df3.withColumn('work_rate', 
               when(col('work_rate') == 'Low/Low', 1)
               .when(col('work_rate') == 'Low/Medium', 2)
               .when(col('work_rate') == 'Medium/Low', 2)
               .when(col('work_rate') == 'Medium/Medium', 3)
               .when(col('work_rate') == 'Low/High', 3)
               .when(col('work_rate') == 'High/Low', 3)
               .when(col('work_rate') == 'Medium/High', 4)
               .when(col('work_rate') == 'High/Medium', 4)
               .when(col('work_rate') == 'High/High', 5)
               .otherwise(col('work_rate'))
              )

# In[19]:


from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="fixed_pos", outputCol="fixed_pos_dummies")
indexed = indexer.fit(df3).transform(df3)
indexed.show()


# In[20]:


df4 = indexed.drop(indexed.fixed_pos)
df4.show()


# In[21]:


df4.na.drop().show()


# In[22]:


from pyspark.sql.types import IntegerType
df4 = df4.withColumn("work_rate", df4["work_rate"].cast(IntegerType()))
df4 = df4.withColumn("overall", df4["overall"].cast(IntegerType()))
df4 = df4.withColumn("value_eur", df4["value_eur"].cast(IntegerType()))
df4 = df4.withColumn("wage_eur", df4["wage_eur"].cast(IntegerType()))
df4 = df4.withColumn("age", df4["age"].cast(IntegerType()))
df4 = df4.withColumn("height_cm", df4["height_cm"].cast(IntegerType()))
df4 = df4.withColumn("weak_foot", df4["weak_foot"].cast(IntegerType()))
df4 = df4.withColumn("skill_moves", df4["skill_moves"].cast(IntegerType()))
df4 = df4.withColumn("pace", df4["pace"].cast(IntegerType()))
df4 = df4.withColumn("shooting", df4["shooting"].cast(IntegerType()))
df4 = df4.withColumn("passing", df4["passing"].cast(IntegerType()))
df4 = df4.withColumn("dribbling", df4["dribbling"].cast(IntegerType()))
df4 = df4.withColumn("defending", df4["defending"].cast(IntegerType()))
df4 = df4.withColumn("physic", df4["physic"].cast(IntegerType()))
df4 = df4.withColumn("weight_kg", df4["weight_kg"].cast(IntegerType()))
df4 = df4.withColumn("potential", df4["potential"].cast(IntegerType()))


# In[23]:


df4.printSchema()


# In[24]:


df4.show()


# In[25]:


#apply ml algos


# In[26]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# In[27]:


from pyspark.sql.types import DoubleType
df4 = df4.withColumn("work_rate", df4["work_rate"].cast(DoubleType()))
df4 = df4.withColumn("overall", df4["overall"].cast(DoubleType()))
df4 = df4.withColumn("value_eur", df4["value_eur"].cast(DoubleType()))
df4 = df4.withColumn("wage_eur", df4["wage_eur"].cast(DoubleType()))
df4 = df4.withColumn("age", df4["age"].cast(DoubleType()))
df4 = df4.withColumn("height_cm", df4["height_cm"].cast(DoubleType()))
df4 = df4.withColumn("weak_foot", df4["weak_foot"].cast(DoubleType()))
df4 = df4.withColumn("skill_moves", df4["skill_moves"].cast(DoubleType()))
df4 = df4.withColumn("pace", df4["pace"].cast(DoubleType()))
df4 = df4.withColumn("shooting", df4["shooting"].cast(DoubleType()))
df4 = df4.withColumn("passing", df4["passing"].cast(DoubleType()))
df4 = df4.withColumn("dribbling", df4["dribbling"].cast(DoubleType()))
df4 = df4.withColumn("defending", df4["defending"].cast(DoubleType()))
df4 = df4.withColumn("physic", df4["physic"].cast(DoubleType()))
df4 = df4.withColumn("weight_kg", df4["weight_kg"].cast(DoubleType()))
df4 = df4.withColumn("potential", df4["potential"].cast(DoubleType()))
df4 = df4.withColumn("value_eur", df4["value_eur"].cast(DoubleType()))


# In[28]:


df4 = df4.dropna()
 
assembler = VectorAssembler(inputCols=['overall',
 'potential',
 'wage_eur',
 'age',
 'height_cm',
 'weight_kg',
 'weak_foot',
 'skill_moves',
 'work_rate',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic'], outputCol="features")
output = assembler.transform(df4)
final_data = output.select("features", "value_eur")


# In[29]:


final_data.cache()
final_data


# In[30]:


train_data, test_data = final_data.randomSplit([0.7, 0.3])


# In[31]:


from pyspark.ml.regression import LinearRegression
 
lr = LinearRegression(featuresCol="features", labelCol="value_eur")
lr_model = lr.fit(train_data)


# In[32]:


print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[33]:


#Summarize the model over the training set and print out some metrics:
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("R squre: %f" % trainingSummary.r2)


# In[34]:


lr_predictions = lr_model.transform(test_data)
lr_predictions.select("prediction","value_eur","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="value_eur",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))


# In[35]:


test_result = lr_model.evaluate(test_data)
print("Root Mean Squared Error (RMSE) on test data = %f" % test_result.rootMeanSquaredError)


# In[51]:


from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol="features", labelCol="value_eur", numTrees=10, maxDepth=5, seed=42)
model = rf.fit(train_data)


# In[52]:


predictions = model.transform(test_data)


# In[54]:


evaluator = RegressionEvaluator(metricName="rmse", predictionCol="prediction", labelCol="value_eur")
rmse = evaluator.evaluate(predictions)
print("RMSE on test data = %g" % rmse)




