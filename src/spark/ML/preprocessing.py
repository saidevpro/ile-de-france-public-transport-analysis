from pyspark.sql import SparkSession
from pyspark.sql.functions import (
  col, udf, regexp_replace, lower, trim, split, size, when, datediff, 
  hour, dayofweek, month, year, lit, to_date, length, array_contains,
  from_unixtime, unix_timestamp, expr
)
from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType, BooleanType
import re
from datetime import datetime
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml import Pipeline

spark = SparkSession.builder \
  .appName("Train Station Messages Enhancement") \
  .enableHiveSupport() \
  .getOrCreate()

input_file = "file:///data/messages_bulk/*.csv"
output_path = "file:///data/ml"

df = spark.read \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .option("delimiter", ",") \
  .option("encoding", "UTF-8") \
  .csv(input_file)


# Does text contains pattern
@udf(returnType=BooleanType())
def contains_pattern(text, pattern):
  if text is None:
    return False
  return bool(re.search(pattern, text.lower()))

# Extract transport line mentions (bus, train, metro, etc.)
@udf(returnType=ArrayType(StringType()))
def extract_transport_lines(text):
  if text is None:
    return []
  
  pattern = r"\b(ligne|bus|train|rer|métro|tramway|tram)\b\s*(\d+|[A-E])"
  matches = re.findall(pattern, text.lower())
  return [f"{match[0]} {match[1]}" for match in matches]


# Extract station names mentioned in the text
@udf(returnType=ArrayType(StringType()))
def extract_stations(text):
  if text is None:
    return []
  
  pattern = r"gare de ([^,.:;<>]+)"
  matches = re.findall(pattern, text.lower())
  return [match.strip() for match in matches]


#Convert cause to a numerical severity level
@udf(returnType=IntegerType())
def categorize_severity(cause):
  severity_map = {
    "INFORMATION": 1,
    "PERTURBATION": 2,
    "TRAVAUX": 3,
    "ACCIDENT": 4
  }
  return severity_map.get(cause, 2)


#Categorize messages by length
@udf(returnType=StringType())
def categorize_string_length(word_count):
  if word_count < 30:
    return "short"
  elif word_count < 100:
    return "medium"
  else:
    return "long"

# Create new column text_content from cleaned
df = df.withColumn("text_content", 
  regexp_replace(
    regexp_replace(
      regexp_replace(col("message"), "<[^>]+>", " "), 
      "&nbsp;", " "
    ),
    "\\s+", " "
))

# Calculate duration
df = df.withColumn("begin_date", to_date(col("begin"))) \
      .withColumn("end_date", to_date(col("end"))) \
      .withColumn("display_duration_hours", 
      (unix_timestamp(col("end")) - unix_timestamp(col("begin"))) / 3600)

# Count words in message
tokenizer = RegexTokenizer(inputCol="text_content", outputCol="words", pattern="\\s+")
df = tokenizer.transform(df)
df = df.withColumn("word_count", size(col("words")))

# Add message length category
df = df.withColumn("message_length_category", categorize_string_length(col("word_count")))

# 5. Extract transportation types
df = df.withColumn("transport_types", extract_transport_lines(col("text_content")))
df = df.withColumn("transportation_types", 
                 expr("array_join(transport_types, '|')"))
df = df.withColumn("transport_count", size(col("transport_types")))

# 6. Extract stations
df = df.withColumn("station_list", extract_stations(col("text_content")))
df = df.withColumn("stations_mentioned", 
                 expr("array_join(station_list, '|')"))
df = df.withColumn("stations_count", size(col("station_list")))

# 7. Binary feature flags
route_change_pattern = r"itinéraire|dévié|déviation"
closure_pattern = r"fermé|fermée|fermeture|non desservi"
delay_pattern = r"retard|perturbation|ralentissement"
schedule_pattern = r"horaire|à partir de|jusqu'au"
morning_pattern = r"matin|aube|matinée|avant-midi"
evening_pattern = r"soir|soirée|nuit"
weekend_pattern = r"week-end|weekend|samedi|dimanche"

df = df.withColumn("is_route_change", 
  when(contains_pattern(col("text_content"),
  lit(route_change_pattern)), 1).otherwise(0)
)

df = df.withColumn("is_station_closure", 
  when(contains_pattern(col("text_content"),
  lit(closure_pattern)), 1).otherwise(0)
)

df = df.withColumn("is_delay_notification", 
  when(contains_pattern(col("text_content"),
  lit(delay_pattern)), 1).otherwise(0)
)

df = df.withColumn("is_schedule_change", 
  when(contains_pattern(col("text_content"),
  lit(schedule_pattern)), 1).otherwise(0)
)

df = df.withColumn("mentions_morning", 
  when(contains_pattern(col("text_content"),
  lit(morning_pattern)), 1).otherwise(0)
)


df = df.withColumn("mentions_evening", 
  when(contains_pattern(col("text_content"),
  lit(evening_pattern)), 1).otherwise(0)
)

df = df.withColumn("mentions_weekend", 
  when(contains_pattern(col("text_content"),
  lit(weekend_pattern)), 1).otherwise(0)
)

# Derive severity and urgency
df = df.withColumn("severity_level", categorize_severity(col("cause")))
df = df.withColumn("is_urgent", 
  when((col("severity_level") >= 3) & (col("display_duration_hours") <= 24), 1).otherwise(0))

# Time features
df = df.withColumn("day_of_week", dayofweek(to_date(col("begin"))))
df = df.withColumn("month_of_year", month(to_date(col("begin"))))
df = df.withColumn("year", year(to_date(col("begin"))))
df = df.withColumn("hour_of_day", hour(col("begin")))
df = df.withColumn("is_weekend", 
  when(col("day_of_week").isin(1, 7), 1).otherwise(0))

# Message features for clustering
df = df.withColumn("message_hash", expr("hash(text_content)"))
df = df.withColumn("title_length", length(col("title")))

df.write.mode("overwrite").format("json").save(f"{output_path}/train_file.json")

# Create text features for ML
print("Creating text features using TF-IDF")

# Create a pipeline for text processing
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", 
  stopWords=StopWordsRemover.loadDefaultStopWords("french"))

# Create term frequency features
cv = CountVectorizer(inputCol="filtered_words", outputCol="tf_features", 
  minDF=2.0, vocabSize=10000)

# Create TF-IDF features
idf = IDF(inputCol="tf_features", outputCol="text_features")

# Create a pipeline for the text processing steps
pipeline = Pipeline(stages=[remover, cv, idf])

# Fit the pipeline to the data
text_model = pipeline.fit(df)
df = text_model.transform(df)

# Save the enhanced data
# print(f"Saving enhanced data to {output_path}")
df.write \
  .mode("overwrite") \
  .parquet(f"{output_path}/models")

# # Save the text processing model
text_model \
  .write() \
  .overwrite() \
  .save(f"{output_path}/text_model")


spark.stop()