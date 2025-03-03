from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, expr

INPUT_PATH='hdfs://namenode:9000/messages/bulk'

spark = SparkSession.builder \
  .appName("Test messages bulk") \
  .getOrCreate()
  
  
df = spark.read \
  .option("multiline", "true") \
  .option("mode", "PERMISSIVE") \
  .json(INPUT_PATH)

json_schema = spark.read.json(df.select("value").rdd.map(lambda x: x[0])).schema
parsed_df = df.select(from_json(col("value"), json_schema).alias("parsed_data"))
rdf = parsed_df.select("parsed_data.*")
rdf = rdf.select(
  col("id"),
  to_timestamp(col("applicationPeriods")[0]["begin"], "yyyyMMdd'T'HHmmss").alias("begin"),
  to_timestamp(col("applicationPeriods")[0]["end"], "yyyyMMdd'T'HHmmss").alias("end"),
  col("cause"),
  col("lastUpdate"),
  col("message"),
  col("severity").alias("pattern"),
  col("title")
)
  
rdf.show()
  
spark.stop()