from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, StructField
from pyspark.sql.functions import col, from_json, to_timestamp, expr, year

SOURCE_PATH='hdfs://namenode:9000/messages/bulk'
CHECKPOINT_LOCATION='hdfs://namenode:9000/messages/batch_hive'

spark = SparkSession.builder \
  .appName("Streams MESSAGES FROM HDFS") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .config("spark.sql.catalogImplementation", "hive") \
  .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
  .config("hive.exec.dynamic.partition", "true") \
  .config("hive.exec.dynamic.partition.mode", "nonstrict") \
  .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/hive/warehouse") \
  .enableHiveSupport() \
  .getOrCreate()
  
spark.sql("CREATE DATABASE IF NOT EXISTS idf;")

schema = StructType([StructField("value", StringType(), True)])

df = spark.readStream \
  .option("multiline", "true") \
  .option("mode", "PERMISSIVE") \
  .schema(schema) \
  .json(SOURCE_PATH)
  
  
def process_batch_data(df, epoch_id): 
  json_schema = spark.read.json(df.select("value").rdd.map(lambda x: x[0])).schema
  parsed_df = df.select(from_json(col("value"), json_schema).alias("parsed_data"))
  rdf = parsed_df.select("parsed_data.*")
  
  rdf = rdf.select(
    col("id"),
    to_timestamp(col("applicationPeriods")[0]["begin"], "yyyyMMdd'T'HHmmss").alias("begin"),
    to_timestamp(col("applicationPeriods")[0]["end"], "yyyyMMdd'T'HHmmss").alias("end"),
    col("cause"),
    to_timestamp(col("lastUpdate"), "yyyyMMdd'T'HHmmss").alias("update_date"),
    col("message"),
    col("severity").alias("pattern"),
    col("title")
  ).withColumn("year", year(col("begin")))
  
  rdf.write \
  .mode("append") \
  .format("parquet") \
  .partitionBy("year") \
  .saveAsTable("idf.messages")

df.writeStream \
  .foreachBatch(process_batch_data) \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .start() \
  .awaitTermination()
  

