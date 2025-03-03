from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, StructField
from pyspark.sql.functions import col, from_json, to_timestamp, expr

spark = SparkSession.builder \
  .appName("Streams MESSAGES FROM HDFS") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .config("spark.sql.catalogImplementation", "hive") \
  .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
  .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/hive/warehouse") \
  .enableHiveSupport() \
  .getOrCreate()
  
  
df = spark.sql(f"SELECT * FROM idf.messages_1")
  
df.coalesce(1).write \
  .mode("overwrite") \
  .format("csv") \
  .option("header", "true") \
  .option("dateFormat", "yyyy-MM-dd") \
  .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
  .csv("file:///data/messages_bulk")
  
  
spark.stop()