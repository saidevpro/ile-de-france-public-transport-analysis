from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, avg as _avg, dayofmonth, col, when, dayofweek
from os.path import abspath

SOURCE_DATA = 'hdfs://namenode:9000/hive/warehouse/idf.db/messages'
CHECKPOINT_LOCATION = 'hdfs://namenode:9000/messages/checkpoints_mysql_raw/'
postgres_url = "jdbc:postgresql://postgres-db:5432/streams"
postgres_properties = {
  "user": "root",
  "password": "root",
  "driver": "org.postgresql.Driver"
}

spark = SparkSession.builder \
  .appName("Streams Validations Data to MYSQL") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .config("spark.sql.catalogImplementation", "hive") \
  .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
  .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/hive/warehouse") \
  .config("spark.sql.streaming.schemaInference", "true") \
  .enableHiveSupport() \
  .getOrCreate()

schema = spark.table("idf.messages").schema

df = spark.readStream \
  .schema(schema) \
  .format("parquet") \
  .load(SOURCE_DATA)
  
def handleBatch(df, epoch_id):
  df.write \
    .jdbc(url=postgres_url,
      table="stream_messages",
      mode="append",
      properties=postgres_properties
    )
  

query = df \
  .writeStream \
  .foreachBatch(handleBatch) \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .start()
  
query.awaitTermination()

  
  
  
