from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, DateType
from pyspark.sql.functions import col

TOPIC_NAME="idfm_message"
KAFKA_HOST="kafka:9092"
OUTPUT_PATH='hdfs://namenode:9000/messages/bulk'
CHECKPOINT_LOCATION="hdfs://namenode:9000/messages/batch"

spark = SparkSession.builder \
  .appName("Streams READ MESSAGES FROM KAFKA") \
  .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
  .getOrCreate()
  
df = spark.readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", KAFKA_HOST) \
  .option("subscribe", TOPIC_NAME) \
  .load()
  
def process_batch_kafka(df, epoch_id):
  dft = df.selectExpr("CAST(value AS STRING)")
  dft.write \
    .mode("append") \
    .json(OUTPUT_PATH)
  
  
# STREAM ON TEXT FILES
df.writeStream \
  .foreachBatch(process_batch_kafka) \
  .outputMode("append") \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .start() \
  .awaitTermination()


  
    