from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType
from pyspark.sql.functions import col

SOURCE_PATH       = 'file:///data/validations'
OUTPUT_PATH         = 'hdfs://namenode:9000/vdata/streamdata'
CHECKPOINT_LOCATION = 'hdfs://namenode:9000/vdata/checkpoints'

spark = SparkSession.builder \
  .appName("Streams RAW DATA -> HDFS") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .getOrCreate()
  
schema = StructType() \
  .add("JOUR", StringType()) \
  .add("CODE_STIF_TRNS", StringType()) \
  .add("CODE_STIF_RES", StringType()) \
  .add("CODE_STIF_ARRET", StringType()) \
  .add("LIBELLE_ARRET", StringType()) \
  .add("lda", StringType()) \
  .add("CATEGORIE_TITRE", StringType()) \
  .add("NB_VALD", StringType()) \
  .add("value", StringType())
  
df = spark.readStream \
  .format("csv") \
  .option("header", "true") \
  .schema(schema) \
  .load(SOURCE_PATH)

query = df.writeStream \
  .outputMode("append") \
  .format("parquet") \
  .option("path", OUTPUT_PATH) \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .start()

query.awaitTermination()
  
  

  
  

    
  





    