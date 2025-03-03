from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, DateType
from pyspark.sql.functions import col

SOURCE_PATH_CSV       = 'file:///data/validations/*.csv'
SOURCE_PATH_TXT      = 'file:///data/validations/*.txt'
OUTPUT_PATH         = 'hdfs://namenode:9000/vdata/streamdata'
CHECKPOINT_LOCATION = 'hdfs://namenode:9000/vdata/checkpoints'

spark = SparkSession.builder \
  .appName("Streams SOURCES DATA -> HDFS") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .getOrCreate()
  
schema = StructType() \
  .add("JOUR", DateType()) \
  .add("CODE_STIF_TRNS", IntegerType()) \
  .add("CODE_STIF_RES", IntegerType()) \
  .add("CODE_STIF_ARRET", IntegerType()) \
  .add("LIBELLE_ARRET", StringType()) \
  .add("ID_ZDC", IntegerType()) \
  .add("CATEGORIE_TITRE", StringType()) \
  .add("NB_VALD", IntegerType()) \
  
# STREAM ON CSV FILES
qc = spark.readStream \
  .format("csv") \
  .option("header", "true") \
  .option("delimiter", ";")\
  .schema(schema) \
  .csv(SOURCE_PATH_CSV) \
  .writeStream \
  .outputMode("append") \
  .format("csv") \
  .option("header", "true") \
  .option("path", OUTPUT_PATH) \
  .option("checkpointLocation", f"{CHECKPOINT_LOCATION}/csv") \
  .start()
  
# STREAM ON TEXT FILES
qt = spark.readStream \
  .format("text") \
  .text(SOURCE_PATH_TXT) \
  .writeStream \
  .outputMode("append") \
  .format("text") \
  .option("path", OUTPUT_PATH) \
  .option("checkpointLocation", f"{CHECKPOINT_LOCATION}/txt") \
  .start()
  
qc.awaitTermination()
qt.awaitTermination()

  
    