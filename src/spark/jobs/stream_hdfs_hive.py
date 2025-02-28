from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType
from pyspark.sql.functions import col

SOURCE_PATH         = 'hdfs://namenode:9000/vdata/streamdata'
CHECKPOINT_LOCATION = 'hdfs://namenode:9000/vdata/checkpoints'

spark = SparkSession.builder \
  .appName("Streams DATA LAKE -> DATAWAREHOUSE") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .config("spark.sql.catalogImplementation", "hive") \
  .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
  .enableHiveSupport() \
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
  
# df = spark.readStream \
#   .format("csv") \
#   .option("header", "true") \
#   .schema(schema) \
#   .load(SOURCE_PATH)

sc = spark.sparkContext

# df = spark.read.parquet(SOURCE_PATH)

# rdd = 

# df.show(truncate=False)

spark.stop()

# query = df.writeStream \
#   .outputMode("append") \
#   .format("parquet") \
#   .option("path", OUTPUT_PATH) \
#   .option("checkpointLocation", CHECKPOINT_LOCATION) \
#   .start()

# query.awaitTermination()
  
  

  
  

    
  





    