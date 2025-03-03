from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, DateType
from pyspark.sql.functions import col

SOURCE_PATH_CSV         = 'hdfs://namenode:9000/vdata/streamdata/*.csv'
SOURCE_PATH_TXT        = 'hdfs://namenode:9000/vdata/streamdata/*.txt'
CHECKPOINT_LOCATION = 'hdfs://namenode:9000/vdata/checkpoints_raw/'

spark = SparkSession.builder \
  .appName("Streams HDFS -> HIVE") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .config("spark.sql.catalogImplementation", "hive") \
  .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
  .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/hive/warehouse") \
  .enableHiveSupport() \
  .getOrCreate()
  
spark.sql("CREATE DATABASE IF NOT EXISTS idf;")

schema = StructType() \
  .add("JOUR", DateType()) \
  .add("CODE_STIF_TRNS", IntegerType()) \
  .add("CODE_STIF_RES", IntegerType()) \
  .add("CODE_STIF_ARRET", IntegerType()) \
  .add("LIBELLE_ARRET", StringType()) \
  .add("ID_ZDC", IntegerType()) \
  .add("CATEGORIE_TITRE", StringType()) \
  .add("NB_VALD", IntegerType()) \
  
df = spark.readStream \
  .format("csv") \
  .option("header", "true") \
  .schema(schema) \
  .csv(SOURCE_PATH_CSV) \
  
# TODO: cleaning data 

df.writeStream \
  .outputMode("append") \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .toTable("idf.validations") \
  .awaitTermination()
  

  
  

  
  

    
  





    