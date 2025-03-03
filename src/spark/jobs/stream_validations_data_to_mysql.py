from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, avg as _avg, dayofmonth, col, when, dayofweek
from os.path import abspath

SOURCE_DATA = 'hdfs://namenode:9000/hive/warehouse/idf.db/validations'
CHECKPOINT_LOCATION = 'hdfs://namenode:9000/vdata/checkpoints_mysql_raw/'
mysql_url = "jdbc:mysql://mariadb:3306/kpi"
mysql_properties = {
  "user": "root",
  "password": "mariadb",
  "driver": "com.mysql.cj.jdbc.Driver"
}
mysql_connector_jars = 'file:///app/bin/mysql-connector-j-9.2.0.jar'

spark = SparkSession.builder \
  .appName("Streams Validations Data to MYSQL") \
  .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
  .config("spark.sql.catalogImplementation", "hive") \
  .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
  .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/hive/warehouse") \
  .config("spark.sql.streaming.schemaInference", "true") \
  .config("spark.driver.extraClassPath", mysql_connector_jars) \
  .config("spark.executor.extraClassPath", mysql_connector_jars) \
  .enableHiveSupport() \
  .getOrCreate()

schema = spark.table("idf.validations").schema

df = spark.readStream \
  .schema(schema) \
  .format("parquet") \
  .load(SOURCE_DATA)
 
def create_metrics_data(df, epoch_id):
  tf = spark.sql("SELECT * FROM idf.validations")
  tf = tf.withColumn(
    "JOUR_DE_LA_SEMAINE",
    when(dayofweek(col("JOUR")) == 1, "Sunday")
    .when(dayofweek(col("JOUR")) == 2, "Monday")
    .when(dayofweek(col("JOUR")) == 3, "Tuesday")
    .when(dayofweek(col("JOUR")) == 4, "Wednesday")
    .when(dayofweek(col("JOUR")) == 5, "Thursday")
    .when(dayofweek(col("JOUR")) == 6, "Friday")
    .when(dayofweek(col("JOUR")) == 7, "Saturday")
  )

  tt = tf.groupBy("CATEGORIE_TITRE", "JOUR_DE_LA_SEMAINE") \
    .agg(_sum("NB_VALD").alias("TOTAL_VALIDATION"), _avg("NB_VALD").alias("MOYEN_VALIDATION")) \
    .sort(col("TOTAL_VALIDATION").desc()) \
    .write \
    .jdbc(url=mysql_url, table="titre_transport", mode="overwrite", properties=mysql_properties)


  tf.groupBy("LIBELLE_ARRET", "JOUR_DE_LA_SEMAINE") \
    .agg(_sum("NB_VALD").alias("TOTAL_VALIDATION"), _avg("NB_VALD").alias("MOYEN_VALIDATION")) \
    .sort(col("TOTAL_VALIDATION").desc()) \
    .write \
    .jdbc(url=mysql_url, table="arret", mode="overwrite", properties=mysql_properties)

query = df \
  .writeStream \
  .foreachBatch(create_metrics_data) \
  .option("checkpointLocation", CHECKPOINT_LOCATION) \
  .start()
  
query.awaitTermination()

  
  
  
