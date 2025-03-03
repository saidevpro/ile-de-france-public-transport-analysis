spark-submit /app/jobs/stream_data_validations_sources.py
spark-submit /app/jobs/stream_hdfs_validations_hive.py
spark-submit --jars /app/bin/mysql-connector-j-9.2.0.jar /app/jobs/stream_validations_data_to_mysql.py

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 /app/jobs/stream_bulk_messages_ile_de_france.py
spark-submit /app/jobs/stream_messages_to_hive.py
spark-submit --jars /app/bin/postgresql-42.7.5.jar /app/jobs/stream_messages_to_postgres.py


