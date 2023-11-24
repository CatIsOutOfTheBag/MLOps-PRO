import findspark
findspark.init()
findspark.find()

import pyspark
from pyspark.sql.functions import *

app_name = "Olga"
spark_ui_port = 4048

spark = (
        pyspark.sql.SparkSession.builder
        .appName(app_name)
        .config("spark.ui.port", spark_ui_port)
        .getOrCreate()
    )

log4jLogger = spark._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)

LOGGER.info("---------------------------------------------------Reading...")

path = "s3a://test-bucket-mlops/test/test.csv"
df = spark.read.option("header", True).option("inferSchema", "true").csv(path)

LOGGER.info("---------------------------------------------------Sampling 10%...")

df_sample = df.sample(0.1)

LOGGER.info("---------------------------------------------------Write csv to s3...")

df_sample.write.mode("overwrite").option('header','true').csv("s3a://test-bucket-mlops/test_sample/")

spark.stop()