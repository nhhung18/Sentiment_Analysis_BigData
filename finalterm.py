from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, regexp_replace
from pyspark.sql.functions import lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF
)

spark = SparkSession.builder \
    .appName("Sentiment Analysis - Preprocessing") \
    .getOrCreate()

df = spark.read.csv(
    "sentiment140.csv",
    header=False,
    inferSchema=True
)

# Doi ten cot, _c0 -> label
df = df.select(
    col("_c0").alias("target"),
    col("_c1").alias("ids"),
    col("_c2").alias("date"),
    col("_c3").alias("flag"),
    col("_c4").alias("user"),
    col("_c5").alias("text")
)

# Tao cot label 
df = df.withColumn(
    "label",
    when(col("target") == 0, 0)
    .when(col("target") == 2, 1)
    .when(col("target") == 4, 2)
)

# Chuyen chu thuong 
df = df.withColumn("text", lower(df["text"]))

# Xoa URL, mention, hashtag
df = df.withColumn(
    "text",
    regexp_replace(col("text"), r"http\S+|www\S+|@\w+|#\w+", "")
)

# Xoa ky tu dac biet
df = df.withColumn(
    "text",
    regexp_replace(col("text"), r"[^a-z\s]", "")
)

# Tach tu, loai bo stopword,
tokenizer = Tokenizer(
    inputCol="text",
    outputCol="tokens"
)
df = tokenizer.transform(df)

remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens"
)
df = remover.transform(df)

# Vector hoa TF-IDF => Quan trong giup mo hinh danh gia cam xuc
tf = HashingTF(
    inputCol="filtered_tokens",
    outputCol="rawFeatures",
    numFeatures=20000
)
df = tf.transform(df)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)
idf_model = idf.fit(df)
df = idf_model.transform(df)

df.select("text", "label", "features").show(5, truncate=False)

df.show(5)