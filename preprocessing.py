# ============================================================
# Tiền xử lý văn bản Tweet với Spark + HDFS + Trực quan hóa
# - Đọc dữ liệu CSV từ HDFS
# - Làm sạch văn bản tweet (xóa URL, mention, ký tự đặc biệt)
# - Trực quan hóa dữ liệu trước và sau khi làm sạch
# - Lưu dữ liệu đã làm sạch vào HDFS định dạng CSV
# ============================================================
import re
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import udf, col, length, when, col, lower, regexp_replace, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# 1. Tạo SparkSession
spark = SparkSession.builder \
    .appName("Twitter_Text_Preprocessing") \
    .getOrCreate()

# 2. Định nghĩa schema cho dataset Sentiment140
schema = """
    sentiment INT,
    id STRING,
    date STRING,
    query STRING,
    user STRING,
    text STRING
"""

# 3. Đọc file CSV từ HDFS
df = spark.read.csv(
    "hdfs://localhost:9000/sentiment/raw/",
    schema=schema,
    header=False
)

# Xóa các dòng có giá trị null
df = df.dropna()

# 4. TRỰC QUAN HÓA - Trước khi làm sạch
print("\n ===== BEFORE CLEAN =====")

# Phân bố sentiment
sentiment_counts = df.groupBy("sentiment").count().toPandas()
print("\n Sentiment distribution:")
print(sentiment_counts)

# Tính độ dài văn bản trước khi làm sạch
df_with_len = df.withColumn("text_length", length(col("text")))

# Lấy mẫu dữ liệu để trực quan hóa (10000 dòng để tránh quá tải bộ nhớ)
sample_df = df_with_len.sample(False, 0.01, seed=42).toPandas()

# Hiển thị ví dụ
print("\n Tweet ex before clean:")
print(sample_df.head(5).to_string(index=False))

# Tạo figure với nhiều biểu đồ con
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phân Tích Dữ Liệu Twitter', fontsize=16, fontweight='bold')

# Biểu đồ 1: Phân bố Sentiment (Biểu đồ cột)
axes[0, 0].bar(sentiment_counts['sentiment'].astype(str), 
               sentiment_counts['count'], 
               color=['red', 'green'])
axes[0, 0].set_xlabel('Sentiment (0= Negative, 4= Positive)')
axes[0, 0].set_ylabel('Quantity')
axes[0, 0].set_title('Sentiment distribution')
axes[0, 0].grid(axis='y', alpha=0.3)

# Biểu đồ 2: Phân bố độ dài văn bản - Boxplot
axes[0, 1].boxplot(sample_df['text_length'])
axes[0, 1].set_ylabel('Text length (characters)')
axes[0, 1].set_title('Text length distribution (Before clean)')
axes[0, 1].grid(axis='y', alpha=0.3)

# Biểu đồ 3: Histogram độ dài văn bản
axes[1, 0].hist(sample_df['text_length'], bins=50, color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Text length (characters)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram text length (Before clean)')
axes[1, 0].axvline(140, color='red', linestyle='--', label='Maximun 140 characters')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Lưu biểu đồ trước khi làm sạch
plt.tight_layout()
plt.savefig("before_cleaning.png", dpi=300, bbox_inches='tight')
print("\n Saved: before_cleaning.png")

# 5. Định nghĩa hàm làm sạch tweet
# Chuẩn hóa nhãn (0/4 -> 0/1)
df = df.withColumn(
    "label",
    when(col("sentiment") == 4, 1).otherwise(0)
)

# Xóa các ký tự thừa
clean_df = df.select("label", "text") \
    .withColumn("text", lower(col("text"))) \
    .withColumn("text", regexp_replace(col("text"), r'@[A-Za-z0-9_]+', '')) \
    .withColumn("text", regexp_replace(col("text"), r'https?://\S+', '')) \
    .withColumn("text", regexp_replace(col("text"), r'#', '')) \
    .withColumn("text", regexp_replace(col("text"), r'[^a-zA-Z\s]', ' ')) \
    .withColumn("text", regexp_replace(col("text"), r'\s+', ' ')) \
    .withColumn("text", trim(col("text")))

clean_df = clean_df.filter(length(col("text")) > 0)

# Thêm độ dài văn bản sau khi làm sạch
clean_df_with_len = clean_df.withColumn("text_length", length(col("text")))


# 7. TRỰC QUAN HÓA - Sau khi làm sạch
print("\n ===== AFTER CLEAN =====")

# Lấy mẫu dữ liệu đã làm sạch
sample_clean_df = clean_df_with_len.sample(False, 0.01, seed=42).toPandas()

# Hiển thị ví dụ
print("\n Tweet ex after clean:")
print(sample_clean_df.head(5).to_string(index=False))

# Tạo biểu đồ so sánh
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Comparing Text Length: Before vs. After Cleaning', 
              fontsize=16, fontweight='bold')

# So sánh Boxplot
axes2[0].boxplot([sample_df['text_length'], sample_clean_df['text_length']], 
                 labels=['Before', 'After'])
axes2[0].set_ylabel('Text length (characters)')
axes2[0].set_title('Comparing Boxplot')
axes2[0].grid(axis='y', alpha=0.3)

# So sánh Histogram
axes2[1].hist(sample_df['text_length'], bins=50, alpha=0.5, 
              label='Before', color='red', edgecolor='black')
axes2[1].hist(sample_clean_df['text_length'], bins=50, alpha=0.5, 
              label='After', color='green', edgecolor='black')
axes2[1].set_xlabel('Text length (Characters)')
axes2[1].set_ylabel('Frequency')
axes2[1].set_title('Comparing Histogram')
axes2[1].legend()
axes2[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("after_cleaning.png", dpi=300, bbox_inches='tight')
print("\n Saved: after_cleaning.png")

# In thống kê
print("\n[Statistics]")
print(f"\n Before clean:")
print(f"  - Avg length: {sample_df['text_length'].mean():.2f} characters")
print(f"  - Maximum length: {sample_df['text_length'].max()} characters")
print(f"  - Minimum length: {sample_df['text_length'].min()} characters")

print(f"\n After clean:")
print(f"  - Avg length: {sample_clean_df['text_length'].mean():.2f} characters")
print(f"  - Maximum length: {sample_clean_df['text_length'].max()} characters")
print(f"  - Minimum length: {sample_clean_df['text_length'].min()} characters")

# 8. Lưu dữ liệu đã làm sạch

# Lưu vào file CSV local (file đơn)
clean_df.coalesce(4).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("clean/")

print("Saved to  local: clean/")

# 9. Dừng SparkSession
spark.stop()