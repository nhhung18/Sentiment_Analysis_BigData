# ============================================================
# Logistic Regression Sentiment Analysis (Final Version)
# - Spark ML Pipeline
# - TF-IDF or N-Gram
# - Train / Test
# - Accuracy + ROC-AUC
# - Visualization: Label dist, Confusion Matrix, ROC Curve
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    NGram,
    CountVectorizer,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
from pyspark.sql.functions import col

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Sentiment Analysis (Spark ML)"
    )
    parser.add_argument(
        "--use_ngram",
        type=int,
        default=0,
        help="0 = HashingTF + TF-IDF | 1 = N-Gram + CountVectorizer"
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=2,
        help="Max N for N-Gram (only used if use_ngram=1)"
    )
    return parser.parse_args()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    # 1. SparkSession
    spark = SparkSession.builder \
        .appName("LogisticRegressionSentiment") \
        .getOrCreate()

    # --------------------------------------------------------
    # 2. Load cleaned data
    # --------------------------------------------------------
    df = spark.read.csv(
        "hdfs://localhost:9000/sentiment/clean/",
        header=True,
        inferSchema=True
    ).dropna()

    print(f"Total samples: {df.count()}")
    df.show(5, truncate=False)

    # --------------------------------------------------------
    # 3. Train / Test split
    # --------------------------------------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # --------------------------------------------------------
    # 4. Common NLP stages
    # --------------------------------------------------------
    tokenizer = Tokenizer(
        inputCol="text",
        outputCol="words"
    )

    stopwords = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )

    stages = [tokenizer, stopwords]

    # --------------------------------------------------------
    # 5. Feature Engineering
    # --------------------------------------------------------
    if args.use_ngram == 0:
        print(">>> Using HashingTF + TF-IDF")

        hashingTF = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=2**18
        )

        idf = IDF(
            inputCol="raw_features",
            outputCol="features",
            minDocFreq=5
        )

        stages += [hashingTF, idf]

    else:
        print(f">>> Using N-Gram (1 → {args.ngram}) + CountVectorizer")

        ngram_cols = []
        for n in range(1, args.ngram + 1):
            ngram = NGram(
                n=n,
                inputCol="filtered_words",
                outputCol=f"{n}gram"
            )

            cv = CountVectorizer(
                inputCol=f"{n}gram",
                outputCol=f"{n}gram_tf",
                vocabSize=10000,
                minDF=5
            )

            idf = IDF(
                inputCol=f"{n}gram_tf",
                outputCol=f"{n}gram_tfidf"
            )

            stages.extend([ngram, cv, idf])
            ngram_cols.append(f"{n}gram_tfidf")

        assembler = VectorAssembler(
            inputCols=ngram_cols,
            outputCol="features"
        )
        stages.append(assembler)

    # --------------------------------------------------------
    # 6. Logistic Regression
    # --------------------------------------------------------
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        regParam=0.01,
        elasticNetParam=0.0
    )

    stages.append(lr)

    # --------------------------------------------------------
    # 7. Pipeline
    # --------------------------------------------------------
    pipeline = Pipeline(stages=stages)

    print("\nTraining model...")
    model = pipeline.fit(train_df)

    # --------------------------------------------------------
    # 8. Prediction
    # --------------------------------------------------------
    test_pred = model.transform(test_df)

    # --------------------------------------------------------
    # 9. Evaluation
    # --------------------------------------------------------
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    auc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    accuracy = acc_eval.evaluate(test_pred)
    roc_auc = auc_eval.evaluate(test_pred)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # --------------------------------------------------------
    # 10. VISUALIZATION
    # --------------------------------------------------------
    print("\n===== VISUALIZATION =====")

    save_dir = "logistic_statistical_img"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Label distribution
    label_pdf = df.groupBy("label").count().toPandas()

    plt.figure(figsize=(5, 4))
    plt.bar(label_pdf["label"], label_pdf["count"])
    plt.xlabel("Label (0 = Negative, 1 = Positive)")
    plt.ylabel("Count")
    plt.title("Label Distribution")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "label_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -------- Confusion Matrix
    cm_pdf = test_pred.select("label", "prediction").toPandas()
    cm = confusion_matrix(cm_pdf["label"], cm_pdf["prediction"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -------- ROC Curve
    roc_pdf = test_pred.select("label", "probability").toPandas()
    roc_pdf["prob"] = roc_pdf["probability"].apply(lambda x: float(x[1]))

    fpr, tpr, _ = roc_curve(roc_pdf["label"], roc_pdf["prob"])
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # 11. Save model (LOCAL – phù hợp user hung)
    # --------------------------------------------------------
    model.write().overwrite().save("model/logistic_regression")
    print("\nModel saved to local: model/logistic_regression")

    spark.stop()

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
