# ============================================================
# Naive Bayes Sentiment Analysis (Final Version)
# - Spark ML Pipeline
# - TF-IDF (HashingTF + IDF)
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
    IDF
)
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import json
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    # 1. SparkSession
    spark = SparkSession.builder \
        .appName("NaiveBayesSentiment") \
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
    # 4. FEATURE ENGINEERING PIPELINE
    # --------------------------------------------------------
    tokenizer = Tokenizer(
        inputCol="text",
        outputCol="words"
    )

    stopwords = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )

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

    nb = NaiveBayes(
        featuresCol="features",
        labelCol="label",
        modelType="multinomial",
        smoothing=1.0
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        stopwords,
        hashingTF,
        idf,
        nb
    ])

    # --------------------------------------------------------
    # 5. Train
    # --------------------------------------------------------
    print("\nTraining Naive Bayes model...")
    model = pipeline.fit(train_df)

    # --------------------------------------------------------
    # 6. Prediction
    # --------------------------------------------------------
    test_pred = model.transform(test_df)

    # --------------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------------
    acc_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
    )

    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    roc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    # ---- Metrics
    accuracy = acc_eval.evaluate(test_pred)
    f1 = f1_eval.evaluate(test_pred)
    roc_auc = roc_eval.evaluate(test_pred)

    print("\n===== TEST RESULTS (Naive Bayes) =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # ---- Confusion Matrix
    cm_pdf = test_pred.select("label", "prediction").toPandas()
    cm = confusion_matrix(cm_pdf["label"], cm_pdf["prediction"])

    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }

    with open(f"{metrics_dir}/nb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)



    # --------------------------------------------------------
    # 8. VISUALIZATION
    # --------------------------------------------------------
    save_dir = "naive_bayes_statistical_img"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Label distribution
    # label_pdf = df.groupBy("label").count().toPandas()

    # plt.figure(figsize=(5, 4))
    # plt.bar(label_pdf["label"], label_pdf["count"])
    # plt.xlabel("Label (0 = Negative, 1 = Positive)")
    # plt.ylabel("Count")
    # plt.title("Label Distribution")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "label_distribution.png"), dpi=300)
    # plt.close()

    # -------- Confusion Matrix
    cm_pdf = test_pred.select("label", "prediction").toPandas()
    cm = confusion_matrix(cm_pdf["label"], cm_pdf["prediction"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
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
    plt.title("ROC Curve - Naive Bayes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 9. Save model (LOCAL)
    # --------------------------------------------------------
    model.write().overwrite().save("model/naive_bayes")
    print("\nModel saved to local: model/naive_bayes")

    spark.stop()

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
