# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from time import time
import argparse
import pyspark as py
import warnings
from pyspark.sql import SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, \
                               NGram, VectorAssembler, CountVectorizer 
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator as BCE

def createContext():
    '''
    Creates SparkContext on all CPUs available. 
    In my case I used 50 CPUs on DiscoveryCluster 
    '''
    try:
        sc = py.SparkContext(appName='logisticRegression')
        sqlContext = SQLContext(sc)
        print("Just created a SparkContext")
    except ValueError:
        warnings.warn("SparkContext already exists in this scope")
    
    df = sqlContext.read.format('com.databricks.spark.csv') \
                   .options(header='true', inferschema='true') \
                   .load('Data/processed_tweets.csv')
    df = df.dropna() # filters data with missing values
    print("Number of tweets in the training set is: ", 
           df.count()) # display the number of tweets
    return df

def split(context):
    '''
        Splits the data into train_set (70% of the data)
        val_set (10% of the data) and 
        test_set (20% of the data)
    '''
    (train_set, val_set, test_set) = context.randomSplit([0.7, 0.1, 0.2], 
                                                         seed=2000)
    return train_set, val_set, test_set

def createPipeline(train_set, val_set, test_set):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    #minDocFreq: remove sparse terms
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) 
    label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
    pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
    pipelineFit = pipeline.fit(train_set)
    train_df = pipelineFit.transform(train_set)
    val_df = pipelineFit.transform(val_set)
    test_df = pipelineFit.transform(test_set)
    return train_df, val_df, test_df

def classify(train_df, val_df, test_df):
    lr = LogisticRegression(maxIter=100)
    lrModel = lr.fit(train_df)
    predictions = lrModel.transform(val_df)
    test_predictions = lrModel.transform(test_df)
    return predictions, test_predictions

def ngramFeatureExtractors(n, inputCol=["text","target"]):
    tokenizer = [Tokenizer(inputCol="text", outputCol="words")]
    ngrams = [
        NGram(n=i, inputCol="words", outputCol="{0}_grams".format(i))
        for i in range(1, n + 1)
    ]

    count_vectorizer = [
        CountVectorizer(vocabSize=5460,inputCol="{0}_grams".format(i),
                        outputCol="{0}_tf".format(i))
        for i in range(1, n + 1)
    ]
    idf = [IDF(inputCol="{0}_tf".format(i), outputCol="{0}_tfidf".format(i), 
           minDocFreq=5) 
           for i in range(1, n + 1)]

    assembler = [VectorAssembler(
        inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)],
        outputCol="features"
    )]
    label_stringIdx = [StringIndexer(inputCol="target", outputCol="label")]
    lr = [LogisticRegression(maxIter=100)]
    return Pipeline(stages=tokenizer + ngrams + count_vectorizer + idf + assembler 
                    + label_stringIdx + lr)

def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Classification', 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--ngrams', type=int, default=3, 
                        help='n-grams for feature extraction')
    parser.add_argument('--classifier', type=int, default=0, 
                        help='use tfidf hashing if 0, otherwise n-grams')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    df = createContext()
    
    print("Dataset (10 samples):\n")
    print(df.show(10))
    train_set, val_set, test_set = split(df)

    if args.classifier == 0:
        train_df, val_df, test_df = createPipeline(train_set, val_set, test_set)
        
        print("Train Data (10 samples)\n")
        print(train_df.show(10))
        predictions, test_predictions = classify(train_df, val_df, test_df)

    else:
        ngram_pipelineFit = ngramFeatureExtractors(args.ngrams).fit(train_set)
        predictions = ngram_pipelineFit.transform(val_set)
        test_predictions = ngram_pipelineFit.transform(test_set)
    
    accuracy = predictions.filter(predictions.label == predictions.prediction)\
                          .count() / float(val_set.count())
    test_accuracy = test_predictions.filter(test_predictions.label == test_predictions.prediction)\
                                    .count() / float(test_set.count())
    
    evaluator = BCE(rawPredictionCol="rawPrediction")
    roc_auc = evaluator.evaluate(predictions)
    test_roc_auc = evaluator.evaluate(test_predictions)
    
    print("Accuracy Score: {0:.4f}".format(accuracy))
    print("ROC-AUC: {0:.4f}".format(roc_auc))
    print("Test Accuracy Score: {0:.4f}".format(test_accuracy))
    print("Test ROC-AUC: {0:.4f}".format(test_roc_auc))


if __name__ == "__main__":
    main()

