from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import DoubleType
import time

start = time.time()

#create spark session
spark = SparkSession \
    .builder \
    .appName("NaiveBayes") \
    .getOrCreate()

# read training and testing data
train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
train_data = spark.read.csv(train_datafile, header=False,inferSchema="true")
test_data = spark.read.csv(test_datafile, header=False,inferSchema="true")

# extract features and convert to vectors.
assembler = VectorAssembler(inputCols=train_data.columns[1:],outputCol="features")
assembler_test = VectorAssembler(inputCols=test_data.columns[1:],outputCol="features")

train_vectors = assembler.transform(train_data).select(train_data.columns[0],"features")
test_vectors = assembler_test.transform(test_data).select(test_data.columns[0],"features")


nb = NaiveBayes(smoothing=1.0, modelType="multinomial",labelCol='_c0')

model = nb.fit(train_vectors)

predictions = model.transform(test_vectors)

end = time.time()

time = end - start
parsed_prediction = predictions.select("prediction","_c0")
#toDouble = udf(lambda x: x.DoubleType())
parsed_prediction = parsed_prediction.withColumn("label",parsed_prediction['_c0'].cast(DoubleType())).drop('_c0')
parsed_prediction.show(5)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label",metricName="accuracy")

evaluator_precision= MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label",metricName="weightedPrecision")

evaluator_recall= MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label",metricName="weightedRecall")

evaluator_f1= MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label",metricName="f1")

accuracy = evaluator.evaluate(parsed_prediction)
precision = evaluator_precision.evaluate(parsed_prediction)
recall = evaluator_recall.evaluate(parsed_prediction)
f1 = evaluator_f1.evaluate(parsed_prediction)


rdd = parsed_prediction.rdd

metrics = MulticlassMetrics(rdd)

confusion_matrix= metrics.confusionMatrix()

print(confusion_matrix)

print("Execution time = %s" %time)
print("Accuray = %s" % accuracy)
print("Overall Precision = %s" % precision)
print("Overall Recall = %s" %recall)
print("F1 = %s" %f1)

# use lambda function to calculate performance metrics
labels = rdd.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

rdd.saveAsTextFile("/Users/alfred/PycharmProjects/demo/t29")

