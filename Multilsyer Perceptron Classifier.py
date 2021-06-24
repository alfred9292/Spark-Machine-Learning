from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import datetime

start = datetime.datetime.now()

spark = SparkSession \
    .builder \
    .appName("logistic regression") \
    .getOrCreate()

train_datafile = "/Users/alfred/PycharmProjects/cc2/data/Train-label-28x28.csv"

test_datafile = "/Users/alfred/PycharmProjects/cc2/data/Test-label-28x28.csv"
train_data = spark.read.csv(train_datafile, header=False,inferSchema="true")
test_data = spark.read.csv(test_datafile, header=False,inferSchema="true")

assembler = VectorAssembler(inputCols=train_data.columns[1:],outputCol="features")
assembler_test = VectorAssembler(inputCols=test_data.columns[1:],outputCol="features")

train_vectors = assembler.transform(train_data).select(train_data.columns[0],"features")
test_vectors = assembler_test.transform(test_data).select(test_data.columns[0],"features")

train_vectors.show(2)

pca = PCA(k=75, inputCol="features",outputCol='pca')

model = pca.fit(train_vectors)

pca_result = model.transform(train_vectors).select(train_vectors.columns[0],'pca')

pca_testresult = model.transform(test_vectors).select(test_vectors.columns[0],'pca')

layer =[4,5,4,5]
#pca_result.show(2)
#pca_testresult.show(2)

classifier = MultilayerPerceptronClassifier(maxIter=100,layers=layer,blockSize=128,seed=1234,featuresCol='pca',labelCol='_c0', predictionCol='prediction')

model1 = classifier.fit(pca_result)
result = model1.transform(pca_testresult)
prediciton = result.select("_c0","prediction")

prediciton.show(n=20)
#evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
#p=result.show(20)
#print(p)

#print("accuracy:+ "+str(evaluator.evaluate(prediciton)))