from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.tree import DecisionTree,DecisionTreeModel
from pyspark.mllib.evaluation import MulticlassMetrics
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

rdd_train = pca_result.rdd
rdd_test = pca_testresult.rdd

def convert(y):
    d = [x for x in y]
    return LabeledPoint(d[0],d[1:])

parsed_traindata = rdd_train.map(lambda x:convert(x))

parsed_testdata = rdd_test.map(lambda x:convert(x))


print(parsed_traindata.take(2))

classifier = DecisionTree.trainClassifier(parsed_testdata,numClasses=10,categoricalFeaturesInfo={},maxDepth=5,maxBins=32)

prediction = classifier.predict(parsed_testdata.map(lambda x:x.features))

test_result = parsed_testdata.map(lambda d:d.label).zip(prediction)

p=test_result.take(20)

print(p)