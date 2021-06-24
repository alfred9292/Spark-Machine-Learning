from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
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
test_vectors = assembler.transform(test_data).select(train_data.columns[0],"features")

train_vectors.show(2)

pca = PCA(k=2, inputCol="features",outputCol='pca')
pca_test = PCA(k=2, inputCol="features",outputCol='pca')

model = pca.fit(train_vectors)

test_model = pca_test.fit(test_vectors)

pca_result = model.transform(train_vectors).select(train_vectors.columns[0],'pca')

pca_testresult = model.transform(test_vectors).select(test_vectors.columns[0],'pca')

#rdd = pca_result.rdd.map(lambda p:"label: "+str(p._c0)+" feature: "+str(p.pca)).collect()

train_set = pca_result.rdd.map(lambda x:LabeledPoint(x._c0,x.pca)).collect()

#pca_result.show(2)
#pca_testresult.show(2)
count=0
for n in train_set:
    count=count+1
    print(n)
print(count)


classifier = LogisticRegressionWithLBFGS()

model1 = classifier.train()