from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.classification import LogisticRegressionWithLBFGS,LogisticRegressionModel
from pyspark.mllib.evaluation import MulticlassMetrics
import time

start =time.time()

#create spark session
spark = SparkSession \
    .builder \
    .appName("logistic regression") \
    .getOrCreate()

# read training and testing data
train_datafile = "/Users/alfred/PycharmProjects/cc2/data/Train-label-28x28.csv"

test_datafile = "/Users/alfred/PycharmProjects/cc2/data/Test-label-28x28.csv"
train_data = spark.read.csv(train_datafile, header=False,inferSchema="true")
test_data = spark.read.csv(test_datafile, header=False,inferSchema="true")

# extract features and convert to vectors.
assembler = VectorAssembler(inputCols=train_data.columns[1:],outputCol="features")
assembler_test = VectorAssembler(inputCols=test_data.columns[1:],outputCol="features")

train_vectors = assembler.transform(train_data).select(train_data.columns[0],"features")
test_vectors = assembler_test.transform(test_data).select(test_data.columns[0],"features")

train_vectors.show(2)

# use PCA reduce dimension to 75.
pca = PCA(k=75, inputCol="features",outputCol='pca')

model = pca.fit(train_vectors)

pca_result = model.transform(train_vectors).select(train_vectors.columns[0],'pca')

pca_testresult = model.transform(test_vectors).select(test_vectors.columns[0],'pca')

#transform result to rdd.
rdd_train = pca_result.rdd
rdd_test = pca_testresult.rdd

# function to convert rdd to labelled point to fit the required format for logistic regression with LBFGS.
def convert(y):
    d = [x for x in y]
    return LabeledPoint(d[0],d[1:])

# apply convert function to all features stored in rdd by using lambda function. 
parsed_traindata = rdd_train.map(lambda x:convert(x))

parsed_testdata = rdd_test.map(lambda x:convert(x))


print(parsed_traindata.take(2))
#pca_result.show(2)
#pca_testresult.show(2)

classifier = LogisticRegressionWithLBFGS()

model1 = classifier.train(parsed_traindata,numClasses=10)

# Using lambda function to make prediction. 
prediciton = parsed_testdata.map(lambda p:(float(model1.predict(p.features)),p.label))

end = time.time()
time = end-start
print("Execution time = %s" % time)
p=prediciton.take(20)
print(p)

# Using lambda function to calculate accurancy. 

accuracy = prediciton.filter(lambda lp:lp[0] ==lp[1]).count() /float(parsed_testdata.count())

metrics = MulticlassMetrics(prediciton)

# performance metrics calculation.
confusion_matrix= metrics.confusionMatrix()
precision = metrics.precision()
recall = metrics.recall()
f1 = metrics.fMeasure()

print("Execution time = %s" %time)
print("Accuray = %s" % accuracy)
print("Overall Precision = %s" % precision)
print("Overall Recall = %s" %recall)
print("F1 = %s" %f1)

print(confusion_matrix)

labels = parsed_traindata.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))


print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

prediciton.saveAsTextFile("/Users/alfred/PycharmProjects/demo/t28")
