import findspark
findspark.init()
findspark.find()

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import DoubleType
from scipy.stats  import norm, ttest_ind
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


import mlflow
import mlflow.sklearn

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.0.20:8000"

os.environ["AWS_ACCESS_KEY_ID"] = "YCAJEhBjCXUDNFDZkq-0dKZS3"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YCMrJJyq51Or0ZHcrzhL3upptA0t5geHbJS3frHh"

TARGET_LABEL = 'loss'

mlflow.set_experiment("Default_prediction_experiment")
mlflow.start_run(description="Inference and best model")

spark = SparkSession.builder\
    .master('local[*]')\
    .appName('CatBoostWithSpark')\
    .config("spark.jars.packages", "ai.catboost:catboost-spark_3.0_2.12:1.2.2")\
    .getOrCreate()
    
import catboost_spark

log4jLogger = spark._jvm.org.apache.log4j   
LOGGER = log4jLogger.LogManager.getLogger(__name__)

# 1. ИНФЕРЕНС НА ТЕСТЕ

LOGGER.info("--------------------------------------------------- Load current model")
path_to_current_model = "s3a://test-bucket-mlops/artifacts/classifier_base"
loaded_model = catboost_spark.CatBoostClassificationModel.load(path_to_current_model)


LOGGER.info("--------------------------------------------------- Reading test_sample")
path_test_sample = "s3a://test-bucket-mlops/test_sample/"
df_test_sample = spark.read.option("header", True).option("inferSchema", "true").csv(path_test_sample)
TARGET_LABEL = 'loss'

LOGGER.info("--------------------------------------------------- Predict on test_sample")
features = ['f674', 'f2', 'f67', 'f471', 'f766', 'f670', 'f596', 'f332', 'f464']
for feature in features:
    df_test_sample_clean = df_test_sample.fillna({feature: 0.})  
assembler = VectorAssembler(inputCols=features, outputCol='features')

def prepare_vector(df):    
    result_df = assembler.transform(df)
    return result_df

test_sample = prepare_vector(df_test_sample_clean)
predict_test_sample = loaded_model.transform(test_sample)
evaluator = BinaryClassificationEvaluator(
                                        labelCol=TARGET_LABEL,
                                        rawPredictionCol="probability", 
                                        metricName="areaUnderROC")
ROC_AUC_test_sample = evaluator.evaluate(predict_test_sample)
  
LOGGER.info("----------------- Test_sample roc_auc -------------")    
LOGGER.info(ROC_AUC_test_sample)
LOGGER.info("---------------------------------------------------") 

# 2. СОХРАНЕНИЕ РЕЗУЛЬТАТА ИНФЕРЕНСА

LOGGER.info("--------------------------------------------------- Inference to s3")
path_test_infer = "s3a://test-bucket-mlops/test_sample_inference/"
predict_test_sample.select(features + ['prediction']).write.mode("overwrite").option('header','true').csv(path_test_infer)

mlflow.log_metric("test_sample ROC_AUC", ROC_AUC_test_sample) 

class CurveMetrics(BinaryClassificationMetrics):
        def __init__(self, *args):
            super(CurveMetrics, self).__init__(*args)

        def _to_list(self, rdd):
            points = []           
            for row in rdd.collect():               
                points += [(float(row._1()), float(row._2()))]
            return points

        def get_curve(self, method):
            rdd = getattr(self._java_model, method)().toJavaRDD()
            return self._to_list(rdd)

# ROC_curve test_sample
preds_test_sample = predict_test_sample.select(TARGET_LABEL,'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row[TARGET_LABEL])))
points = CurveMetrics(preds_test_sample).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("AUC for test_sample")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.plot(x_val, y_val)

plt.savefig("ROC_test_sample.png")
mlflow.log_artifact("ROC_test_sample.png")



# 3. РАЗДЕЛЕНИЕ ТЕСТОВОЙ ВЫБОРКИ НА ЧАСТЬ ДЛЯ ДООБУЧЕНИЯ И ЧАСТЬ ДЛЯ ВАЛИДАЦИИ

df_train, df_val = df_test_sample_clean.randomSplit([0.7, 0.3])

# 4. ДООБУЧЕНИЕ - получение новой модели

# читаем старый трейн
LOGGER.info("--------------------------------------------------- Reading train")

path_train = "s3a://test-bucket-mlops/train/train.csv"
train_old = spark.read.option("header", True).option("inferSchema", "true").csv(path_train)

# соединяем старый трейн и первую часть теста
train_sup = train_old.union(df_train)

features = ['f674', 'f2', 'f67', 'f471', 'f766', 'f670', 'f596', 'f332', 'f464']
for feature in features:
    train_sup = train_sup.fillna({feature: 0.})
    

LOGGER.info("--------------------------------------------------- Vectorizing")
assembler = VectorAssembler(inputCols=features, outputCol='features')

def prepare_vector(df):    
    result_df = assembler.transform(df)
    return result_df

train = prepare_vector(train_sup)
test = prepare_vector(df_val)

LOGGER.info("--------------------------------------------------- Fit new model")

train_pool = catboost_spark.Pool(train.select(['features', TARGET_LABEL]))
train_pool.setLabelCol(TARGET_LABEL)
train_pool.setFeaturesCol('features')

classifier = catboost_spark.CatBoostClassifier(featuresCol='features', labelCol=TARGET_LABEL)
model = classifier.fit(train_pool)

# 5. ПРЕДИКТ на валидации обеих моделей
LOGGER.info("--------------------------------------------------- Predict both models")

predictions_old = loaded_model.transform(test)
predictions_new = model.transform(test)

# ROC curve on val model_old
preds_test_old = predictions_old.select(TARGET_LABEL,'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row[TARGET_LABEL])))
points = CurveMetrics(preds_test_old).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("AUC for validation old model")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.plot(x_val, y_val)

plt.savefig("ROC_val_old_model.png")
mlflow.log_artifact("ROC_val_old_model.png")

# ROC curve on val model_old
preds_test_new = predictions_new.select(TARGET_LABEL,'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row[TARGET_LABEL])))
points = CurveMetrics(preds_test_new).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("AUC for validation new model")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.plot(x_val, y_val)

plt.savefig("ROC_val_new_model.png")
mlflow.log_artifact("ROC_val_new_model.png")

# 6. Метрики обеих моделей
LOGGER.info("--------------------------------------------------- Evaluate both models")
ROC_AUC_old = evaluator.evaluate(predictions_old)
ROC_AUC_new = evaluator.evaluate(predictions_new)

LOGGER.info(ROC_AUC_old)
LOGGER.info(ROC_AUC_new)
LOGGER.info("------------------------------------------------------------------------")

mlflow.log_metric("val_old_model ROC_AUC", ROC_AUC_old) 
mlflow.log_metric("val_new_model ROC_AUC", ROC_AUC_new) 

# 7. Сравнение моделей. А/В тест

LOGGER.info("--------------------------------------------------- Bootstrap")
scores_old = []
scores_new = []

bootstrap_iterations = 100

for i in range(bootstrap_iterations):
        LOGGER.info(i)
        LOGGER.info("-------------")
        sample1 = predictions_old.sample(fraction=1.0, withReplacement=True)
        scores_old.append(evaluator.evaluate(sample1))
        sample2 = predictions_new.sample(fraction=1.0, withReplacement=True)
        scores_new.append(evaluator.evaluate(sample2))


# Log bootstrap hists
plt.figure()
plt.hist(scores_old)
plt.savefig("Hist_old.png")
mlflow.log_artifact("Hist_old.png")

plt.figure()
plt.hist(scores_new)
plt.savefig("Hist_new.png")
mlflow.log_artifact("Hist_new.png")


alpha = 0.05   
pvalue = ttest_ind(scores_old, scores_new).pvalue
best_model_changed = 0
ROC_AUC_best = 0

mlflow.log_param('alpha', alpha)
mlflow.log_param('pvalue', pvalue)

LOGGER.info("------------- pvalue")
LOGGER.info(pvalue)

if pvalue < alpha:
    LOGGER.info("Reject null hypothesis.")
    if ROC_AUC_new >= ROC_AUC_old:
        LOGGER.info("Change model")        
        best_model_changed = 1
        ROC_AUC_best = ROC_AUC_new
        # 8. Логгирование лучшей модели
        LOGGER.info("---------------------------------------------------Save best model...")
        path_to_best_model = "s3a://test-bucket-mlops/artifacts/classifier_best"
        model.write().overwrite().save(path_to_best_model)       

    else:
        ROC_AUC_best = ROC_AUC_old
else:
    ROC_AUC_best = ROC_AUC_old
    LOGGER.info("Accept null hypothesis.")  


mlflow.log_param('ROC_AUC_best', ROC_AUC_best)
mlflow.log_param('best_model_changed flag', best_model_changed)


# 9. СОХРАНЕНИЕ test_sample + train

LOGGER.info("--------------------------------------------------- TRAIN + TEST_SAMPLE")
train_new = train_old.union(df_test_sample)

LOGGER.info("---------------------------------------------------REWrite train_new to s3")

path_train_sup = "s3a://test-bucket-mlops/train_sup/"
train_new.write.mode("overwrite").option('header','true').csv(path_train_sup)


# 10. ГРАНИЦА метрики
LOGGER.info("--------------------------------------------------- Load auc_min_border")

path_to_border = "s3a://test-bucket-mlops/artifacts/border.txt"
border = spark.read.text(path_to_border, wholetext=True)
LOGGER.info("--------------------------------------------------- border")
border = border.withColumn("value",border["value"].cast(DoubleType())).collect()[0].__getitem__('value')
LOGGER.info(border)

if ROC_AUC_best < border:
    LOGGER.info("!!!!!!!!!!!!!--- ALLERT ---!!!!!!!!!!!!!!!")
else:
    LOGGER.info("!!!!!!!!!!!!!--- GOOD ---!!!!!!!!!!!!!!!")
    
spark.stop()
        
    