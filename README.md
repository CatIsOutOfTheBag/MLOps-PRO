# DEFAULT PREDICTION MLOPS PIPELINE
## _MLOps - проект по прогнозированию дефолта_

Прогнозирование дефолта - классическая задача, решаемая Data Scientist-ами повсеместно в финтех секторе.
Вероятность дефолта - одна из ключевых составляющих кредитного риска, в свою очередь влияющего на управление Банком своим портфелем и резервами.

Построение процесса скоринга и принятия на его основе решения представляет из себя интересную и многогранную задачу с возможностью использовать широкий спектр инструментов автоматизации.

Одним из вариантов построения архитектуры процесса является использование облачных технологий. Облачная инфраструктура для финтех решения должна покрываться более строгими контурами защиты информации, чем аналогичное on-premise решение, но вопросы безопасности выходят за рамки данной работы.

Далее пошагово описывается структура разработанного решения с методами использования различных технологий и применяемых инструментов.

## Данные


<image src="screens/1.jpg" alt="data">

Получение реальных данных о договорах и заемщиках вне банковских контуров - задача непосильная, а иногда и нелегльная :) Так, для разработки модели был выбран датасет деперсонализированных данных с сайта конкурсов по машинному обучению https://www.kaggle.com/competitions/loan-default-prediction/data

Это набор данных без указания наименования признаков, насчитывающий около 800 параметров и таргет 'loss' - указывающий одновременно и на факт дефолта, и на величину потерь в случае его реализации. Для задачи данного проекта таргет был преобразован в бинарную величину (1-был факт дефолта, 0-не было). Данные датасета соответствуют реальному набору финансовых операций, но анонимизированы, стандартизировны и лишены трендов.

В наборе данных присутствуют как дискретные, так и непрерывные и категориальные признаки. Объем датасета порядка 500 Мб.

## Поступление новых данных для инференса
В проекте выбран подход к искусственной генерации данных для инференса. От исходного train датасета была отделена часть на test, из которого, в последствии, сэмплируются "новые" данные, игнорируя наличие разметки.

Примечание: здесь можно интересным образом эмулировать процесс data leak и наблюдать устойчивый рост метрик - просто сэмплируя данные для инференса с возвращением, что позволит модели зафиксировать некоторые наблюдения дважды. Также случайное сэмплирование позволяет модели "заглядывать в будущее", если распределение данных схоже по поведению с временным рядом.

Процесс сэмплирования данных реализован в скрипте data_gen.py в папке data_gen

## Baseline python

Для построения baseline модели использовался случайно семплированная из исходного датасета выборка данных размером в 100 000 записей. В качестве метода машинного обучения выбрана библиотека от Яндекса - Catboost.

Пайплайн решения включает в себя небольшой EDA, предобработку признаков - заполнение пропусков в соответствии с типами, построение нисходящего списка важности переменных, а так же отбор наиболее информативных из них. После - оценку базового решения на кросс-валидации и forward feature selection на подмножесте отобранных признаков. Этап тюнинга гиперпараметров пропущен намеренно, в связи с тем, что классификатор хорошо себя показывает "из коробки" на дефолтных настройках.

Процесс моделирования можно увидеть в ноутбуке model.ipynb в папке model.

## Baseline PySpark
Особым челленджем проекта стало использование API для Spark - PySpark, в связи с желанием сделать решение масштабируемым на большие объемы данных, а вычисления - распределенными.
Разработка модели на PySpark представлена в файле spark_model.ipynb в папке spark

## Хранение данных
В батчовых моделях финтеха заказчик всегда знает, во сколько и где должен забирать выходы из продакшн-решений. Обычно это внутренние хранилища, но в рамках проекта для получения результатов инференса, логгирования артефактов и хранения наборов данных для обучения/дообучения используется s3 - объектное хранилище, разработанное компанией AWS и доступное в облачном сервисе Яндекса.

В хранилище представлены следующие папки:
1. train - исходный датасет для обучения модели
2. test - отложенная выборка для эмуляции потока новых данных
3. test_sample - сэплированная выборка из test размером 10%, она же "новые данные"
4. train_sup - обогащенная обучающая выборка для дообучения модели
5. test_sample_inference - выходные данные инференса - точка, где можно забирать скор для бизнеса
6. artifacts - папка хранения артефактов: гистограмм, графиков ROC_Curve, сериализованных "лучших" по результатам дообучения моделей, а так же различных иных метрик и величин.


<image src="screens/3. s3 folders.jpg" alt="s3">
<image src="screens/4. s3 pics.jpg" alt="s3">
<image src="screens/5. s3 models.jpg" alt="s3">

## Распределенные вычисления
Чтобы в будущем добиться масштабирования проекта и освоить инструменты распределеной обработки данных, был выбран Apache Spark —  фреймворк для обработки и анализа больших объёмов информации, входящий в инфраструктуру Hadoop. Он позволяет быстро выполнять операции с данными в вычислительных кластерах и поддерживает запуск PySpark-приложений.
Основной скрипт app.py, который можно найти в папке spark проекта выполняется да Spark-кластере Data Proc Яндекс Клауда. Минимальной конфигурацией для быстрого выполнения является мастер-нода класса s3-c4-m16 и 3 compute-ноды той же конфигурации. Объемы памяти возрастают пропорционально объемам обрабатываемых датасетов (Spark хранит и обрабатывает данные в оперативной памяти).
Для корректного запуска catboost на Spark, необходимо найти соответствующий используемым в окружении весиям scala и spark пакет maven и указать его в команде spark-submit. Для данного проекта это необходимо делать следующим образом:
```sh
spark-submit --packages ai.catboost:catboost-spark_3.0_2.12:1.2.2 app.py
```
Ресурс maven: https://maven.apache.org/index.html

Запуск catboost_spark на кластере Data Proc:

<image src="screens/1. запуск catboost-spark.jpg" alt="catboost">

## Запуск по расписанию
Бизнесу важно получать скор батчовых моделей в указанный интервал времени с точностью до минуты. Это связано с тем, что после скоринга на его основе запускаются внутренние сценарии, влияющие на бизнес-метрики и работу людей. Чтобы поставить процесс в проекте на расписание, изспользовался Apache Airflow - оркестратор, позволяющий наладить разработку, планирование и мониторинг сложных рабочих процессов. 
В основе концепции Airflow лежит DAG — направленный ациклический граф. 
В проекте даги предстваляют из себя следующие процессы из задач (тасок - "tasks"), выполняемых последовательно:

dag_app.py:
__sftp_task2 >> exec_task2 >> send_email_task__
Запускает трансфер скрипта приложения на Spark-кластер, затем его запуск и отправку отчета о запуске процесса

dag_new_data.py:
__sftp_task1 >> exec_task2 >> send_email_task__
Запускает трансфер скрипта генерации новых данных на Spark-кластер, затем его запуск и отправку отчета о поступлении новых данных

В папке dags можно найти два вышеуказанных дага, а так же даг с PythonOperator и зависимостями в папке bots, где находится пример текста сообщения "Testing success", направляемого на email получателя.

Второй параметр метода x.login("<gmail>", "") пропущен (является персональными данными автора проета) - для запуска необходимо указать свой app-password на google-account.

Для успешного запуска скрипта на Spark-кластере под оркестрацией Airflow важно настроить connect между кластером и виртуальной машиной, на которой развернут Airflow. Для этого необходимо установить недостающие провайдеры и создать ssh-connection в UI Airflow:

<image src="screens/6. air providers.jpg" alt="providers">
<image src="screens/6. air ssh.jpg" alt="ssh">

Выполнение и отладку дагов так же можно наблюдать через UI:

<image src="screens/14. email dag2.jpg" alt="email">
<image src="screens/12. good log 3.jpg" alt="log">
<image src="screens/13. email dag.jpg" alt="log_email">

Убедиться в корректной рассылке email-сообщений можно, используя 2 собственных google-ящика:

<image src="screens/15. email send received.jpg" alt="email">







