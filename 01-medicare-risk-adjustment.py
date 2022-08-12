# Databricks notebook source
# MAGIC %md
# MAGIC # Automated Patient Risk Adjustment and Medicare HCC Coding from Clinical Notes
# MAGIC In this notebook, we first use our NLP pipelines to extract disease entities and their correspodning ICD10 codes. In addition we use our models to infer demographic information such as gender at birth and age. We then use these information to calculate the Risk Adjustment Factor.
# MAGIC 
# MAGIC [![](https://mermaid.ink/img/pako:eNp9kcFqwzAMhl9F-NRCC9s1h0HbJIWdxrpbUoYaK6khsYOtbGxN32LPtWeaM7sQGMwXWT-f9AvpIiojSSSisdif4SUtNfjnhlMQvr_gZI3-pKBvQiAt_3JOtW9kg74NYQchpjFm8E99Y9qo7mdU9C0svoM2TO4I6_XDqCp5fweWnJlMR9gWv8ox4hMDY-M7kJ1RuyJIxwmaoSM2NMPSwuexVVb0yIo0g2PkwUV5Gxxq1TLZBM5VBagloHNkWRkd6RHyxeLRKJ1AbPOq5HJ5287UI7-taJZk8yQPVr013o1G2BdPcaLnTQ6HythpVrESHdkOlfTHvEyVpeAzdVSKxH8l1Ti0XIpSXz069BKZMqnYWJHU2DpaCRzYHD50JRK2A92gVKE_URep6w9ImbCu)](https://mermaid-js.github.io/mermaid-live-editor/edit/#pako:eNp9kcFqwzAMhl9F-NRCC9s1h0HbJIWdxrpbUoYaK6khsYOtbGxN32LPtWeaM7sQGMwXWT-f9AvpIiojSSSisdif4SUtNfjnhlMQvr_gZI3-pKBvQiAt_3JOtW9kg74NYQchpjFm8E99Y9qo7mdU9C0svoM2TO4I6_XDqCp5fweWnJlMR9gWv8ox4hMDY-M7kJ1RuyJIxwmaoSM2NMPSwuexVVb0yIo0g2PkwUV5Gxxq1TLZBM5VBagloHNkWRkd6RHyxeLRKJ1AbPOq5HJ5287UI7-taJZk8yQPVr013o1G2BdPcaLnTQ6HythpVrESHdkOlfTHvEyVpeAzdVSKxH8l1Ti0XIpSXz069BKZMqnYWJHU2DpaCRzYHD50JRK2A92gVKE_URep6w9ImbCu)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disease Coding and Demographic Inference
# MAGIC We first use NLP pipelines to infer ICD10 codes as well as patinet age and gender at birth.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0. Initial configurations

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.util import *
from sparknlp.annotator import *
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
# from sparknlp.pretrained import ResourceDownloader

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
# from sparknlp.training import CoNLL

# pd.set_option('max_colwidth', 100)
# pd.set_option('display.max_columns', 100)  
# pd.set_option('display.expand_frame_repr', False)

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

notes_path='/FileStore/HLS/nlp/data/'
delta_path='/FileStore/HLS/nlp/delta/jsl/'
dbutils.fs.mkdirs(notes_path)
os.environ['notes_path']=f'/dbfs{notes_path}'

# COMMAND ----------

# MAGIC %md
# MAGIC #### data preparation
# MAGIC In this notebook we will use the transcribed medical reports in [www.mtsamples.com](https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/databricks/python/healthcare_case_studies/www.mtsamples.com). 
# MAGIC You can download those reports by the script [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/mt_scrapper.py).
# MAGIC We will use slightly modified version of some clinical notes which are downloaded from [www.mtsamples.com](https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/databricks/python/healthcare_case_studies/www.mtsamples.com).

# COMMAND ----------

# DBTITLE 1,sample data download
# MAGIC %sh
# MAGIC cd $notes_path
# MAGIC wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/mt_oncology_10.zip
# MAGIC unzip -o mt_oncology_10.zip

# COMMAND ----------

display(dbutils.fs.ls(f'{notes_path}/mt_oncology_10'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Read Data and Write to Bronze Delta Layer
# MAGIC In this dataset, each file represents a note for a patient. Perhpas in real applications we can have multiple notes for a given patient.
# MAGIC To make our downstream processing easier, we assign a unique id to each note and also a patinet id. 
# MAGIC We load notes as a deltatable and combine with other information available and then write the resulting table into the bronze layer.

# COMMAND ----------

# DBTITLE 1,create notes dataset
notes_df = (
  sc.wholeTextFiles(f'{notes_path}/mt_oncology_10/mt_note_0*.txt')
  .toDF()
  .withColumnRenamed('_1','path')
  .withColumn('note_id',F.abs(F.hash('path')%5000))
  .withColumn('patient_id',F.abs(F.hash('path')%1000))
  .withColumnRenamed('_2','text') 
)
display(notes_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC In addition, in real life applications, there are administrative information regarding a patinet's status that can not be found in a note. 
# MAGIC Bellow We manually creat a mock datasets and later combine extratced IC10 codes for each patient with this dataset.

# COMMAND ----------

# DBTITLE 1,create a mock patient status dataset
patient_ids=[m.patient_id for m in notes_df.select('patient_id').collect()]
eligibility_codes=["CFA", "CND", "CPA"]
orec_codes=["0","1","2","3"]

np.random.seed(420)

def random_selection(choices_arr,sz):
  return(list(np.random.choice(choices_arr,sz)))

df_len=len(patient_ids)
patient_status_df = spark.createDataFrame(pd.DataFrame(
  {"patient_id" : patient_ids,
  "eligibility" : random_selection(eligibility_codes,df_len),
  "orec" : random_selection(orec_codes,df_len),
  "medicaid":random_selection([True, False],df_len),
}))
display(patient_status_df)

# COMMAND ----------

# DBTITLE 1,Write to delta
notes_df.write.format('delta').mode('overwrite').save(f'{delta_path}/bronze/mt-oc-notes')
patient_status_df.write.format('delta').mode('overwrite').save(f'{delta_path}/bronze/patient-status')
display(dbutils.fs.ls(f'{delta_path}/bronze/'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. ICD-10 code extraction
# MAGIC Now, we will create a pipeline to extract ICD10 codes. This pipeline will find diseases and problems and then map their ICD10 codes. We will also check if this problem is still present or not.

# COMMAND ----------

# DBTITLE 1,ICD10 extraction pipeline
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")
 
sentenceDetector = SentenceDetectorDLModel.pretrained()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")
 
tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")\
 
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")
 
c2doc = Chunk2Doc()\
      .setInputCols("ner_chunk")\
      .setOutputCol("ner_chunk_doc") 
 
clinical_ner = MedicalNerModel.pretrained("ner_jsl_enriched", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
 
ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Oncological", "Disease_Syndrome_Disorder", "Heart_Disease"])
 
sbert_embedder = BertSentenceEmbeddings\
      .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sbert_embeddings")
 
icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")\
    .setReturnCosineDistances(True)
 
clinical_assertion = AssertionDLModel.pretrained("jsl_assertion_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
 
resolver_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        ner_converter,
        c2doc,
        sbert_embedder,
        icd10_resolver,
        clinical_assertion
    ])
 
data_ner = spark.createDataFrame([[""]]).toDF("text")
 
icd_model = resolver_pipeline.fit(data_ner)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we use this model to transform 

# COMMAND ----------

df=notes_df.select('note_id','patient_id','text')
icd10_df = icd_model.transform(df)
icd10_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see how our model extracted ICD Codes on a sample.

# COMMAND ----------

sample_text = df.limit(3).select("text").collect()[0][0]
print(sample_text)

# COMMAND ----------

light_model = LightPipeline(icd_model)
light_result = light_model.fullAnnotate(sample_text)

from sparknlp_display import EntityResolverVisualizer
vis = EntityResolverVisualizer()
# Change color of an entity label
vis.set_label_colors({'PROBLEM':'#008080'})
icd_vis = vis.display(light_result[0], 'ner_chunk', 'icd10cm_code', return_html=True)
displayHTML(icd_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ICD resolver can also tell us HCC status. HCC status is 1 if the Medicare Risk Adjusment model contains ICD code.

# COMMAND ----------

# DBTITLE 1,HCC status extraction
icd10_hcc_df = icd10_df.select("patient_id","note_id", F.explode(F.arrays_zip(icd10_df.ner_chunk.result, 
                                                                   icd10_df.icd10cm_code.result,
                                                                   icd10_df.icd10cm_code.metadata,
                                                                   icd10_df.assertion.result
                                                                  )).alias("cols")) \
                            .select("patient_id","note_id", F.expr("cols['0']").alias("chunk"),
                                    F.expr("cols['1']").alias("icd10_code"),
                                    F.expr("cols['2']['all_k_aux_labels']").alias("hcc_list"),
                                    F.expr("cols['3']").alias("assertion")
                                   ).cache()
icd10_hcc_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC We can now filter the ICD codes based on HCC status and add hcc_status column. We then write the resulting table into the silver layer.

# COMMAND ----------

from pyspark.sql.types import StringType

@udf(StringType())
def get_hcc_status(hcc_list):
  return(hcc_list.split("||")[1])

icd10_hcc_status_df = (
   icd10_hcc_df
   .withColumn("hcc_status",get_hcc_status('hcc_list'))
   .drop('hcc_list')
 )
icd10_hcc_status_df.display()

# COMMAND ----------

icd10_hcc_status_df.write.format('delta').mode('overwrite').save(f'{delta_path}/silver/icd10-hcc-status')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Infer demographic information
# MAGIC In addition to ICD10 codes, we need patient demographic information - gender and age - to calculate the Risk Adjustment Factor (RAF). 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gender at Birth
# MAGIC We will use pre-trained gender classifier to infer assigned gender at birth, using the `ClassifierDLModel`, based on the content of the text.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

tokenizer = Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")\

biobert_embeddings = BertEmbeddings().pretrained('biobert_pubmed_base_cased') \
        .setInputCols(["document",'token'])\
        .setOutputCol("bert_embeddings")

sentence_embeddings = SentenceEmbeddings() \
     .setInputCols(["document", "bert_embeddings"]) \
     .setOutputCol("sentence_bert_embeddings") \
     .setPoolingStrategy("AVERAGE")

genderClassifier = ClassifierDLModel.pretrained('classifierdl_gender_biobert', 'en', 'clinical/models') \
       .setInputCols(["document", "sentence_bert_embeddings"]) \
       .setOutputCol("gender")

gender_pipeline = Pipeline(stages=[documentAssembler,
                                   #sentenceDetector,
                                   tokenizer, 
                                   biobert_embeddings, 
                                   sentence_embeddings, 
                                   genderClassifier])

# COMMAND ----------

# DBTITLE 1,create the gender model
data_ner = spark.createDataFrame([[""]]).toDF("text")
gender_model = gender_pipeline.fit(data_ner)

# COMMAND ----------

# DBTITLE 1,load raw notes from bronze
notes_df=spark.read.load(f'{delta_path}/bronze/mt-oc-notes')

# COMMAND ----------

# DBTITLE 1,infer gender 
gender_info_df = gender_model.transform(notes_df)
inferred_gender_details_df=(
  gender_info_df
  .select("patient_id",F.explode(F.arrays_zip(gender_info_df.gender.result,gender_info_df.gender.metadata)).alias("inferred_gender"))
)
inferred_gender_details_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC We also write the resulting details of the gender inference to silver layer for future access. 

# COMMAND ----------

inferred_gender_details_df.write.format('delta').mode('overwrite').save(f'{delta_path}/silver/inferred-gender-details')

# COMMAND ----------

inferred_gender_details_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Detecting Age
# MAGIC We can get patient's age from the notes by another pipeline. We are creating an age pipeline to get `AGE` labelled entities. In a note, more than one age entity can be extracted. We will get the first age entity as patient's age.

# COMMAND ----------

# DBTITLE 1,create the age inference model
date_ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Age"])

age_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        date_ner_converter
    ])

data_ner = spark.createDataFrame([[""]]).toDF("text")

age_model = age_pipeline.fit(data_ner)

# COMMAND ----------

# DBTITLE 1,visualize age entities in a sample text
sample_text = notes_df.limit(3).select("text").collect()[0][0]

light_model = LightPipeline(age_model)
light_result = light_model.fullAnnotate(sample_text)

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)

displayHTML(ner_vis)

# COMMAND ----------

# DBTITLE 1,apply the pipeline
age_result = age_model.transform(notes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see on the sample text, there can be multiple age entities in a given text, and not all of those entities may refer to the patient's age. Our age detection model, extracts age and assignes a confidcne level to ahc inferred entity. We store all the inferred information in a delta table that will be used later to seelct the most likely age entity as the age of the patient.

# COMMAND ----------

inferred_age_details=(
  age_result
  .select("patient_id",F.explode(F.arrays_zip(age_result.ner_chunk.result, age_result.ner_chunk.metadata))
  .alias("age_details"))
  .cache()
)
inferred_age_details.display()

# COMMAND ----------

# DBTITLE 1,write data to silver
inferred_age_details.write.format('delta').mode('overwrite').save(f'{delta_path}/silver/inferred-age-details')

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculating Medicare Risk Adjusment Score

# COMMAND ----------

# MAGIC %md
# MAGIC ## load iformation from silver tables
# MAGIC Now, that we have all data which we extracted from clinical notes, we can calculate Medicare Risk Adjusment Score.

# COMMAND ----------

# DBTITLE 1,load data from silver
patient_status_df=spark.read.load(f'{delta_path}/bronze/patient-status')
icd10_hcc_status_df = spark.read.load(f'{delta_path}/silver/icd10-hcc-status')
inferred_age_details_df = spark.read.load(f'{delta_path}/silver/inferred-age-details')
inferred_gender_details_df = spark.read.load(f'{delta_path}/silver/inferred-gender-details')

# COMMAND ----------

# MAGIC %md
# MAGIC We filter records to limit only to the records with `hcc_status=1` and assertion status excluding "Family", "Past" and "Absent", indications.

# COMMAND ----------

icd10_hcc_status_df.display()

# COMMAND ----------

icd10_df= ( 
  icd10_hcc_status_df
 .filter("hcc_status=1")
 .filter("assertion not in ('Family','Past', 'Absent')")
 .drop('hcc_status','assertion')
 .selectExpr('patient_id',"chunk as indication",'icd10_code')
)
icd10_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we group all records by patient, to create a dataset of all indications and respective icd10 codes for each patient:

# COMMAND ----------

# DBTITLE 1,prepare ICD10 codes (1 row per patient)
icd10_grouped_df=icd10_df.groupBy('patient_id').agg(F.collect_list('icd10_code').alias('icd10_codes'),F.collect_list('indication').alias('indications'))
icd10_grouped_df.display()

# COMMAND ----------

# DBTITLE 1,prepare age dataset
age_df = (
  inferred_age_details_df
  .selectExpr('patient_id','age_details.`0` as hypothetical_age','age_details.`1`.confidence')
  .select('patient_id',F.regexp_extract('hypothetical_age', '(\\d+)', 0).alias('age'),'confidence')
  .orderBy(F.desc('confidence'))
  .groupBy('patient_id')
  .agg(F.first('age').alias('inferred_age'))
)
display(age_df)

# COMMAND ----------

# DBTITLE 1,prepare gender dataset
gender_df = inferred_gender_details_df.selectExpr('patient_id','inferred_gender.`0` as inferred_gender')
gender_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can combine all the information we have about the patinets to calculate the RAF score.

# COMMAND ----------

patients_df =(
  patient_status_df
  .join(icd10_grouped_df,on='patient_id')
  .join(age_df,on='patient_id')
  .join(gender_df,on='patient_id')
)

# COMMAND ----------

# DBTITLE 1,Patient dataset
display(patients_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate RAF
# MAGIC Now that we have all the information required for calculating RAF, we use `from sparknlp_jsl.functions.profile` function to calculate the score.

# COMMAND ----------

from sparknlp_jsl.functions import profile
from pyspark.sql.types import *
import pyspark.sql.functions as F

def get_profile():
  hcc_profile_schema = StructType([
            StructField('risk_score', FloatType()),
            StructField('hcc_lst', StringType()),
            StructField('parameters', StringType()),
            StructField('details', StringType())])
  profile_str=profile(
    patients_df.icd10_codes,
    patients_df.inferred_age,
    patients_df.inferred_gender,
    patients_df.eligibility,
    patients_df.orec,
    patients_df.medicaid
  )
  return(F.from_json(profile_str, hcc_profile_schema).alias("hcc_profile"))

# COMMAND ----------

# DBTITLE 1,Patient risk scores
patients_risk_df=patients_df.withColumn('patient_id', get_profile().risk_score.alias('risk_score'))
patients_risk_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we have all the risk scores corresponding to each patient. We can write this dataset to gold for future access

# COMMAND ----------

patients_risk_df.write.format('delta').mode('overwrite').save(f'{delta_path}/gold/patients-risk')

# COMMAND ----------

# MAGIC %md
# MAGIC # License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |BeautifulSoup|MIT License|https://www.crummy.com/software/BeautifulSoup/#Download|https://www.crummy.com/software/BeautifulSoup/bs4/download/|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
