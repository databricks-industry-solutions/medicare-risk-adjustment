# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are user-specific, so you can alter the workflow and cluster via UI without affecting other users. Running this script again after modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /databricks/driver/
# MAGIC wget -N https://amir-hls.s3.us-east-2.amazonaws.com/public/263572c0_25a1_46ce_9009_2ae456966ea9-smolder_2_12_0_0_1_SNAPSHOT-615ef.jar -P /dbfs/tmp/smolder/jar/

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

hls_jsl_cluster = dbutils.secrets.get("solution-accelerator-cicd", "hls_jsl_cluster_v4") # This cluster is available in Databricks' internal environment only. Reach out to Databricks and JSL sales engineering and get a cluster set up to run these accelerators in your own environment
job_json = {
        "timeout_seconds": 7200,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "HLS"
        },
        "tasks": [
            {
                "existing_cluster_id": hls_jsl_cluster,
                "notebook_task": {
                    "notebook_path": "00-README"
                },
                "task_key": "MRA_01",
                "description": ""
            },
            {
                "existing_cluster_id": hls_jsl_cluster,
                "libraries": [],
                "notebook_task": {
                    "notebook_path": "01-medicare-risk-adjustment"
                },
                "task_key": "MRA_02",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "MRA_01"
                    }
                ]
            }
        ]
    }
    

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


