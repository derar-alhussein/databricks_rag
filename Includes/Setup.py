# Databricks notebook source
# DBTITLE 1,Install required libraries
# MAGIC %pip install --quiet --upgrade databricks-vectorsearch langchain langchain-community PyMuPDF llama_index transformers mlflow[databricks]

# COMMAND ----------

# DBTITLE 1,Restart Python Kernel
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup User-Specific Variables
#setup catalog and show widget at top
dbutils.widgets.text("catalog_name", "labs_43876_cs2a8f")
catalog_name = dbutils.widgets.get("catalog_name")

#break user in their own schema
current_user = spark.sql("SELECT current_user() as username").collect()[0].username
schema_name = f'odsc_workshop_{current_user.split("@")[0].split(".")[0]}'

#create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
print(f"\nUsing catalog & schema: {catalog_name}.{schema_name}")

# COMMAND ----------

def path_exists(path):
  try:
    dbutils.fs.ls(path)
    return True
  except Exception as e:
    if 'java.io.FileNotFoundException' in str(e):
      return False
    else:
      raise

def download_dataset(source, target):
    files = dbutils.fs.ls(source)

    for f in files:
        source_path = f"{source}/{f.name}"
        target_path = f"{target}/{f.name}"
        if not path_exists(target_path):
            print(f"Copying {f.name} ...")
            dbutils.fs.cp(source_path, target_path, True)

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
          print(f'An unexpected error occurred while retrieving the index. This may be due to a permission issue.')
          raise e
  return False

# COMMAND ----------

# DBTITLE 1,Configuration
# Reduce the arrow batch size as pdf files can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
