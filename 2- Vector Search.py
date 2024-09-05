# Databricks notebook source
# MAGIC %run ./Includes/Setup

# COMMAND ----------

# MAGIC %md
# MAGIC # Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

vs_client = VectorSearchClient(disable_notice=True)

# COMMAND ----------

vs_endpoint_name = "demo_vs_endpoint"

# Create the endpoint (permission needed)
vs_client.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC View your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Vector Search Index

# COMMAND ----------

source_table_fullname = f"{catalog_name}.{schema_name}.gold_articles_chunks"
vs_index_fullname = f"{catalog_name}.{schema_name}.articles_vs_index_10"


# COMMAND ----------

# create or sync the index
if not index_exists(vs_client, vs_endpoint_name, vs_index_fullname):
    print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
    vs_client.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED", # Sync is manually triggered
        primary_key="chunk_id",

        # Embedding source:
        ## Option 1: Compute embeddings
        embedding_source_column="chunked_text",
        embedding_model_endpoint_name="databricks-bge-large-en"

        ## Option 2: self-managed embeddings (use existing embedding column)
        #embedding_vector_column="embedding",
        #embedding_dimension=1024 # Match your model embedding size (example: GTE)
        
    )
else:
    # trigger a sync to update our vs content with the new data saved in the table
    print(f"Index {vs_index_fullname} already exists. Triggering a sync on endpoint {vs_endpoint_name}...")
    vs_client.get_index(vs_endpoint_name, vs_index_fullname).sync()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for Similar Content

# COMMAND ----------

#from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableLambda

def get_retriever(query, k: int=10):
    if isinstance(query, dict):
        query = next(iter(query.values()))
    
    vs_index = vs_client.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)

    results = vs_index.similarity_search(query_text=query, columns=["chunked_text"], num_results=k, query_type = "hybrid")

    # format the results to be ready for prompt
    chucks = [r[0] for r in results.get("result", {}).get("data_array", [])]
    return "\n\n".join(chucks)

runnable_retriever = RunnableLambda(get_retriever)

# COMMAND ----------

# test our retriever
question = "What is FusionFlow?"
similar_chucks = runnable_retriever.invoke(question)
print(similar_chucks)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup the Foundation Model

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-405b-instruct",
    max_tokens = 300,
)

test_prompt = "What is FusionFlow?"
print(chat_model.invoke(test_prompt).content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assembling the Complete RAG Solution

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


prompt_template = """You are an assistant for academic research. If you don't know the answer, just say that you do not know, don't try to make up an answer. Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {question}

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = (
    {"context": runnable_retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

# COMMAND ----------

question = {"input": "What is FusionFlow?"}
answer = rag_chain.invoke(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the Model to Model Registery in UC

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain
import langchain_community

# Create a new mlflow experiment or get the existing one if already exists.
experiment_name = f"/Users/{current_user}/odsc-rag-workshop"
mlflow.set_experiment(experiment_name)

# set model registery to UC
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.rag_app_odsc"

# get experiment id to pass to the run
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
with mlflow.start_run(experiment_id=experiment_id):
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        rag_chain,
        loader_fn=runnable_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "langchain-community==" + langchain_community.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
