# integrates the processing, train and deploy pipeline
from google.cloud import aiplatform as aip

from kfp.v2 import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component 
from kfp.v2 import compiler

from pipeline_components import (
    processing,
    train,
)

# where is this coming from? 

import os 
from dotenv import load_dotenv

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DEST_BUCKET_URI = os.getenv("DEST_BUCKET_URI")
SOURCE_FILE = os.getenv("SOURCE_FILE")
PIPELINE_BUCKET_URI = os.getenv("PIPELINE_BUCKET_URI")
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")

# API service endpoint
API_ENDPOINT = "{}-aiplatform.googleapis.com".format(REGION)
PIPELINE_ROOT = "{}/pipeline_root/intro".format(PIPELINE_BUCKET_URI)
# print(f"PIPELINE_ROOT DEBUG: {PIPELINE_ROOT}")

@dsl.pipeline(




)
def pipeline(dest_bucket_uri: str, source_file: str):
    processing(run_id = run_id, 
                    dest_bucket_uri = dest_bucket_uri,
                    source_file = source_file).\
            set_cpu_limit('1').\
            set_memory_limit('3G').\
            set_display_name("Ingest Data & Perform EDA")#.\)
    

    train(run_id = run_id, 
        dest_bucket_uri = dest_bucket_uri, 
        source_file = source_file).\
    set_display_name("Train Models").\
    after(perform_eda_task)

    deploy()

    predict()

if __name__ == "__main__":
    aip.init(project=PROJECT_ID, staging_bucket=PIPELINE_BUCKET_URI)

    compiler.Compiler().compile(pipeline_func=pipeline, package_path="churn_pipeline.json")

    DISPLAY_NAME = "churn_pipeline"

    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="churn_pipeline.json",
        # pipeline_root is where information is saved off for every run
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values = {
            "dest_bucket_uri": DEST_BUCKET_URI,
            "source_file": SOURCE_FILE
        }
    )

    print(type(job))

    job.run()