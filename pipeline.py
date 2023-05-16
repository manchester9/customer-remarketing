#!/usr/bin/env/python3

"""Pipeline customer-remarketing."""

from kfp.components import load_component_from_file
from kfp import dsl
from kfp import compiler
from kfp.gcp import use_gcp_secret
import json
from string import Template

download_op = load_component_from_file("components/download_component.yml")
upload_op = load_component_from_file("components/upload_component.yml")
ingestion_op = load_component_from_file("components/data_ingestion_component.yml")  # pylint: disable=not-callable
prep_op = load_component_from_file("components/pre_processing_component.yml")  # pylint: disable=not-callable
train_op = load_component_from_file("components/train_component.yml")
deploy_op = load_component_from_file("components/k8s_apply_component.yml")

@dsl.pipeline(
    name="Training pipeline", description=""
)
def customer_remarketing(deploy="sklearn-deployment", namespace="kubeflow-user-example-com", model_uri="gs://churn_data_382919/model/data"):
    """Thid method defines the pipeline tasks and operations"""

    download_data = (
        download_op(
            gcs_path='gs://churn_weekly_input_data_382919',
       ).set_display_name("Download data from GCS") 
    ).apply(use_gcp_secret('user-gcp-sa'))

    ingestion_task = (
        ingestion_op(
            input_data=download_data.outputs["Data"],
        ).set_display_name("Data Ingestion and Split")
    )

    prep_task = (
        prep_op(
            input_data=ingestion_task.outputs["output_data"],
        ).after(ingestion_task).set_display_name("Data Preparation")
    )

    train_task = (
        train_op(
            input_data=prep_task.outputs["output_data"],
        ).after(prep_task).set_display_name("Training")
    )

    upload_model = (
        upload_op(
            data=train_task.outputs["output_data"],
            gcs_path='gs://churn_data_382919/model'
        ).after(train_task).set_display_name("Upload model to GCS")
    ).apply(use_gcp_secret('user-gcp-sa'))

    seldon_serving_json_template = Template("""
    {
        "apiVersion": "machinelearning.seldon.io/v1alpha2",
        "kind": "SeldonDeployment",
        "metadata": {
            "name": "$deployment_name"
        },
        "spec": {
            "name": "iris",
            "predictors": [
                {
                    "graph": {
                        "children": [],
                        "implementation": "SKLEARN_SERVER",
                        "modelUri": "$model_uri",
                        "name": "classifier",
                        "envSecretRefName": "seldon-rclone-secret"
                    },
                    "name": "default",
                    "replicas": 1
                }
            ]
        }
    }
    """)

    seldon_serving_json = seldon_serving_json_template.substitute({ 'deployment_name': str(deploy),'namespace': str(namespace),'model_uri': str(model_uri)})

    deploy_task = (
        deploy_op(
            object=seldon_serving_json,
            namespace=namespace,
        ).after(upload_model).set_display_name("Deploy model with Seldon")
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        customer_remarketing, package_path="customer-remarketing.yaml"
    )
