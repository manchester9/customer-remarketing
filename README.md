<p align="center"><img width=60% src="images/customer_remarketing.png"></p>

<p align="center" style="color:DodgerBlue; font-family:cambria; font-variant: normal; font-size:20pt; font-weight:bold; font-weight: 900">CUSTOMER REMARKETING 
</p>

![Python](https://img.shields.io/badge/python-3.9-blue) ![Pandas](https://img.shields.io/badge/pandas-1.5.3-blue) ![Tensorflow](https://img.shields.io/badge/tensorflow-2.12.0-blue) ![Kubeflow](https://img.shields.io/badge/kubeflow-1.6-blue) ![Docker](https://img.shields.io/badge/docker-4.12-blue) ![Seldon](https://img.shields.io/badge/seldon-1.15.1-blue) ![Kubernetes](https://img.shields.io/badge/kubernetes-1.26-blue) ![Status](https://img.shields.io/badge/status-work%20in%20progress-orange) ![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-green)

# **Table Of Contents**

### **Background**
- A top tier Portuguese bank aims to promote a marketing campaign with existing customers to increase their term deposits. A similar campaign that was run last quarter showed an average single-digit conversion rate. In the last town hall, the marketing head wants the analytics team to come up with an effective remarketing model to increase the conversion ratio to double-digit with same budget as per the last campaign

### **Challenge**
- Build a classification model to identify customers that have a high probability of converting into depositors based on their historical transaction history
- Ensure train and deployment pipelines are created separately to mirror real world scenarios
- Automate entire workflow with relevant technologies 
- Create each ML stage as a reusable component
- Automate entire workflow with relevant technologies

### **Requirements**
- `Input data` folder on GCS has the latest historical data for training
- `Weekly data` folder is created to receive weekly files that need to be scored 
- `Output data` folder is created to receive output files with probability scores for each customer
- Trigger training pipeline when new input data file is added to a GCS folder. Output of the training pipeline is a final model object placed in a GCS folder called `Model output`
- Trigger deploy pipeline when new weekly file is submitted to the frontend UI built with Streamlit 
    - Use the FastAPI frontend web endpoint in conjunction with google cloud  storage to upload the file to a GCS bucket 
    - Deploy the model using k-serve [Load, create deployment specifications, dockerization, push to GCR, deploy on Kubernetes, generate predictions, & save on GCS bucket]
    - Return status on frontend UI
      return {"status": "success", "predictions URL": predictions_URL}
- Visualization of pipeline can be done in Kubeflow

### **Additional scope**
- Track metrics of deployed model using Grafana and Prometheus
- Integrate MLFlow into the Kubeflow pipelines for model registry and metadata store
- Include explainability concepts using SHAP
- Deploy GCP infrastructure using Terraform for futher automation of pipeline 

### **High level architecture**
<p align="center"><img width=60% src="images/customer_churn_architecture.png"></p>

### **Implementation plan for the above architecture** 
- Experiment in notebook to train-test-evaluate historical data in GCS 
- Create separate classes, functions script files
- Create a sklearn training pipeline including all classes, functions
- The training pipeline is triggered when there is a new input data file added to a GCS bucket
- The output of this is a final model object which is stored on a GCS bucket 
- Kubeflow deployment pipeline is triggered by cloud functions which is invoked by a GCS trigger on receiving a new weekly file that needs to be scored
- The serialized Model (ONNX) is deployed on Kubernetes  
- Output of the scored features is stored in on a GCS bucket 
- The client gets a message on success of predictions within the UI 
- Visualization of entire end to end pipeline in Kubeflow 

### **Technologies**
- Python, TensorFlow, & Keras (Need to add deep learning models)
- Google Cloud Storage
- Cloud storage trigger
- Cloud functions
- Docker
- FastAPI
- Uvicorn
- Streamlit
- Kubernetes
- K-serve

