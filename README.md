<p align="center"><img width=60% src="images/customer_churn.png"></p>

<p align="center" style="color:DodgerBlue; font-family:cambria; font-variant: normal; font-size:20pt; font-weight:bold; font-weight: 900">CUSTOMER REMARKETING 
</p>

![Python](https://img.shields.io/badge/python-3.9-blue) ![Pandas](https://img.shields.io/badge/pandas-1.5.3-blue) ![Tensorflow](https://img.shields.io/badge/tensorflow-2.12.0-blue) ![Kubeflow](https://img.shields.io/badge/kubeflow-1.6-blue) ![Docker](https://img.shields.io/badge/docker-4.12-blue) ![Seldon](https://img.shields.io/badge/seldon-1.15.1-blue) ![Kubernetes](https://img.shields.io/badge/kubernetes-1.26-blue) ![Status](https://img.shields.io/badge/status-work%20in%20progress-orange) ![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-green)

# **Table Of Contents**

### **Background**
- A top tier bank is having a problem retaining their customers. They are looking for someone to help them predict the customers that are going to churn so they can   proactively connect with the customer and provide relevant personalized messaging to reduce the Churn Rate

### **Challenge**
- Build a classification model to identify customers that have a high probability of churning
- Ensure train and deployment pipelines are created separately to mirror real world scenarios
- Create each ML stage as a reusable component
- Automate entire workflow with relevant technologies

### **Requirements**
- Input file with historical data is uploaded to an input_folder on GCS for training our model
- Weekly file with additional data is uploaded to a weekly_folder on GCS for us to apply our trained model
- Output file with probability scores for each customer is posted in the output_folder on GCS
- Trigger main pipeline from most recent weekly file uploaded to GCS weekly_folder 
- The main pipeline uses the latest training image to deploy the model object and apply it to the new weekly file

### **Additional scope**
- Track metrics of deployed model using Grafana and Prometheus
- Integrate MLFlow into the Kubeflow pipelines for model registry and metadata store
- Include explainability concepts using SHAP
- Deploy cloud functions using Terraform for futher automation of pipeline 

### **High level architecture**
<p align="center"><img width=60% src="images/customer_churn_architecture.png"></p>

- Experiment in notebook to train-test-evaluate historical data in GCS 
- Create separate classes, and function script files
- Create a kubeflow training and deployment pipeline including all classes, functions
- Create a cloud storage trigger 01 and function 01 to invoke the main pipeline
- Main pipeline is invoked by cloud function 01 when a new weekly batch file is uploaded to weekly data folder
- The main pipeline kicks of the training pipeline [Need to add logic that if the training pipeline image hasnt changed dont need to kick it off] 
- Deployment pipeline is invoked by cloud function 02 when a new final model object is uploaded to a GCS bucket 
- Model is deployed on Kubernetes using Seldon and applied to the weekly batch extract on GCS
- Output of the scored file is stored in the GCS bucket to be used by the marketing team
- The processing steps of the pipeline can be monitored within Vertex AI or the Kubeflow UI 

### **Technologies**
- Python, TensorFlow, & Keras (Need to add deep learning models)
- Google Cloud Storage
- Cloud storage trigger
- Cloud functions
- Docker
- Seldon  
- Kubernetes
- Vertex AI

