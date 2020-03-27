<h1 align="center">Part 4: Put it in production ☁️</h1>

> <p  align="center"><em>
>  Building data products is more than just putting code into production,<br>
>  it also includes DevOps and lifecycle management of live systems.
> </em></p>

|                   | Server (Hosted)      | Google GCP (Serverless) | Amazon AWS (Serverless) |
|-------------------|----------------------|-------------------------|-------------------------|
| **Concept**       | Manual control       | Abstraction of server   | Abstraction of server   |
| **Hardware**      | Single machine (EC2) | Scalable environment    | Scalable environment    |
| **Usage**         | SSH to the machine   | UI and command line     | UI                      |
| **API Endpoints** | Flask + Guicorn      | Flask + Cloud Functions | Flask + AWS Lambda      |
| **Container**     | Docker               | Kubernetes              | Elastic Container Server (ECS) |
| **Messaging**     | Apache Kafka         | PubSub                  | Kinesis                 |
| **Storage**       | Your hard drive      | Cloud Storage (GCS)     | S3 + Athena             |


# Saving models

- Skealearn
  - Pickle: Errors when your training environment (python 3.7) is different from your production environment (3.6)
  - MLflow: Best


```python
import pickle
pickle.dump(model, open("logit.pkl", 'wb'))
model = pickle.load(open("logit.pkl", 'rb'))
model.predict_proba(x)
```

```python
import mlflow.sklearn
mlflow.sklearn.save_model(model, "models/logit_games_v1")
loaded = mlflow.sklearn.load_model("models/logit_games_v1")
loaded.predict_proba(x)
```

```python
import mlflow.keras
mlflow.keras.save_model(model, "models/keras_games_v1")
loaded = mlflow.keras.load_model("models/keras_games_v1")
loaded.evaluate(x, y, verbose = 0))
```


# Deploy

- Flask models as web API,
- Gunicorn: A WSGI Server for putting API in production
- Heroku: Hosting, free but limited.
- intractive Front End
  - Dash (similar to Shiny)


# Data warehoses

This approach has worked well in the past, but it creates issues when building large-scale data science pipelines. Moving data in and out of a relational database can be a bottleneck for these types of workflows. One of the solutions to this bottleneck is to use vendor solutions that separate storage from compute, such as Snowflake or Delta Lake.
