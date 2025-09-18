
---

### **System Design for Real-Time AI Applications – Core Components**

| **#** | **Component**                 | **Description**                                                      | **Tools & Technologies**                                  |
| ----- | ----------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| 1     | Stream Processing             | Constant flow of live data for real-time analysis.                   | Apache Kafka, Apache Flink, Apache Spark Streaming        |
| 2     | Real-Time Inference           | Makes predictions on live data with low latency.                     | TensorFlow Serving, Triton Inference Server, ONNX Runtime |
| 3     | Event-Driven Systems          | Triggers actions based on specific events.                           | AWS EventBridge, Apache Pulsar, Kafka Streams             |
| 4     | Data Queues                   | Buffers and organizes data to handle bursts smoothly.                | RabbitMQ, Kafka, Amazon SQS                               |
| 5     | Low-Latency Serving           | Delivers model output in milliseconds.                               | TensorRT, NVIDIA Jetson, FastAPI                          |
| 6     | Edge Computing                | Runs models close to the source for lower latency and bandwidth use. | AWS IoT Greengrass, Google Coral                          |
| 7     | Model Optimization            | Reduces model size and complexity for faster execution.              | OpenVINO, Hugging Face Optimum                            |
| 8     | Real-Time Feature Engineering | Transforms raw data into features instantly.                         | Tecton, Amazon SageMaker Feature Store, GCP Feature Store |
| 9     | Auto-Scaling                  | Dynamically adjusts resources based on demand.                       | Kubernetes HPA, AWS Auto Scaling                          |
| 10    | Monitoring and Alerting       | Tracks system health and sends alerts on anomalies.                  | AWS CloudWatch, Datadog, Prometheus, Grafana              |

---

