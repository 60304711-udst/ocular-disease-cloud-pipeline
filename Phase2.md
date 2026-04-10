# Ocular Disease Recognition (ODIR-5K) - Cloud MLOps Pipeline
**Author:** Mohammad Shurbaji  
**Student ID:** 60304711  
**Course:** CCIT-DSAI3202 | University of Doha for Science and Technology  
**Phase:** 2  (Model Training, MLOps, and Live Deployment)

---

## I. Project Objective
Building upon the Phase 1 data foundation, this phase focuses on training a deep learning model (ResNet50) entirely within the Azure ecosystem. The primary goal was to transition from a local prototype to a production-ready **Managed Online Endpoint** capable of real-time clinical predictions.

## II. Pipeline Architecture
The architecture utilizes a decoupled Cloud MLOps approach:
1. **Compute:** Azure Machine Learning Compute Cluster (`Standard_DS3_v2`).
2. **Storage:** Azure Blob Storage mounted via `rslex` for high-throughput data access.
3. **Tracking:** **MLflow** integration for experiment telemetry and model versioning.
4. **Deployment:** Managed Online Endpoint using a Blue/Green deployment strategy via Azure CLI v2.

## III. Implementation Details

### III.1. Model Training
* **Architecture:** ResNet50 (Transfer Learning).
* **Environment:** PyTorch 2.4.0 with MLflow tracking.
* **Optimization:** Cross-Entropy Loss with an Adam optimizer, trained on the pre-processed (CLAHE enhanced) ODIR-5K dataset.
* **Performance:** Achieved a **74% Validation Accuracy** across 8 distinct disease classes.

### III.2. Managed Deployment
* **Endpoint Name:** `ocular-disease-endpoint`
* **Deployment Name:** `ultimate-victory`
* **Infrastructure:** Managed Online Inference with a custom environment to bypass registry and dependency conflicts.

---

## IV. Engineering Roadblocks & Solutions
Instructors and Cloud Architects value troubleshooting. This project navigated three critical failure points:

1. **Pathing & Metadata Error:** * *Issue:* The inference server failed to locate model artifacts within the MLflow structure.
   * *Solution:* Injected the `MLFLOW_MODEL_FOLDER: "model"` environment variable into the deployment YAML.

2. **Dependency Versioning Conflict (`post101` Error):** * *Issue:* The training cluster utilized a specific PyTorch version (`2.4.0.post101`) unavailable in public registries, causing image build failures.
   * *Solution:* Authored a manual `conda_fix.yml` to override model metadata and force-load a stable, production-ready environment.

3. **Registry Access Bug:**
   * *Issue:* Azure "Curated" images encountered internal registry timeout errors.
   * *Solution:* Bypassed curated images by utilizing a generic Microsoft base image (`mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04`) and building the stack manually via the Conda specification.

---

## V. Final Results

### Training Accuracy
Training Accuracy Graph <img width="782" height="520" alt="image" src="https://github.com/user-attachments/assets/84a62d88-e05a-4242-aa5a-7d49ec69c690" />

*The model shows a steady learning curve reaching 74%, indicating a robust feature extraction from the CLAHE-enhanced fundus images.*

### Live Deployment & Environment Provisioning
<img width="959" height="443" alt="image" src="https://github.com/user-attachments/assets/ef4cbb36-d896-4715-b3cb-c1aead146b9e" />

*The custom inference environment (victory-env-final:2) was successfully built and provisioned. This confirms the resolution of the PyTorch post101 dependency conflict and the successful containerization of the ResNet50 model for cloud-native deployment.*

---

## VI. Conclusion
This project successfully demonstrates a full-cycle MLOps workflow. By moving away from local hardware and UI-based "wizards" to a **CLI-driven, YAML-configured cloud architecture**, I established a reproducible, scalable foundation for medical AI diagnostics.
