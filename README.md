# ocular-disease-cloud-pipeline
# Multi-Class Ocular Disease Recognition System: Phase 1 Data Pipeline

## I. Project Objective
This project addresses the challenge of diagnosing multiple retinal conditions (Diabetic Retinopathy, Glaucoma, Cataracts, etc.) using a unified cloud-based AI system. Phase 1 focuses on building a reproducible, cloud-native data foundation using Azure Blob Storage.

## II. Pipeline Architecture & Deliverables

### II.1. Data Ingestion
* **Data Source:** Ocular Disease Intelligent Recognition (ODIR-5K) dataset (~2GB) from Kaggle.
* **Ingestion Mode:** Batch processing. The raw dataset is streamed safely to the cloud to prevent local memory overload.
* **Storage Layout:** Azure Blob Storage utilizing a Data Lake architecture with specific zones:
  * `raw-data` container: Stores the immutable, versioned original data.
  * `processed-data` container: Stores the cleaned and transformed features ready for ML training.

### II.2. ETL Process & Data Quality
* **Cleaning & Validation:** The pipeline verifies image integrity and standardizes formats using OpenCV. 
* **PII Handling:** Patient metadata is governed, ensuring any identifiable ID columns are decoupled from the training features to maintain privacy.
* **Reproducibility:** Transformations are executed via an automated Python pipeline (`etl_pipeline.py`) rather than manual manipulation.

### II.3. Cataloging and Governance
The schema, data types, and lineage are formally documented in the `data_catalog.json` file included in this repository. It defines the transition from unstructured JPGs to engineered ML features.

### II.4. Exploratory Analysis & II.5 Feature Extraction
* **EDA:** Analysis focuses on assessing the multi-label class distribution and preparing a uniform dataset.
* **Feature Engineering:** The core feature extraction utilizes **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. 
* **Justification:** Retinal images often suffer from poor lighting. CLAHE enhances the visibility of retinal vessels and microaneurysms, which are critical biomarkers for diseases like Glaucoma and Diabetic Retinopathy. Images are resized to a uniform 224x224 dimension to ensure tensor compatibility for future ResNet50 model training.