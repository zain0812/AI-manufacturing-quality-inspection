# AI-Powered Manufacturing Quality Inspection

An experimental computer vision project combining **PyTorch, Microsoft Fabric Lakehouse, and Power BI** to explore anomaly detection for manufacturing quality control.

## Project Overview

Manufacturing quality inspection often requires identifying defects that may be difficult to detect consistently through manual inspection. This project explores an AI-based approach that learns patterns from normal bottle images and flags images that may contain anomalies.

The solution uses the bottle category from the MVTec Anomaly Detection (MVTec AD) dataset.

**Project status:** Experimental prototype completed; model improvement and further validation are needed before practical deployment.

## Objectives

- Organize image data in Microsoft Fabric Lakehouse.
- Build an image inventory using PySpark.
- Preprocess images for deep learning.
- Train a convolutional autoencoder using PyTorch.
- Calculate image reconstruction errors and flag potential anomalies.
- Store inspection results in Lakehouse Delta tables.
- Build a Power BI report for quality-inspection analysis.

## Technology Stack

| Technology | Purpose |
|---|---|
| Microsoft Fabric | Data engineering and analytics environment |
| Fabric Lakehouse | Image storage and Delta tables |
| PySpark | File inventory and data processing |
| Python and Pillow | Image processing |
| NumPy | Numerical image representation |
| PyTorch | Autoencoder training and inference |
| Power BI | Inspection results visualization |

## Solution Architecture

1. **Data ingestion:** Organize the MVTec AD bottle dataset in the Fabric Lakehouse.
2. **Inventory:** Record dataset files and their metadata.
3. **Preprocessing:** Convert images to RGB, resize them to 128 × 128 pixels, and normalize pixel values.
4. **Model training:** Train a convolutional autoencoder using normal bottle images.
5. **Anomaly scoring:** Calculate reconstruction errors and apply a threshold to flag potential anomalies.
6. **Data storage:** Save inspection results and category-level summaries as Delta tables.
7. **Reporting:** Use a Power BI semantic model and report to explore inspection results.

## Dataset

This project uses the **bottle category** of the MVTec Anomaly Detection dataset.

The dataset includes normal training images, normal and defective test images, and ground-truth masks for defective regions.

Dataset source: https://www.mvtec.com/company/research/datasets/mvtec-ad

The dataset is licensed under **CC BY-NC-SA 4.0**. Review the dataset's license and attribution requirements before redistribution or commercial use. The dataset itself is not included in this repository.

## Model Approach

The model is a convolutional autoencoder.

An encoder compresses the input image into a lower-dimensional representation, and a decoder attempts to reconstruct the original image.

The reconstruction error measures the difference between the input and reconstructed image. Images with higher reconstruction errors can be flagged as potential anomalies.

The model was trained on normal bottle images. The test images were then scored and compared with their known categories to evaluate the approach.

### Preprocessing

- Image format: RGB
- Image dimensions: 128 × 128 pixels
- Pixel normalization: 0–1
- Training images successfully loaded: 177

Ten listed training files were not included by the Spark binary-file loading step. They were skipped in this experimental run.

## Microsoft Fabric Data Assets

Workspace: `WS_Manufacturing_AI`

Lakehouse: `LH_Manufacturing_AI`

Notebook: `NB_01_Image_Inventory`

| Table | Description |
|---|---|
| `image_inventory` | Inventory of dataset files |
| `inspection_results` | Image-level reconstruction errors and predictions |
| `quality_inspection_summary` | Category-level inspection summary |
| `model_experiment_metrics` | Experiment metrics recorded during an earlier run |
| `category_inspection_metrics` | Category-level metrics from an earlier evaluation |

**Note:** The earlier experiment-metrics tables may not reflect the final retrained model. The results below refer to the latest recorded evaluation of the restored model.

## Results

The latest evaluation included 83 test images.

| Metric | Result |
|---|---:|
| Test images inspected | 83 |
| Normal test images | 20 |
| Defective test images | 63 |
| Images flagged by the model | 25 |
| True positives | 21 |
| False positives | 4 |
| False negatives | 42 |
| True negatives | 16 |
| Precision | 84.0% |
| Recall | 33.3% |
| F1-score | 47.7% |

The threshold was selected using reconstruction errors from normal test images. Consequently, these figures are exploratory and should not be interpreted as an unbiased estimate of generalization performance.

### Interpretation

The model correctly flagged 21 of the 63 defective test images, while missing 42 defects. It also flagged 4 normal images.

The results demonstrate an end-to-end anomaly-detection workflow, but the low recall means the current model is not suitable for autonomous production quality decisions.

## Power BI Report

The report contains four primary visuals:

- Images inspected KPI
- Images flagged for review KPI
- Flagged images by category
- Individual inspection results table

The dashboard provides an overview of inspection volume, predicted anomalies, and image-level reconstruction errors.

## Limitations and Future Improvements

- Investigate the ten training files that were not loaded.
- Evaluate thresholds using a separate validation set rather than selecting them from the test set.
- Improve recall while monitoring false-positive rates.
- Compare alternative architectures and image resolutions.
- Evaluate performance across defect types and multiple independent test sets.
- Consider pixel-level localization using the ground-truth masks.
- Improve reproducibility by versioning model weights, preprocessing settings, and evaluation metrics.
- Validate the approach under realistic production conditions before deployment.

## Responsible Use

This project is an educational and experimental prototype. Model predictions should be reviewed by a human inspector. It has not been validated for production use or safety-critical decisions.

## Acknowledgements

The dataset is provided by MVTec Software GmbH. Please refer to the official MVTec AD dataset page for the original dataset and licensing information.

## Author

Created as a portfolio project exploring computer vision, machine learning, Microsoft Fabric, and Power BI.
