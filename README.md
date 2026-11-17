# Floorplan Detection

**Computer Vision Research Project for Building Element Detection in 2D Floor Plans**

Floorplan Detection is an early-stage Computer Vision project focused on exploring the detection and segmentation of building elements from 2D residential floor plan images.

The long-term objective is to investigate how image-based building element recognition can support geometric extraction and construction quantity estimation.

> **Project Status:** Sprint 01 — Environment Setup and Dataset Exploration  
> **Development Status:** Early Development

---

## 1. Project Overview

### Background

Building floor plans contain spatial and geometric information about rooms, walls, doors, and other building elements.

Extracting this information manually can be time-consuming, particularly when the information is later used for measurement and construction quantity estimation.

This project explores the use of Computer Vision to identify building elements from floor plan images.

The initial focus is understanding the dataset, preparing image data, and establishing a foundation for future detection or segmentation experiments.

### Objectives

The planned objectives are:

1. Explore and understand floor plan image datasets.
2. Prepare image data for Computer Vision experiments.
3. Investigate methods for identifying building elements.
4. Explore semantic segmentation as a possible approach.
5. Evaluate the accuracy of building element predictions.
6. Investigate the conversion of visual information into geometric measurements.
7. Explore the potential application of geometric measurements for construction quantity estimation.

These objectives describe the intended direction of the project, not completed functionality.

---

## 2. Current Scope

### Current Implementation

The repository is currently in its initial development stage.

The verified project context is:

- Python 3.12.16 virtual environment has been created.
- The virtual environment is located in `.venv/`.
- Dataset acquisition is represented by `download_dataset.py`.
- A preprocessing module exists at `src/preprocessing.py`.
- Dataset exploration and environment preparation are the current priorities.

The presence of a script or module does not necessarily mean its functionality has been fully tested or completed.

### Planned Scope

The following capabilities are planned for future development:

- Floor plan image preprocessing.
- Building element detection or semantic segmentation.
- Segmentation mask generation.
- Geometric extraction.
- Pixel-to-real-world scale calibration.
- Wall length and room area measurement.
- Construction material quantity estimation.
- Material cost estimation.

None of these planned capabilities should be considered implemented until the corresponding code and tests exist.

---

## 3. Dataset

### Current Dataset Source

The dataset currently identified in the project is:

**Kaggle — `asutoshprad/floor-plan-dataset`**

Dataset reference:

https://www.kaggle.com/datasets/asutoshprad/floor-plan-dataset

The dataset is handled through the project's `download_dataset.py` script.

The dataset's annotation structure, class definitions, licensing conditions, and suitability for semantic segmentation still need to be inspected and documented.

### Dataset Directory

The repository uses `dataset/` as its dataset directory.

```text
dataset/
```

The exact internal directory structure and contents should be documented after inspecting the downloaded files.

### Dataset Exploration Tasks

- [ ] Verify dataset download behavior.
- [ ] Inspect the downloaded directory structure.
- [ ] Identify image formats and resolutions.
- [ ] Inspect annotation files, if available.
- [ ] Identify available building element categories.
- [ ] Review dataset licensing and usage conditions.
- [ ] Determine whether annotations can support object detection or semantic segmentation.
- [ ] Document dataset characteristics.

**Important:** CubiCasa5K is not currently established as the dataset used by this repository. It remains a possible alternative for future exploration, subject to compatibility and licensing review.

---

## 4. Technical Stack

### Currently Established

| Component | Status |
|---|---|
| Python | 3.12.16 virtual environment |
| Environment | `.venv/` |
| Operating environment | Ubuntu on WSL |
| Dataset source | Kaggle floor-plan dataset |
| Preprocessing code | `src/preprocessing.py` |
| Dataset download script | `download_dataset.py` |

### Not Yet Established

| Technology | Status |
|---|---|
| PyTorch | Not installed / not verified |
| TensorFlow | Not established |
| U-Net | Planned |
| DeepLabV3+ | Planned |
| OpenCV dependency | Not verified |
| NumPy dependency | Not verified |
| Matplotlib dependency | Not verified |
| Shapely | Planned |
| scikit-image | Planned |
| Training pipeline | Not implemented |
| Model evaluation | Not implemented |

The project currently has no verified dependency manifest (`requirements.txt`).

A dependency manifest should be created after the actual dependencies have been selected and installed.

---

## 5. Repository Structure

The currently identified structure is:

```text
floorplan-detection/
│
├── .venv/
│
├── dataset/
│
├── src/
│   └── preprocessing.py
│
├── download_dataset.py
│
└── README.md
```

This is a representation of the currently identified files and directories, not a complete architecture.

Additional files and directories may be introduced as development progresses.

### Planned Structure

The following structure is a future organizational proposal:

```text
floorplan-detection/
│
├── dataset/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── geometry/
│   └── evaluation/
│
├── configs/
├── experiments/
├── outputs/
├── tests/
│
├── download_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

These planned directories should only be added when the corresponding implementation requires them.

---

## 6. Research Direction

### Semantic Segmentation

Semantic segmentation is one of the candidate approaches for future experiments.

It assigns a class label to each pixel in an image, allowing building elements to be represented as segmentation masks.

Potential target classes:

| Class ID | Class | Status |
|---|---|---|
| 0 | Background | Proposed |
| 1 | Wall | Proposed |
| 2 | Room | Proposed |
| 3 | Door | Proposed |
| 4 | Window | Proposed |
| 5 | Column | Proposed |

The final classes depend on the actual dataset annotations and the project's annotation strategy.

### Candidate Model Architectures

The following architectures are research candidates, not implemented models.

| Architecture | Intended Role | Status |
|---|---|---|
| U-Net | Initial segmentation baseline | Planned |
| DeepLabV3+ | Comparison architecture | Planned |
| SegFormer-B0 | Optional experiment | Planned |

The model selection will be revisited after dataset inspection and resource assessment.

---

## 7. Planned Development Pipeline

The long-term processing pipeline is:

```text
2D Floor Plan Image
        |
        v
Image Preprocessing
        |
        v
Building Element Detection
        |
        v
Segmentation / Prediction
        |
        v
Geometric Extraction
        |
        v
Scale Calibration
        |
        v
Quantity Takeoff
        |
        v
Material Estimation
        |
        v
Cost Estimation
```

This pipeline describes the intended research direction. It is not currently an operational end-to-end system.

---

## 8. Planned Evaluation

If semantic segmentation is implemented, candidate evaluation metrics include:

- Intersection over Union (IoU).
- Mean Intersection over Union (mIoU).
- Dice Score.
- Precision.
- Recall.

If geometric extraction is implemented, potential evaluation metrics include:

- Mean Absolute Error (MAE).
- Relative Error.
- Geometric measurement error.

If material estimation is implemented, quantity estimates can be compared against manually verified ground truth measurements.

No model performance or accuracy results are currently reported.

---

## 9. Development Roadmap

### Sprint 01 — Environment and Dataset Exploration

**Current sprint**

- [x] Establish project direction.
- [x] Identify Python 3.12.16 as the project Python version.
- [x] Create the `.venv` environment.
- [x] Identify the Kaggle dataset source.
- [x] Establish the initial repository files.
- [ ] Verify the active virtual environment.
- [ ] Inspect the dataset contents.
- [ ] Verify the dataset download script.
- [ ] Review dataset licensing.
- [ ] Understand the available annotations.
- [ ] Document dataset characteristics.

**Expected deliverable:** A verified development environment and documented understanding of the dataset.

### Sprint 02 — Dataset Preparation

**Planned**

- Inspect image and annotation formats.
- Define target classes.
- Prepare image and annotation pairs.
- Implement annotation conversion if necessary.
- Validate the processed dataset.

**Expected deliverable:** A dataset suitable for the selected Computer Vision task.

### Sprint 03 — Baseline Model

**Planned**

- Select a suitable model architecture.
- Establish the training environment.
- Implement dataset loading.
- Implement the training pipeline.
- Train the first baseline model.

**Expected deliverable:** First trained model and experiment artifacts.

### Sprint 04 — Model Evaluation

**Planned**

- Evaluate model predictions.
- Calculate appropriate segmentation metrics.
- Visualize prediction results.
- Analyze model limitations.

**Expected deliverable:** Evaluation report.

### Sprint 05 — Model Improvement

**Planned**

- Experiment with alternative architectures.
- Evaluate data augmentation.
- Tune model parameters.
- Compare experimental results.

**Expected deliverable:** Selected segmentation model.

### Sprint 06 — Geometry Extraction

**Planned**

- Extract wall geometry.
- Extract room boundaries.
- Investigate wall centerline extraction.
- Calculate pixel-based measurements.

**Expected deliverable:** Geometry extraction module.

### Sprint 07 — Scale Calibration

**Planned**

- Implement reference-point calibration.
- Convert pixel measurements into physical dimensions.
- Validate measurements against known dimensions.

**Expected deliverable:** Pixel-to-meter conversion module.

### Sprint 08 — Material Estimation

**Planned**

- Calculate wall area.
- Calculate room floor area.
- Define material coefficients.
- Estimate selected material quantities.

**Expected deliverable:** Initial material estimation module.

### Sprint 09 — Cost Estimation

**Planned**

- Define material price data.
- Calculate material costs.
- Generate preliminary cost reports.

**Expected deliverable:** Initial material cost estimation module.

### Sprint 10 — Integration

**Planned**

- Integrate the developed modules.
- Add result visualization.
- Add manual verification and correction where needed.
- Perform end-to-end testing.

**Expected deliverable:** Initial integrated prototype.

---

## 10. Current Development Status

**Last updated:** 17 November 2026

| Area | Status |
|---|---|
| Project direction | Established |
| Repository | Exists |
| Python 3.12.16 | Available |
| `.venv` | Created |
| `.venv` dependencies | Not yet verified / currently empty |
| Dataset source | Identified |
| Dataset inspection | Pending |
| Dataset preprocessing | Initial module exists; behavior not yet verified |
| Annotation conversion | Not implemented / not verified |
| Deep Learning model | Not implemented |
| Training pipeline | Not implemented |
| Model evaluation | Not implemented |
| Geometry extraction | Not implemented |
| Quantity estimation | Not implemented |
| Cost estimation | Not implemented |

The project remains in **Sprint 01**.

---

## 11. Development Environment

The project is developed using Ubuntu on WSL.

### Python Environment

Target Python version:

```text
Python 3.12.16
```

Virtual environment:

```text
.venv/
```

Activate the environment:

```bash
source .venv/bin/activate
```

Verify the interpreter:

```bash
python --version
which python
```

Expected interpreter location:

```text
/path/to/floorplan-detection/.venv/bin/python
```

The virtual environment should be selected as the Python interpreter in VS Code.

### Dependency Management

Dependency installation and the creation of `requirements.txt` are still pending.

Do not assume that a library is installed until its import and version have been verified.

---

## 12. Development Principles

1. **Implementation First:** Documentation must reflect actual repository contents.
2. **Reproducibility:** Experiments should have documented configurations and results.
3. **Data Integrity:** Preserve image and annotation consistency.
4. **Modularity:** Separate data processing, model development, geometry, and estimation as implementation grows.
5. **Resource Awareness:** Choose models and training configurations according to available hardware.
6. **Scientific Evaluation:** Report performance only after running reproducible experiments.
7. **Traceability:** Record dataset sources, licenses, assumptions, and material coefficients.
8. **Incremental Development:** Complete and verify each sprint before claiming its deliverables.

---

## 13. Project Vision

The long-term vision is to transform 2D residential floor plans into structured building information that can support preliminary construction quantity estimation.

The intended progression is:

**Floor Plan Images → Building Element Recognition → Geometry Extraction → Quantity Takeoff → Material Estimation**

The immediate priority, however, is much smaller:

**Establish a reliable environment, inspect the actual dataset, and understand its annotations before implementing the first model.**