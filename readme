#  Diabetic Retinopathy Severity Grading (Siamese ResNet50)

A deep learning pipeline for automated, patient-level screening of Diabetic Retinopathy (DR) using paired retinal fundus images from the Messidor-2 dataset. 

Unlike standard classifiers that evaluate single images, this project mimics a clinician's workflow by analyzing **both the left and right eyes simultaneously** using a Siamese network with shared weights.

##  Key Features
* **Dual-Input Siamese Architecture:** Utilizes a customized ResNet50 backbone to process left and right eye scans concurrently, merging features to output a holistic patient diagnosis.
* **Clinical-Grade Preprocessing:** Implements the **Ben Graham method** (Local Mean Subtraction via Gaussian blur) to normalize varied lighting conditions and highlight microaneurysms and hemorrhages.
* **Imbalance Handling:** Employs Weighted Cross-Entropy Loss to heavily penalize misclassifications of rare, severe disease stages (Proliferative DR).
* **Explainable AI (XAI):** Integrated **Grad-CAM** generates heatmaps over the retinal images, proving to clinicians exactly which lesions influenced the AI's decision.
* **Strict Evaluation:** Model performance is scored using **Quadratic Weighted Kappa (QWK)**, the gold-standard metric for ordinal medical diagnoses.

---

##  Tech Stack & Requirements
* **Framework:** PyTorch
* **Computer Vision:** OpenCV (`cv2`), PIL
* **Data Processing:** Pandas, NumPy
* **Evaluation & Explainability:** Scikit-Learn, `pytorch-grad-cam`

**Installation:**
```bash
pip install torch torchvision opencv-python pandas scikit-learn grad-cam matplotlib