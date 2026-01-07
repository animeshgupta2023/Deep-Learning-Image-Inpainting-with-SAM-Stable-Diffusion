
---

# Deep Learning Image Inpainting with SAM & Stable Diffusion

## Project Description

This project was developed for the **ADL (Advanced Deep Learning)** course during the 7th semester. It implements an image editing pipeline that combines **Segment Anything Model (SAM)** for precise mask generation with **Stable Diffusion** for text-guided image inpainting.

The workflow allows users to interactively select an object or region in an image (specifically tested on the CelebA face dataset) and modify it using natural language prompts.

## Features

* **Dataset Integration:** Custom data loader for the **CelebA** dataset with attribute parsing.
* **Interactive Segmentation:** Utilizes Meta's **Segment Anything Model (SAM)** to generate binary masks via point-click interactions using OpenCV.
* **Manual Refinement:** Includes a brush tool for manual mask creation and refinement using OpenCV.
* **Generative Inpainting:** Leverages the Hugging Face `diffusers` library to perform text-guided inpainting on the masked regions (modifying specific facial attributes based on prompts).

## Tech Stack

* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **Key Libraries:**
* `segment-anything` (Model segmentation)
* `diffusers` (Stable Diffusion pipeline)
* `transformers` & `accelerate` (Hugging Face utilities)
* `opencv-python` (Image processing & UI)
* `pandas` (Data manipulation)



## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

```


2. **Install dependencies:**
```bash
pip install torch torchvision pandas pillow opencv-python matplotlib
pip install diffusers transformers accelerate safetensors
pip install segment-anything

```


3. **Download Model Checkpoints:**
* Download the SAM checkpoint (e.g., `sam_vit_b_01ec64.pth`) and place it in the project root.
* Ensure you have access to the CelebA dataset.



## Usage

1. **Configure Paths:**
Update the `img_dir` and `csv_path` variables in the notebook to point to your local CelebA dataset folder.
2. **Run the Notebook:**
Open `test.ipynb` in Jupyter Notebook or Google Colab.
3. **Workflow Steps:**
* **Load Data:** The notebook loads images from the CelebA dataset.
* **Create Mask:** Run the interactive SAM cell. Click on the part of the image you want to edit to generate a mask. Alternatively, use the brush tool cell.
* **Inpaint:** (Ensure your pipeline code is active) Provide a text prompt (e.g., "blonde hair", "wearing glasses") to generate the edited version of the image using the created mask.



## Project Structure

```
├── final.ipynb              # Main project notebook
├── README.md               # Project documentation
├── data/                   # Dataset directory (not included in repo)
└── sam_vit_b_01ec64.pth    # SAM Model checkpoint (download separately)

```

## Acknowledgments

* Course: 7th Semester ADL (Advanced Deep Learning)
* **CelebA Dataset:** For facial attributes and images.
* **Meta AI:** For the Segment Anything Model (SAM).
* **Hugging Face:** For the Diffusers library.
