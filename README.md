# Causal Face Attribute Editing with SAM and Stable Diffusion

## Project Overview
This project explores a human-in-the-loop image editing pipeline for facial attribute modification using a combination of:

- CelebA dataset for face attribute learning
- Segment Anything Model (SAM) for interactive segmentation
- Stable Diffusion inpainting for realistic image editing
- Latent-space causal probing to identify attribute-relevant semantic directions
- A custom perturbed diffusion pipeline that injects a learned latent direction during inpainting

The final goal is to edit or reinforce selected facial attributes such as hair style, smile, necklace, or other face-related features while keeping the rest of the image realistic and visually consistent.

The project is implemented as a notebook-based workflow in the workspace, with the main logic stored in `final.ipynb` and a pretrained SAM checkpoint `sam_vit_b_01ec64.pth` used for mask generation.

---

## Motivation
Traditional image editing methods often modify the entire image or require careful manual segmentation. In this project, the workflow aims to combine:

1. Semantic understanding of face attributes from the CelebA dataset
2. Precise instance segmentation using SAM
3. Latent manipulation inside the diffusion space to guide edits toward a chosen attribute
4. Realistic output generation through text-guided inpainting

This creates a practical attribute editing framework that can be used for tasks like:

- changing hairstyle
- modifying facial expression
- adding accessories
- enhancing or suppressing attribute-specific visual cues

---

## Core Idea
The project follows a multi-stage pipeline:

1. Load a subset of face images from the CelebA dataset.
2. Encode these images into latent representations using a VAE.
3. Train a linear probe on the latent representation to discover dimensions strongly associated with a target attribute.
4. Extract a top-k latent direction that represents the chosen attribute.
5. Use SAM to segment the editable region interactively.
6. Feed the original image and segmentation mask into a Stable Diffusion inpainting pipeline.
7. Inject a latent perturbation during denoising to push the output toward the desired attribute.

This makes the editing process more attribute-aware than a standard text-only inpainting approach.

---

## Dataset
The project uses the CelebA dataset, a large-scale face attribute dataset with labeled binary attributes.

Expected dataset structure:

- `data/list_attr_celeba.csv`
- `data/img_align_celeba/img_align_celeba/`

The CSV file contains attribute labels for each image. The notebook converts the original CelebA attribute coding from `-1/1` to binary `0/1` using the standard transformation:

- `-1 -> 0`
- `1 -> 1`

Examples of attributes used in the notebook include:

- `Wavy_Hair`
- `Smiling`
- `Wearing_Necklace`
- `Black_Hair`

The dataset is loaded in a small custom `CelebASmall` PyTorch dataset class that selects a subset of images (initially 500) and resizes every image to 128x128 for training and exploration.

---

## Model Components

### 1. Segment Anything Model (SAM)
The project uses the `segment-anything` package and the SAM ViT-B checkpoint:

- `sam_vit_b_01ec64.pth`

The notebook contains functions to:

- create a SAM predictor for a selected image
- accept a point click in the image
- generate a mask covering the selected object/region
- optionally create a brush-based mask via mouse interaction

This is useful for defining the editable area in a face image, such as a hairstyle region or accessory region.

### 2. Stable Diffusion Inpainting
The notebook uses the Hugging Face `diffusers` pipeline:

- `StableDiffusionInpaintPipeline`
- base model: `runwayml/stable-diffusion-inpainting`

This helps generate realistic image edits by filling a user-defined masked area according to a text prompt.

### 3. Variational Autoencoder (VAE)
A pretrained `AutoencoderKL` from Stable Diffusion is used to encode the face images into latent space. The code flattens the latent representation and stores it for attribute analysis.

### 4. Linear Causal Probe
A `LinearRegression` model is trained to predict an attribute from latent vectors. After fitting, the absolute value of the learned coefficients is used to identify the dimensions most relevant to the target attribute.

This approximate “causal probe” is used as a latent direction representing the relationship between the latent space and a chosen attribute.

---

## Project Workflow

### Step 1: Install dependencies
The notebook first installs the required packages:

- `torch`
- `torchvision`
- `pandas`
- `pillow`
- `opencv-python`
- `matplotlib`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `segment-anything`

### Step 2: Dataset preparation
A custom dataset class loads images and attribute labels from CelebA. Each sample returns:

- image tensor
- attribute dictionary

The project resizes input images to 128x128 for general exploration and 512x512 for VAE encoding in later stages.

### Step 3: Interactive SAM masking
The interactive functions allow the user to click the image and generate a mask around the region of interest.

Two interaction styles are available:

- point-based mask generation using SAM
- brush-based mask drawing using OpenCV

The resulting mask is converted to a binary image with white foreground and black background.

### Step 4: Create a paired image/mask for inpainting
After the mask is created:

- the original image is converted to a PIL image
- the binary mask is converted to grayscale
- the pair is passed to the inpainting pipeline

### Step 5: Inpainting using a prompt
The model can generate attribute-specific edits using prompts such as:

- "add tiara on the head"
- "add a big smile"
- "change the hair color"
- "add a necklace on the neck of the woman"

Then the image is reconstructed with the selected region inpainted according to the text prompt.

### Step 6: Latent perturbation for controlled semantic editing
The project also defines a custom `CustomPerturbedPipeline` that extends a standard Stable Diffusion inpainting pipeline. It adds a `perturbation_function` hook that can modify the latent representation during the diffusion process.

The notebook defines a custom perturbation function:

- `my_perturbation(latents, k=30, alpha=0.8)`

This function:

- flattens the latent tensor
- selects the top latent attributes discovered by the linear probe
- adds a scaled latent direction to the selected dimensions
- reshapes the output back into the diffusion latent layout

This means attribute edits can be guided by learned directions rather than only by text prompts.

---

## Custom Pipeline Design
The notebook contains a custom pipeline class named `CustomPerturbedPipeline`, which subclasses `StableDiffusionInpaintPipeline`.

It overrides the pipeline call and injects the following logic:

```python
if perturbation_function is not None:
    print("Applying perturbation to latents...")
    masked_image_latents = perturbation_function(masked_image_latents)
```

This is the key mechanism that introduces an attribute-aware perturbation into the latent flow before the denoising process continues.

The custom pipeline is then instantiated as:

```python
new_pipe = CustomPerturbedPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    safety_checker=pipe.safety_checker,
    feature_extractor=pipe.feature_extractor,
    image_encoder=None
)
```

This allows the original diffusion model to be reused while adding attribute-driven latent edits.

---

## Example of Attribute Probing
The attribute analysis part performs the following core steps:

1. Load a face image batch
2. Encode each image into latent space with the VAE
3. Flatten latent tensors
4. Build a target label vector `y` based on a selected attribute, such as `Wavy_Hair`
5. Fit a linear regression model to predict the attribute from latent features
6. Sort the coefficients by magnitude
7. Keep the strongest `k` latent dimensions as the most relevant subspace for editing

The notebook uses a function called:

```python
def train_causal_probe(k=60):
    probe = LinearRegression()
    probe.fit(X, y)
    importance = np.abs(probe.coef_)
    S_indices = np.argsort(importance)[-k:]
    return probe, S_indices, X.shape[1]
```

This returns the subspace of latent features most responsible for the attribute signal.

---

## Expected Visual Outcome
The project produces side-by-side visual comparisons of:

- the original image
- segmentation mask
- edited output image

The outputs can show, for example:

- hair color or texture edits
- smile enhancement
- accessory insertion
- facial appearance changes aligned with the selected attribute

The edited results are typically displayed using Matplotlib with a 1x3 grid:

- Original
- Mask
- Inpainted Output

---

## How to Run the Project

### Prerequisites
You need:

- Python 3.9+
- CUDA-enabled NVIDIA GPU strongly recommended
- Internet access for downloading model weights
- sufficient VRAM for Stable Diffusion and SAM

### Install dependencies
Run the following in the workspace:

```bash
pip install torch torchvision pandas pillow opencv-python matplotlib
pip install diffusers transformers accelerate safetensors
pip install segment-anything
```

### Download the models
You must have:

- `sam_vit_b_01ec64.pth` in the project root
- access to the Stable Diffusion inpainting model `runwayml/stable-diffusion-inpainting`

### Prepare the dataset
Place the CelebA files in the following structure:

```text
Final project/
├── final.ipynb
├── README.md
├── sam_vit_b_01ec64.pth
├── data/
│   ├── list_attr_celeba.csv
│   └── img_align_celeba/
│       └── img_align_celeba/
```

### Run the notebook
Open `final.ipynb` in VS Code/Jupyter and execute the cells in order.

The typical flow is:

1. install dependencies
2. load the dataset
3. select an image
4. run SAM mask generation
5. run inpainting with a prompt
6. optionally apply the perturbation pipeline
7. inspect the output

---

## Important Notes

### GPU Requirement
The pipeline was built and tested with CUDA. Some cells call `.to("cuda")`, so the project is intended for a GPU-enabled environment. Without a GPU, execution may be slow or fail depending on your hardware and driver setup.

### Model Size and Resource Use
This project loads several heavy models:

- SAM ViT-B
- Stable Diffusion inpainting pipeline
- VAE encoder
- CLIP/tokenizer components

This can consume a significant amount of GPU memory, especially with larger resolutions or multiple runs.

### Manual Interaction
Some of the masks are generated interactively using OpenCV and mouse callbacks. This means the user must click or draw in a window to define the region being edited.

### Prompt Sensitivity
Stable Diffusion results depend strongly on the prompt wording, mask quality, and edit strength. Good prompts plus a clean mask lead to much better outputs.

---

## Limitations
Although the project is effective for concept demonstration, it has important limitations:

- mask selection is largely manual and interactive
- prompt-based edits can be unpredictable
- latent perturbation quality depends on attribute direction quality
- the method may not generalize perfectly across all face types or poses
- larger or more consistent editing results may require stronger model tuning and better dataset preparation

---

## Possible Extensions
This project could be extended with:

- automatic face landmark-based masking instead of manual intervention
- more robust attribute-specific latent editing using stronger causal methods
- multi-attribute editing in a single pipeline
- comparison against unperturbed baseline generation
- quantitative evaluation using classifier scores or user studies
- generation of before/after galleries for various attributes

---

## Summary
This project demonstrates a practical pipeline for attribute-aware face editing by combining:

- CelebA-based attribute learning
- SAM segmentation for local editing regions
- Stable Diffusion inpainting for realistic reconstruction
- latent-space probing to guide edits toward meaningful attribute directions

It is an experimental but visually impressive approach to controllable image editing and can serve as a foundation for further work in semantic and causal face editing.

---

## Repository Contents
- `final.ipynb` — full project workflow and experiments
- `README.md` — project documentation
- `sam_vit_b_01ec64.pth` — pretrained SAM checkpoint
- `data/` — CelebA dataset directory (expected to be added locally)

---

## Final Note
This notebook is best understood as a research/prototype implementation rather than a production-ready application. It is designed to showcase an end-to-end pipeline for controllable face attribute manipulation using modern computer vision and generative modeling techniques.
