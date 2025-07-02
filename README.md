# 🧠 Conditional Beta-VAE on MNIST

This project explores the use of **Conditional Beta-Variational Autoencoders (Beta-VAEs)** to generate digit images from the **MNIST dataset**, focusing on how different beta values affect the quality of generated samples and latent space disentanglement.

## 📝 Abstract

We experiment with different β (beta) values to balance the trade-off between reconstruction accuracy and latent space regularization. This project highlights how Conditional Beta-VAEs can be optimized for generating class-conditioned images and learning interpretable latent representations.

## 📚 Table of Contents

- [Abstract](#-abstract)
- [Dataset](#-dataset)
- [Model](#-model)
- [Training](#-training)
- [Visualizations](#-visualizations)
- [Results](#-results)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [License](#-license)

## 📊 Dataset

- **MNIST**: A benchmark dataset of handwritten digits (0–9).
- Input images are 28x28 grayscale, with 60,000 training and 10,000 test samples.

## 🔧 Model

### Conditional Beta-VAE
- Learns digit generation conditioned on class labels.
- β (beta) controls the weight of the KL-divergence in the ELBO loss.
- Higher β → more disentangled latent space, but worse reconstruction.

### Core Components
- **Encoder**: Maps image + label to latent space
- **Reparameterization**: Samples from latent distribution
- **Decoder**: Reconstructs image from latent vector and label
- **Loss**: `ELBO = MSE + β * KL-Divergence`

## 🚀 Training

- Trained Conditional Beta-VAE on MNIST with multiple β values: `[0.1, 0.5, 1.0, 2.0, 4.0]`
- Used best-performing β to retrain final model

```python
def train_model_conditional(beta, epochs):
    ...
```

## 📈 Visualizations

- **Generated Samples**: Visualize digits generated for each class
- **Latent Space (t-SNE & PCA)**: Plots show how different β values affect separation
- **Loss Curves**: Training loss vs. epochs

### Sample Function:
```python
visualize_samples(model, digit_label)
```

## ✅ Results

| Beta | Reconstruction Quality | Latent Separation |
|------|------------------------|-------------------|
| 0.1  | High                   | Low               |
| 1.0  | Balanced               | Good              |
| 4.0  | Poor                   | Excellent         |

> The best trade-off between reconstruction and interpretability was found at **β = 1.0**.

## 🧪 Installation

```bash
pip install torch torchvision matplotlib scikit-learn
```

## ▶️ How to Run

```bash
jupyter notebook "Using_the_Beta-VAE_Model_to_generate_digit_images_on_the_MNIST_Dataset.ipynb"
```

Proceed through the notebook step by step to:
- Load data
- Train model with different β values
- Visualize results
- Generate digits by condition

## 📁 Project Structure

```
.
├── Using_the_Beta-VAE_Model_to_generate_digit_images_on_the_MNIST_Dataset.ipynb
└── README.md
```

## 📜 License

MIT License – Feel free to reuse and modify this project with attribution.

