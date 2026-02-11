# PyTorch
This repository contains Jupyter notebooks created for learning PyTorch.
# 📂 Notebooks Guide

## 🔹 Fundamentals

### 1. `1.tensors-in-pytorch.ipynb`
Introduction to tensors
- Creating tensors
- Indexing & slicing
- Reshaping
- Broadcasting
- NumPy ↔ Torch conversion
- CPU vs GPU

---

### 2. `2.pytorch_autograd.ipynb`
Automatic differentiation
- requires_grad
- backward()
- gradient calculation
- simple linear regression example

---

### 3. `3.pytorch_training_pipeline.ipynb`
Basic training loop from scratch
- forward pass
- loss
- backward pass
- optimizer step
- manual training process

---

### 4. `4_pytorch_nn_module.ipynb`
Using `nn.Module`
- custom model class
- layers
- forward() method
- clean model structure

---

### 5. `5_pytorch_training_pipeline_using_nn_module.ipynb`
Full training pipeline with nn.Module
- loss functions
- optimizers
- training loop
- evaluation

---

### 6. `6_dataset_and_dataloader.ipynb`
Dataset handling
- custom Dataset class
- DataLoader
- batching
- shuffling
- efficient loading

---

### 7. `7_pytorch_training_pipeline_using_dataset_and_dataloader.ipynb`
Complete pipeline using Dataset + DataLoader
- clean production-style code
- scalable training

---

## 🔹 Computer Vision (FashionMNIST)

### 8. `8_ann_fashion_mnist_pytorch.ipynb`
ANN model (CPU)
- fully connected network
- baseline accuracy

---

### 9. `9_ann_fashion_mnist_pytorch_gpu.ipynb`
ANN model with GPU
- CUDA training
- faster execution

---

### 10. `10_ann_fashion_mnist_pytorch_gpu_optimized.ipynb`
GPU optimized ANN
- performance improvements
- cleaner training
- faster convergence

---

### 11. `11_cnn_fashion_mnist_pytorch_gpu.ipynb`
CNN implementation
- Conv2D
- MaxPool
- Flatten
- better accuracy than ANN

---

### 12. `12_cnn_fashion_mnist_pytorch_gpu_optuna.ipynb`
Hyperparameter tuning with Optuna
- automated search
- best params selection
- improved performance

---

### 13. `13_cnn_fashion_mnist_vgg16.ipynb`
VGG16 style CNN
- deeper architecture
- transfer learning style design
- higher accuracy

---

## 🔹 NLP / Sequence Models

### 14. `14_pytorch_rnn_based_qa_system.ipynb`
RNN based Question Answering system
- tokenization
- embeddings
- RNN model
- sequence learning

---

### 15. `15_pytorch_lstm_next_word_prediction.ipynb`
LSTM next word prediction
- language modeling
- text generation
- sequential prediction
