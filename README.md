
[![Portfolio](https://img.shields.io/badge/portfolio-active-green)](https://github.com/v0lia)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![ML](https://img.shields.io/badge/ML-Deep%20Learning-blueviolet)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

  
# 🧠 ML-Engine
Modular deep learning engine with logging, TensorBoard visualization, checkpointing and pluggable CNN architectures.

## 🧩 Overview

This repository provides a modular engine for building and experimenting with deep learning pipelines.  
It contains plug-and-play components, including:
- dataset preparation tools
- several convolutional neural network architectures (e.g., LeNet, EfficientNet-B0)
- training loop with customizable loss functions and optimizers
- evaluation utilities
- checkpointing functionality
- visualization through TensorBoard (during both training & evaluation)
- logging system across the entire pipeline
- reproducibility tools (seed setup)

Current reference pipeline: training convolutional neural networks on the Fashion-MNIST dataset.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** PyTorch
- **Dataset:** Fashion-MNIST
- **Tools:** torch, torchvision, TensorBoard

## [📊 Visual Outputs](#visuals)

## 🚀 How to Use

### Project Structure
```
ML-Engine/
├── checkpoints/
├── config/
├── datasets/
├── results/
├── src/
│   ├── core/
│   ├── data/
│   ├── loss/
│   ├── models/
│   ├── optimizer/
│   └── utils/
└── main.py
```

### Install  
```bash
git clone https://github.com/v0lia/ML-Engine.git  
cd ML-Engine  
pip install -r requirements.txt
```

### Command-Line Interface
Always run `main.py` as an entry-point of this engine.  
See all available CLI arguments:  
```bash
python main.py --help
```

### Train
```bash
python main.py train --model <model_file_path>
python main.py eval --checkpoint <checkpoint_file_path>
```

### Evaluate
```bash
python main.py eval --checkpoint <checkpoint_file_path>
python main.py eval --auto-latest
```

### Models & Checkpoints Locations
- Models: `src/models/`
- Checkpoints
    - final: `checkpoints/`
    - intermediate: `results/<run_name>/checkpoints/`

## ⚙️ Configuration
The engine is configured through the `config/config.yaml` file.  
The YAML config defines global settings (seed, batch size, loss function, optimizer, etc.), while the CLI (Command-Line Interface: `main.py`) determines the specific action to execute (train, eval).  
Any run can also use a custom configuration file by passing its path via command-line arguments:  
```bash
python main.py train --model LeNet --config config/my_config.yaml
```  

## 📁 Results

### Results Structure
- Tensorboard: `results/<run_name>/tensorboard`
- Checkpoints (intermediate): `results/<run_name>/checkpoints`
- Checkpoints (final): `checkpoints/`
- Logs: `results/<run_name>/logs`

### Visualizing Results
```bash
cd ML-Engine
```
- Visualizing all final checkpoints:
```bash
tensorboard --logdir checkpoints
```
- Visualizing intermediate results of all runs:
```bash
tensorboard --logdir results
```
- Visualizing intermediate results of a specific run:
```bash
tensorboard --logdir results/<run_name>/tensorboard
```
Then open http://localhost:6006 in your browser.

## ✨ Lessons Learned
During this project, I gained hands-on experience with:
- **Designing modular acrhitecture** for deep learning pipelines
- Implementing and comparing multiple CNN architectures (LeNet, MLP, ResNet18)
- Managing data sets and preprocessing workflows
- Logging, checkpointing, and visualizing experiments with TensorBoard
- Structuring a Python project for scalability and maintainability
- Applying best practices in code organization, testing, and reproducibility

## 📡 Contacts
- GitHub: [v0lia](https://github.com/v0lia)
- Email: <vitvolia@gmail.com>

## 📊 Visuals
<p align="center">
  <img src=".gitimages/sample_grid.png" alt="Data sample grid"/><br>
  <em>Data sample grid</em>
</p>

<p align="center">
  <img src=".gitimages/data_embedding.PNG" alt="Data embedding" width="400"/><br>
  <em>Data embedding</em>
</p>

<p align="center">
  <img src=".gitimages/epoch_loss_curves.PNG" alt="Epoch loss curves" width="400"/><br>
  <em>Data Epoch loss curves</em>
</p>

<p align="center">
  <img src=".gitimages/epoch_accuracy_curves.PNG" alt="Epoch accuracy curves" width="400"/><br>
  <em>Epoch accuracy curves</em>
</p>

<p align="center">
  <img src=".gitimages/pr-curves.PNG" alt="Precision-Recall curves" width="400"/><br>
  <em>Precision-Recall curves</em>
</p>

<p align="center">
  <img src=".gitimages/model_graph.PNG" alt="ResNet18 graph" width="150"/><br>
  <em>ResNet18 graph</em>
</p>

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.


## Citation
Vitalii Volia. ML-Engine (2025). GitHub repository: https://github.com/v0lia/ML-Engine
