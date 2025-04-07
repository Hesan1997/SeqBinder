# SeqBinder

Sequence-based deep learning model for predicting protein–ligand binding affinity using a dual-input fusion architecture.

[![Open in Colab](https://colab.research.google.com/assets/colab-btn.svg)](https://colab.research.google.com/github/Hesan1997/SeqBinder/blob/main/notebooks/model_training.ipynb)
> _Click the button above to launch an example Notebook in Google Colab._

---

## Table of Contents
1. [Overview](#overview)  
2. [Model Architecture](#model-architecture)  
    - [Protein Branch (LSTMBranch)](#protein-branch-lstmbranch)  
    - [Compound Branch (ConvBranch)](#compound-branch-convbranch)  
    - [Fusion Module (Trunk)](#fusion-module-trunk)  
4. [Contributing](#contributing)  
5. [License](#license)  
6. [Contact](#contact)

---

## Overview
SeqBinder is a sequence-based deep learning model designed to predict protein–ligand binding affinity. It uses a **dual-input fusion architecture** that incorporates distinct branches to process protein sequences and ligand SMILES, merging them in the final layers to produce a binding affinity score.

---

## Model Architecture
The SeqBinder model consists of three main components: a **Protein Branch**, a **Compound Branch**, and a **Fusion Module**.

### Protein Branch (LSTMBranch)
- **Input**: Protein sequence tokens.
- **Process**:  
  1. Token embedding of protein sequence.  
  2. Bidirectional LSTM to capture contextual information.  
  3. Global max pooling to reduce the LSTM output to a fixed-size feature vector.

### Compound Branch (ConvBranch)
- **Input**: Ligand SMILES tokens.
- **Process**:  
  1. Token embedding of SMILES string.  
  2. 1D convolutional layers with ReLU activation.  
  3. Adaptive max pooling to produce a fixed-size feature vector.

### Fusion Module (Trunk)
- **Input**: Concatenated feature vectors from the LSTMBranch and ConvBranch.
- **Process**:  
  1. Several fully connected (dense) layers with ReLU activation and dropout.  
  2. Outputs a single scalar value (predicted binding affinity).

---

## Contributing
Contributions are welcome! Please open issues or submit pull requests for bug fixes, improvements, or new features.

---

## License
This project is licensed under the [Apache License, Version 2.0](LICENSE). Feel free to use and modify the code as permitted by the license.

---

## Contact
Created by **Hesan Hashemi**.  
If you have any questions or suggestions, feel free to reach out at [hesan.hashemi.edu@gmail.com](mailto:hesan.hashemi.edu@gmail.com).
