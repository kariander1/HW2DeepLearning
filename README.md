
# RNN Regularization by Zaremba et al. (2014)

This repository contains an implementation and replication of the "Recurrent Neural Network Regularization" by Zaremba et al. (2014). The implementation uses the Penn Tree Bank (PTB) dataset for training and evaluating the models.

## Overview
The codes evaluates performance on language modeling tasks when introducing dropout regularization for Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory networks (LSTMs) and Gated Recurrent Unit (GRUs).

## Features
- Loading and visualizing the Penn Tree Bank dataset.
- Creating train and test splits.
- Defining RNN models with w/o regularization.
- Logging the model in TensorBoard.
- Training and evaluating the model using perplexity.
- Evaluating the trained model by perplexity and predicting sentence completion.

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Training and Evaluating the Model
To train and evaluate the model, run:
```bash
python main.py --cuda --epochs 10
```
This will train the model on the PTB dataset and evaluate its performance using perplexity.

### Generating Text
To generate text using the trained model, run:
```bash
python generate.py --cuda --start_text "The quick brown fox"
```
This will generate a continuation of the given start text.

## Example Notebooks
- `example_notebook.ipynb`: Contains the implementation details, including dataset visualization, model training, evaluation, and text generation.

## References
- **Recurrent Neural Network Regularization by Zaremba et al. (2014)**:
  - Paper link: [Recurrent Neural Network Regularization (arxiv.org)](https://arxiv.org/abs/1409.2329)
- **GitHub Repositories**:
  - [ahmetumutdurmus/zaremba](https://github.com/ahmetumutdurmus/zaremba)
  - [rajwanraz/Recurrent-Neural-network-regularization](https://github.com/rajwanraz/Recurrent-Neural-network-regularization)
  - [idancz/RNN-Regularization](https://github.com/idancz/RNN-Regularization)
  - [davide-belli/nlp-RNN-pytorch](https://github.com/davide-belli/nlp-RNN-pytorch)

## Acknowledgements
This implementation is inspired by the works of many researchers and developers. Special thanks to the authors of the original paper and the contributors of the referenced GitHub repositories.

