# Project Introduction

RMM-TCVAE (Robust Multimodal Transformer-based Conditional Variational Autoencoder) is a multimodal emotion recognition model designed to handle missing modality scenarios. This model improves emotion recognition performance under incomplete multimodal data through Transformer reconstruction networks and graph-based multimodal fusion networks.

# HIGHLIGHTS
- Addresses dynamic missing-modality challenges in multimodal emotion recognition
- Introduces Transformer-based conditional generative reconstruction mechanism
- Learns robust cross-modal representations under incomplete data conditions
- Achieves state-of-the-art performance across multiple benchmark datasets
- Demonstrates strong robustness under high missing-rate scenarios

## Project Structure

```
RMM-TCVAE/
├── data/              # Dataset directory
├── logs/              # Log directory
├── output/            # Output directory (model saving)
├── config.py          # Configuration file
├── dataset.py         # Dataset processing
├── rmm_tcvae.py       # Model implementation
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── predict.py         # Prediction script
├── requirements.txt   # Dependencies file
└── README.md          # Project documentation
```

## Environment Requirements

- Python 3.7+
- PyTorch 2.0.0+
- Transformers 4.28.1+
- NumPy 1.24.3+
- Pandas 2.0.3+
- scikit-learn 1.3.0+

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The project supports the following datasets:
- CMU-MOSI
- CMU-MOSEI
- IEMOCAP

### Data Preprocessing

1. **Text features**: Encoded using BERT model
2. **Audio features**: Can be extracted using OpenSMILE
3. **Visual features**: Can be extracted using facial expression recognition tools

### Data Format

Datasets should contain the following fields:
- `text`: Text content
- `audio`: Audio feature vector
- `visual`: Visual feature vector
- `label`: Emotion label
- `missing_mask`: Modality missing mask (optional)

## Training the Model

### Configuration Parameters

Modify parameters in `config.py`:
- `dataset`: Choose dataset (cmu_mosi, cmu_mosei, iemocap)
- `data_dir`: Dataset directory
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `alpha`: Reconstruction loss weight
- `beta`: KL divergence loss weight

### Run Training

```bash
python train.py
```

During training, the model will automatically save the model with the best validation loss to `output/model.pth`.

## Evaluate the Model

```bash
python evaluate.py
```

The evaluation script will load the trained model and calculate loss and accuracy on the test set.

## Prediction

```bash
python predict.py
```

The prediction script provides a simple interface for emotion prediction using the trained model.

### Example Usage

```python
from predict import Predictor

# Create predictor
predictor = Predictor()

# Predict emotion
text = "I'm so happy today!"
prediction = predictor.predict(text)
print(f"Predicted emotion: {prediction}")
```

## Model Architecture

The RMM-TCVAE model consists of the following components:

1. **Multimodal Encoders**:
   - Text encoder: Using BERT model
   - Audio and visual encoders: Using linear layers

2. **Conditional Variational Autoencoder**:
   - Encoder: Maps multimodal features to latent space
   - Reparameterization: Samples from latent distribution
   - Decoder: Uses Transformer to reconstruct missing modalities

3. **Graph-based Multimodal Fusion Network**:
   - Attention mechanism: Captures inter-modal dependencies
   - Modality weights: Adaptively adjusts contributions of each modality
   - Fusion layer: Fuses multimodal features into a unified representation

4. **Classifier**:
   - Performs emotion classification based on fused features

## Loss Function

The total loss consists of three parts:
1. **Classification loss**: Cross-entropy loss for emotion classification
2. **Reconstruction loss**: Mean squared error to ensure the quality of reconstructed modalities
3. **KL divergence loss**: Regularizes the latent distribution to be close to standard normal distribution

## Experimental Results

The model is evaluated on the following datasets:
- CMU-MOSI
- CMU-MOSEI
- IEMOCAP

Evaluation metrics include:
- Accuracy
- Weighted Accuracy

## Notes

1. **Computational Resources**: The model uses Transformer architecture, which requires significant memory and computational resources
2. **Data Preprocessing**: Proper preprocessing is needed for specific datasets in practical use
3. **Hyperparameter Tuning**: Different datasets may require different hyperparameter settings

## Future Work

1. Improve training efficiency and reduce model complexity
2. Explore lightweight reconstruction strategies
3. Extend to other multimodal tasks, such as emotion intensity prediction and cross-domain adaptation

## References

- Original Paper: RMM-TCVAE: Robust Multimodal Transformer-based Conditional Variational Autoencoder for Emotion Recognition with Missing Modalities


## Results

Below are key experimental results of our proposed RMM-TCVAE framework.

**Overall architecture of the proposed RMM-TCVAE:**  
![Framework](./Figs/Fig3.svg)

**t-SNE visualizations of the latent representations on the IEMOCAP (four-class) test set:**  
![t-SNE visualizations](./Figs/Fig8.svg)

**Confusion matrices of different models on the IEMOCAP:**  
![Confusion matrices](./Figs/Fig9.svg)

---

### Upon acceptance of our manuscript, we will release our code to this code repository at the earliest possible time. We greatly appreciate your attention and look forward to your feedback.
