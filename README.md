# Toxic Comment Classifier

A deep learning-based toxic comment classification system using Bidirectional LSTM (BiLSTM) neural networks. This project detects multiple categories of toxicity in online comments including toxic, severe toxic, obscene, threat, insult, and identity hate.

##  DATASET

-https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip

## 🌟 Features

- **Multi-label Classification**: Detects 6 different types of toxicity simultaneously
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate

- **Deep Learning Model**: BiLSTM architecture for understanding context in both directions
- **FastAPI Backend**: RESTful API for making predictions
- **Gradio Web Interface**: User-friendly web UI for testing the model
- **Text Preprocessing**: Automated text cleaning and normalization

## 🏗️ Architecture

### Model Architecture
- **Embedding Layer**: 128-dimensional word embeddings (MAX_FEATURES: 200,000)
- **Bidirectional LSTM**: 128 units with return sequences
- **Global Max Pooling**: Extracts maximum features
- **Dropout Layer**: 0.3 dropout rate for regularization
- **Dense Layer**: 128 units with ReLU activation
- **Output Layer**: 6 units with sigmoid activation (multi-label classification)

### Tech Stack
- **Framework**: TensorFlow/Keras
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Data Processing**: Pandas, NumPy
- **Text Vectorization**: Keras TextVectorization

## 📁 Project Structure

```
Toxic_comment_classification/
│
├── main.py                 # FastAPI server with prediction endpoint
├── train.py               # Model training script
├── test.py                # Test script for model evaluation
├── preprocessing.py       # Text cleaning utilities
├── gradio_app.py          # Gradio web interface
│
├── model.h5               # Trained model (HDF5 format)
├── model.keras            # Trained model (Keras format)
├── vectorizer.keras       # Text vectorization layer
├── train (2).csv          # Training dataset
│
└── __pycache__/           # Python cache files
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pamurirajiv/Toxic_comment_classification.git
   cd Toxic_comment_classification
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install tensorflow pandas scikit-learn fastapi uvicorn gradio requests
   ```

## 📊 Training the Model

To train the model from scratch:

```bash
python train.py
```

**Training Details:**
- Uses 40% of the dataset for faster training
- 80/20 train-test split
- Sequence length: 200 tokens
- Max vocabulary: 200,000 words
- Model is saved as `model.h5` and `model.keras`
- Vectorizer is saved as `vectorizer.keras`

## 🔧 Usage

### 1. Running the FastAPI Server

Start the backend API server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

**API Endpoints:**
- `POST /predict` - Predict toxicity of a comment
  ```json
  {
    "comment": "Your comment here"
  }
  ```
- `GET /health` - Health check endpoint

### 2. Running the Gradio Interface

In a separate terminal, start the Gradio web interface:

```bash
python gradio_app.py
```

The interface will open in your browser at `http://127.0.0.1:7860`

### 3. Testing the Model

Run the test script to see sample predictions:

```bash
python test.py
```

### Example API Request

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"comment": "This is a sample comment"}
)

print(response.json())
```

**Response:**
```json
{
  "comment": "this is a sample comment",
  "prediction": {
    "toxic": 0.123,
    "severe_toxic": 0.045,
    "obscene": 0.078,
    "threat": 0.012,
    "insult": 0.089,
    "identity_hate": 0.034
  }
}
```

## 📈 Model Performance

The model uses the following metrics:
- **Precision**: Measures accuracy of positive predictions
- **Recall**: Measures coverage of actual positive cases
- **Binary Crossentropy**: Loss function for multi-label classification

## 🔄 Preprocessing

The `preprocessing.py` module includes:
- Removal of non-ASCII characters
- Lowercase conversion
- Whitespace normalization
- Special character handling

## 🛠️ Development

### Adding New Features

1. Modify the model architecture in `train.py`
2. Update the preprocessing pipeline in `preprocessing.py`
3. Retrain the model
4. Update the API endpoints in `main.py` if needed

### Hyperparameter Tuning

Key parameters to experiment with:
- `MAX_FEATURES`: Vocabulary size (currently 200,000)
- `SEQ_LENGTH`: Maximum sequence length (currently 200)
- `Embedding dimensions`: Currently 128
- `LSTM units`: Currently 128
- `Dropout rate`: Currently 0.3

## 📝 Dataset

The project uses a toxic comments dataset with labeled examples across 6 toxicity categories. The dataset is loaded from `train (2).csv`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- FastAPI for the web framework
- Gradio for the UI interface
- The open-source community for various tools and libraries

## 📧 Contact

For questions or feedback, please contact:
- GitHub: [@pamurirajiv](https://github.com/pamurirajiv)

---

**Note**: This model is designed for educational and research purposes. Use responsibly and be aware of potential biases in the training data.
