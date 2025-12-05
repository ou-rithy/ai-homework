# 🧠 Handwritten Digit Prediction API

## Project Overview

This project provides a machine learning model, trained on the $8 \times 8$ pixel Scikit-Learn Digits dataset, exposed as a high-performance **REST API** using **FastAPI**. The service's primary objective is to classify a flattened 64-pixel array input and return the predicted handwritten digit (0-9).

### 🎯 Objective

To classify raw $8 \times 8$ grayscale image data (flattened to 64 features) and predict the corresponding numerical digit (0-9).

-----

## 📂 Project Structure

```
digit-prediction-api/
├── main.py                  # FastAPI application code (your script, may be named main.py)
├── digits_model.pkl        # Pre-trained ML model (loaded by joblib)
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

-----

## 🛠️ Prerequisites

Before you begin, ensure you have the following software installed:

  * **Python 3.8+**
  * **pip** (Python package installer)
  * **Git** (for cloning and pushing to GitHub)

-----

## 🚀 Installation and Setup

### 1\. Clone the Repository

If you haven't already, clone the project from GitHub:

```bash
git clone https://github.com/ou-rithy/ai-homework.git
cd ai-homework
```

### 2\. Create Virtual Environment (Recommended)

Creating a virtual environment isolates your project dependencies:

```bash
python -m venv venv
# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3\. Install Dependencies

Install all required Python packages (FastAPI, Uvicorn, joblib, scikit-learn, etc.). Ensure your `requirements.txt` file is up-to-date and run:

```bash
pip install -r requirements.txt
```

-----

## ▶️ Running the Application

Use **Uvicorn**, the ASGI server, to run the FastAPI application. Assuming your main script is named `app.py`:

```bash
uvicorn app:app --reload
```

  * The API will start running at `http://127.0.0.1:8000`.
  * The `--reload` flag is useful during development, as it restarts the server whenever code changes are saved.

-----

## 🔍 API Endpoints and Usage

### Documentation

The API provides automatic interactive documentation via Swagger UI:

  * **Swagger UI (Interactive Docs):** `http://127.0.0.1:8000/docs`

### Prediction Endpoint

#### `POST /predict_digit`

This endpoint accepts the flattened pixel data and returns the model's prediction.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `pixels` | `List[float]` | A list of exactly 64 floating-point numbers representing the $8 \times 8$ grayscale image (flattened row by row). |

### Example Request (Predicting '0')

You can test the endpoint using the `curl` command line utility:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict_digit' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "pixels": [
    0.0, 0.0, 5.0, 13.0, 9.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 13.0, 15.0, 10.0, 15.0, 5.0, 0.0,
    0.0, 3.0, 15.0, 2.0, 0.0, 11.0, 8.0, 0.0,
    0.0, 4.0, 12.0, 0.0, 0.0, 8.0, 8.0, 0.0,
    0.0, 5.0, 8.0, 0.0, 0.0, 9.0, 8.0, 0.0,
    0.0, 4.0, 11.0, 0.0, 1.0, 12.0, 7.0, 0.0,
    0.0, 2.0, 14.0, 5.0, 10.0, 12.0, 0.0, 0.0,
    0.0, 0.0, 6.0, 13.0, 10.0, 0.0, 0.0, 0.0
  ]
}'
```

### Example Successful Response (200 OK)

```json
{
  "prediction": 0
}
```

-----

## 🎨 Visualization

To view the pixel data (the 64 numbers) as an actual image, you must use a visualization library like **Matplotlib** or **Pillow** to reshape the 1D list back into an $8 \times 8$ matrix.

```python
import numpy as np
import matplotlib.pyplot as plt

# ... pixel_data list goes here ...
image_matrix = np.array(pixel_data).reshape(8, 8)
plt.imshow(image_matrix, cmap=plt.cm.gray_r)
plt.show()
```
