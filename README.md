# 🪙 Thai Coin Detection AI

A real-time object detection web application built with **YOLOv5** and **Flask**. This project is designed to detect Thai coins (10, 5, 2, 1, and 0.50 Baht) using a webcam and automatically calculate the total sum of the detected coins.

---

## 🛠️ System Architecture
```text
[ Web Browser ]  <-- (MJPEG Stream) -->  [ Flask Web Server (app.py) ]
                                                |
                                                v
[ Webcam ]       <-- (Read Frames)  -->  [ YOLOv5 Inference Engine ]
                                                |
                                                v
                                         [ PyTorch Model (.pt) ]
```
* **Frontend:** A simple HTML UI displaying the video stream (`templates/index.html`).
* **Backend:** Flask serving the frontend and streaming the processed video frames (`app.py`).
* **AI Model:** YOLOv5 model trained on custom Thai coin datasets to draw bounding boxes and calculate the total currency sum.

---

## 🚀 Features
- **Real-Time Detection:** Uses YOLOv5 for fast and accurate coin detection via webcam.
- **Auto Sum Calculation:** Automatically identifies the coin classes and calculates the total amount (e.g., two 10 Baht + one 5 Baht = 25 Baht).
- **Data Augmentation:** Built-in Python script (`augment_dataset.py`) to easily expand your dataset size by generating variations (flipped, darker, brighter) of your training images.
- **Web Interface:** Clean and simple web UI that streams the AI's output.

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Thai-coin-detection.git
   cd Thai-coin-detection/coin-detection-main
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Use

### 1. Run the Web Application
Start the Flask server to open the webcam and detect coins:
```bash
python app.py
```
*Then, open your web browser and navigate to: `http://localhost:5000`*

### 2. Augment the Dataset (Optional)
If you want to train the model with more data, you can expand your existing dataset by running the augmentation script:
```bash
python augment_dataset.py
```
This script will read images and labels from the `coin_dataset` folder and generate new variations to improve model accuracy.

---

## 📂 Project Structure

```
coin-detection-main/
│
├── app.py                   # Main Flask application and YOLOv5 wrapper
├── augment_dataset.py       # Script for multiplying the dataset (Data Augmentation)
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── README.md                # Project documentation
│
├── templates/               # HTML templates for Flask
│   └── index.html           # Web UI
│
├── coin_dataset/            # Your training images and YOLO labels
├── trained/                 # Historical trained model weights
│
└── yolov5/                  # YOLOv5 source code and engine
    └── weights/             # Contains the custom trained weights (e.g., coin_v1-9_last.pt)
```

---

## 📝 License
This project is licensed under the [MIT License](LICENSE).
