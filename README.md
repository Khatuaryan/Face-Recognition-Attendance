# Face Recognition Attendance System

A real-time attendance management system powered by **OpenCV** and **Machine Learning (KNN)**. This project captures face data, trains a model on the fly using K-Nearest Neighbors, and logs attendance into daily CSV files which can be viewed through a Streamlit web dashboard.

## 🚀 Features

* **Real-time Face Detection:** Uses Haar Cascade classifiers for fast and accurate face detection.
* **Automated Data Collection:** Capture 100 samples per person to ensure high accuracy.
* **KNN Classification:** Utilizes the K-Nearest Neighbors algorithm to recognize faces based on stored data.
* **Daily Attendance Logs:** Automatically creates a new CSV file for each day (e.g., `Attendance_30-12-2025.csv`).
* **Interactive Dashboard:** A Streamlit-based web interface to monitor attendance records in real-time.
* **Visual Feedback:** Custom UI overlay with background images and bounding boxes.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV
* **Machine Learning:** Scikit-Learn (KNeighborsClassifier)
* **Data Handling:** Pandas, Pickle, NumPy, CSV
* **Web Framework:** Streamlit

---

## 📁 Project Structure

```text
FaceRecognition/
├── Data/
│   ├── haarcascade_frontalface_default.xml  # Face detection model
│   ├── names.pkl                           # Stored labels/names
│   └── faces_data.pkl                      # Stored face feature vectors
├── Attendance/
│   └── Attendance_DD-MM-YYYY.csv           # Daily logs (auto-generated)
├── add_data.py                             # Script to register new faces
├── test.py                                 # Main recognition & logging script
├── webapp.py                               # Streamlit dashboard script
└── background.jpg                          # UI background image

```

---

## ⚙️ Installation & Setup

1. **Clone the repository** (or save the files in a folder).
2. **Install Dependencies:**
```bash
pip install opencv-python scikit-learn pandas streamlit numpy streamlit-autorefresh

```


3. **Prepare Directories:**
Ensure you have a folder named `Data` and a folder named `Attendance` in your project root.
4. **Add a Background:**
Place an image named `background.jpg` in the root directory for the UI to display correctly.

---

## 🚦 How to Use

### 1. Register a New Face

Run `add_data.py` to capture face samples for a new user.

```bash
python add_data.py

```

* Enter the name when prompted.
* The camera will open and automatically capture 100 frames of your face.
* Data is saved in `Data/names.pkl` and `Data/faces_data.pkl`.

### 2. Run Attendance System

Run `test.py` to start the recognition process.

```bash
python test.py

```

* **Key Bindings:**
* Press **'o'**: To log the detected person's attendance into the CSV file.
* Press **'c'**: To close the application.



### 3. View the Dashboard

Run `webapp.py` to see the attendance logs in your browser.

```bash
streamlit run webapp.py

```

* The dashboard auto-refreshes every 2 seconds to show new entries.

---

## ⚠️ Important Notes

* **Hardcoded Paths:** In `webapp.py`, ensure the path to the CSV file matches your local environment.
* **Lighting:** For best results, ensure the face is well-lit during both data collection and recognition.
* **Resolution:** The `test.py` script is configured for a specific UI size (). Adjust the `desired_width` and `desired_height` variables if your monitor resolution differs.

---
