### Table Tennis Stroke Classifier (Wrist-Mounted IMU)
This project implements a real-time system for classifying table tennis strokes using a wrist-mounted IMU (Inertial Measurement Unit) and machine learning. The system captures acceleration and gyroscope data via Bluetooth, trains a model offline, and enables live stroke prediction through a Python-based interface.


**Repository Contents**
- `combined_.csv`: Labeled sensor dataset containing accelerometer and gyroscope data across six stroke categories:
Forehand Topspin, Backhand Topspin, Forehand Chop, Backhand Chop, Random Motion, Rest.

- `train.py`: Python script to train a Random Forest classifier on the dataset. Outputs a stroke_classifier_model.pkl file and prints classification metrics.

- `scan_devices.py`: Utility script using the bleak library to scan for BLE-enabled devices such as the Arduino Nano 33 IoT and list their names and addresses.

- `real_time.py`: Main real-time inference pipeline. Connects to the BLE device, reads IMU data, buffers and preprocesses it, and predicts stroke type using the trained model.
  

**Requirements**
- Python 3.8+
- Libraries: pandas, scikit-learn, joblib, matplotlib, seaborn, bleak
- `pip install pandas scikit-learn joblib matplotlib seaborn bleak`


**Real-Time Prediction Setup**
- Upload the Arduino BLE firmware (e.g., ble.ino) to your Nano 33 IoT.
- Run `scan_devices.py` to identify your board’s MAC address.
- Train the model with `train.py` or use the provided .pkl.
- Run `real_time.py` for live predictions over BLE.


**Dataset Format**
Each row in combined_dataset.csv includes: `Timestamp`, `Accel_X`, `Accel_Y`, `Accel_Z`, `Gyro_X`, `Gyro_Y`, `Gyro_Z`, `Label`
