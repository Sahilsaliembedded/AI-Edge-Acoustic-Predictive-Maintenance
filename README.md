# 🚀 AI-Edge Acoustic Predictive Maintenance Node

## 📌 About the Project

The AI-Edge Acoustic Predictive Maintenance Node is a prototype designed to explore motor condition monitoring using acoustic signals and machine learning.

The system captures motor sound using a microphone and an ESP32-based setup. The recorded audio is transferred to a laptop, where Python-based signal processing and a trained machine learning model analyze the sound and classify it as normal or abnormal.

This project explores an affordable approach to identifying unusual motor behavior through sound analysis.

## 🎯 Problem Statement

Motors can develop abnormal operating conditions that may lead to equipment failure and unexpected downtime. Regular manual inspection may not identify every change as it occurs.

Our project explores sound-based motor condition monitoring as a potential way to identify unusual operating conditions.

## 💡 Proposed Solution

We developed a prototype that combines embedded hardware with laptop-based machine learning.

The ESP32-based setup is used to acquire motor audio, while the laptop performs audio processing, feature extraction, machine learning inference, and result visualization.

### 🔄 System Workflow

1. A DC motor operates under different conditions.
2. A microphone captures the motor sound.
3. The ESP32-based setup acquires the audio data.
4. The audio data is transferred to a laptop.
5. Python processes the audio and extracts relevant features.
6. A trained machine learning model predicts whether the recorded sound is normal or abnormal.
7. The prediction results can be viewed through a laptop-based dashboard, depending on the implemented version.

## 🛠️ Technologies Used

* **Microcontroller:** ESP32
* **Programming Languages:** C/C++ for ESP32 and Python for audio processing and machine learning
* **Signal Processing:** Audio preprocessing and feature extraction
* **Machine Learning:** Model training, evaluation, and prediction
* **Dashboard:** Streamlit, if used in the final working version
* **Hardware:** DC motor and microphone module

## ⚙️ Key Features

* Motor sound acquisition
* Audio recording and preprocessing
* Feature extraction from acoustic signals
* Machine learning-based sound classification on a laptop
* Normal and abnormal condition prediction
* Dashboard-based result visualization, if enabled in the final version

## 🧠 Current Implementation

The current prototype uses a **laptop for machine learning inference**. The trained model has not yet been deployed directly onto the ESP32.

Python handles feature extraction, model prediction, and dashboard processing. The ESP32-based hardware setup is used for audio acquisition and communication with the laptop.

This approach allowed us to develop and test the machine learning workflow before attempting deployment on resource-constrained embedded hardware.

## 🧪 Hackathon Experience

Developing this project gave us practical experience combining embedded hardware, audio signal processing, and machine learning.

During development, we worked on motor sound recording, ESP32 communication, audio preprocessing, feature extraction, model training, and prediction testing.

We encountered challenges involving microphone readings, audio noise, hardware-software integration, and prediction reliability. Troubleshooting these issues helped us understand the practical difficulties of building a complete hardware-based AI prototype.

The experience also taught us the importance of testing individual components, evaluating model performance, and clearly identifying the limitations of a prototype.

## 📚 What We Learned

* Working with the ESP32 microcontroller
* Acquiring and processing audio signals
* Extracting features from motor sound
* Training and evaluating a machine learning model
* Integrating embedded hardware with laptop-based software
* Debugging hardware and software issues
* Collaborating as a team during a hackathon

## 🚧 Current Limitations

* Machine learning inference currently runs on a laptop rather than directly on the ESP32.
* Prediction reliability depends on the quality and diversity of the audio dataset.
* Background noise and microphone placement may affect predictions.
* More testing with different motors and operating conditions is required.
* The current prototype has not been validated for industrial predictive maintenance.

## 🔮 Future Improvements

* Improve the diversity and quality of the motor sound dataset.
* Improve model accuracy and reliability through further testing.
* Explore model conversion and deployment on suitable embedded hardware.
* Investigate TinyML techniques to perform inference directly on the ESP32, subject to memory and processing constraints.
* Add real-time alerts using LEDs or a buzzer.
* Improve the dashboard with live monitoring and historical results.
* Validate the system under a wider range of motor operating conditions.

## 👥 Team Members

* Sahil Sali
* Akanksha Shirgire
* Rani Sathe

## 🏁 Conclusion

This project represents our initial step toward sound-based motor condition monitoring using embedded hardware and machine learning.

Although the current model runs on a laptop, the prototype helped us understand the complete workflow, from audio acquisition and signal processing to model training and prediction.

Our next goal is to improve the model and explore deployment on embedded hardware so that motor condition predictions can eventually be performed directly on a suitable microcontroller.

**Project Status:** Prototype developed; machine learning inference runs on a laptop. Direct TinyML deployment on the ESP32 remains future work.

