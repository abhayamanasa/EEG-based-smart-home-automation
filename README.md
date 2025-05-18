# EEG-based-smart-home-automation
Cerebral Smart Home: BCI-Based Smart Home Automation Using EEG Signals
Overview
The Cerebral Smart Home project is a groundbreaking assistive technology solution that leverages Brain-Computer Interface (BCI) technology to enable hands-free, voice-free control of smart home devices. Designed to empower individuals with mobility impairments, the system uses EEG signals captured via the BioAmp EXG Pill and eye-blink detection powered by MediaPipe to allow users to control virtual devices like lights, fans, and thermostats through mental focus and intentional blinks. The project integrates machine learning (Random Forest classifier), real-time signal processing, and secure IoT communication to deliver an accessible, efficient, and scalable smart home solution. This repository contains the source code, documentation, and resources for the system, which was developed over a 5-month period from January to May 2025.

#Key Features
Hands-Free Control: Operates devices using brainwaves and eye blinks, eliminating the need for physical or vocal input.
Accessibility: Tailored for users with mobility or speech impairments, ensuring inclusivity.
Real-Time Responsiveness: Executes commands in under 1 second, with an average latency of 0.8 seconds.
Energy Efficiency: Consumes 15% less power than expected (averaging 2 watts), due to optimized algorithms.
Secure Communication: Uses encrypted MQTT protocols (TLS) for safe data transmission.
Personalized Interaction: Adapts to individual brainwave patterns with a 25-minute calibration process.
Scalability: Modular design supports future device integration and enhancements.
High Accuracy: Achieves 95% mental state classification accuracy and 96% eye-blink detection accuracy.

#Project Details
Authors: Dasari Abhaya Manasa, Nelluru Kusuma, G. Santhoshi
Published In: TIJER - International Research Journal, Volume 12, Issue 5, May 2025 (Impact Factor: 8.57)
Paper ID: TUER2505030
Registration ID: 157766
Published URL: https://tijer.org/tijer/viewpaperforall.php?paper=TUER2505030
Plagiarism Report: 9% overall similarity (Turnitin, May 16, 2025)

#Prerequisites
To set up and run the Cerebral Smart Home system, ensure you have the following:

#Hardware
BioAmp EXG Pill (or compatible non-invasive EEG headset) for brainwave signal acquisition.
Webcam for eye-blink detection.
A mid-range computer (e.g., Intel i5 processor, 8 GB RAM) for running the system.

#Software
Python 3.9+
Required Python libraries:
numpy (numerical computations and signal processing)
scipy (signal filtering and feature extraction)
mediapipe (real-time eye-blink detection)
scikit-learn (Random Forest classifier for mental state classification)
pandas (data handling and preprocessing)
paho-mqtt (secure IoT communication)


Operating System: Compatible with Windows, macOS, or Linux.

#Installation
Clone the Repository:git clone https://github.com/abhayamanasa/EEG-based-smart-home-automation.git
Navigate to the Project Directory:cd EEG-based-smart-home-automation
Install Dependencies:pip install numpy scipy mediapipe scikit-learn pandas paho-mqtt


Hardware Setup:
Connect the BioAmp EXG Pill to your computer and place electrodes on the forehead as per the device manual.
Ensure your webcam is connected and functional for eye-blink detection.



#Usage
Run the System:python main.py


Calibrate the System:
The system will prompt a 25-minute calibration process to adapt to your brainwave patterns. Follow the on-screen instructions to record your EEG signals during relaxed and focused states.


Interact with the System:
Use intentional long blinks (detected via webcam) to navigate through a list of virtual devices (e.g., lamp, fan, thermostat).
Focus mentally to toggle the selected device on or off. A relaxed state will not trigger any action.
Visual feedback on the virtual interface will confirm actions (e.g., "Lamp On").


Emergency Stop:
Perform a predefined blink pattern (e.g., three rapid blinks) to stop the system, as detailed in the user guide.



#Project Structure
main.py: Main script to run the entire system.
eeg_processing.py: Handles EEG signal acquisition, preprocessing, and feature extraction.
blink_detection.py: Implements eye-blink detection using MediaPipe.
mental_state_classifier.py: Uses a Random Forest classifier to interpret mental states.
device_control.py: Manages IoT communication with virtual devices via MQTT.
interface.py: Provides the virtual interface for user interaction and feedback.
docs/: Contains user guides, project report, and additional documentation.
tests/: Includes test scripts and results for system validation.

#System Performance
The system was rigorously tested to ensure reliability and efficiency:

EEG Signal Processing: 90% noise reduction, with accurate feature extraction (e.g., power spectral density).
Eye-Blink Detection: 96% accuracy in detecting intentional long blinks.
Mental State Classification: 95% accuracy in distinguishing focused vs. relaxed states.
Latency: Average response time of 0.8 seconds, meeting real-time requirements.
Energy Efficiency: Consumes 2 watts on average, 15% below expectations.
Robustness: Maintains 90% classification accuracy with noisy EEG signals.
Scalability: Successfully integrates additional devices without performance degradation.

#Limitations
Virtual Device Limitation: Currently supports only virtual devices, not physical appliances.
Basic Commands: Limited to on/off toggling; lacks support for complex commands (e.g., dimming lights).
Calibration Time: Requires a 25-minute initial calibration, which may feel lengthy.
Cognitive Load: Prolonged use may cause mental fatigue due to sustained focus.
EEG Signal Quality: Susceptible to noise from muscle movements or improper electrode placement.

#Future Enhancements
Integrate with physical smart home devices for real-world applicability.
Support complex commands (e.g., adjusting brightness or temperature).
Reduce calibration time through improved EEG hardware or algorithms.
Implement adaptive learning for long-term personalization and reduced cognitive load.
Enable multi-user support for shared households.
Add voice feedback for users with visual impairments.
Use more comfortable, portable EEG devices for extended use.

#Ethical Considerations
Informed Consent: Obtained from all test participants, with clear explanations of data usage.
Data Privacy: EEG data anonymized, stored securely, and deleted post-testing. Transmission encrypted with TLS.
User Safety: Non-invasive BioAmp EXG Pill used to ensure comfort and safety.
Inclusivity: Designed for diverse users, avoiding bias in the machine learning model.
Transparency: Clear feedback provided via the interface; limitations communicated openly.

#Development Timeline
The project was developed over 5 months (January to May 2025):

Requirement Analysis & Stakeholder Interviews: January 2025 (1 month)
System Design & Hardware Setup: January 2025 (1 month)
Implementation (EEG Processing, Blink Detection, IoT Integration): February to March 2025 (2 months)
Testing (Unit, Integration, System): March to April 2025 (1.5 months)
Deployment, User Calibration, Documentation: April to May 2025 (1 month)

#Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a detailed description of your changes.Please ensure your code adheres to the project's coding standards and includes relevant tests.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions, feedback, or collaboration:


#Acknowledgments
Published in TIJER - International Research Journal (ISSN: 2349-9249).
Special thanks to the test participants for their valuable feedback.
References to prior research are listed in the project report (Chapter 9).

