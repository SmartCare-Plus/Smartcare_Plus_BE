# 🧓 SMARTCARE+:AI-BASED REMOTE CARE SYSTEM FOR MONITORING ELDERLY PARENTS’ DAILY HEALTH ACTIVITIES
![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Domain](https://img.shields.io/badge/Domain-Healthcare-green)
![AI](https://img.shields.io/badge/AI-Computer%20Vision-orange)
![Platform](https://img.shields.io/badge/Platform-Mobile-lightgrey)

An intelligent elderly care system designed to **monitor daily activities**, **detect and prevent falls**, and **support healthy aging** through **personalized meal plans and exercise recommendations**.

---

## 📌 Table of Contents
- [Research Problem](#-research-problem)
- [Proposed Solution](#-proposed-solution)
- [Main Features](#-main-features)
  - [1. Meal Recommendation Plan](#1-meal-recommendation-plan)
  - [2. Fall Detection & Prevention](#2-fall-detection--prevention)
  - [3. Exercise Generator](#3-exercise-generator)
- [System Architecture](#-system-architecture)
- [UI/UX Highlights](#-uiux-highlights)
- [Target Users](#-target-users)
- [Future Enhancements](#-future-enhancements)
- [Conclusion](#-conclusion)

---

## 🔍 Research Problem

The elderly population, especially individuals with **arthritis and mobility limitations**, faces multiple challenges:

- Increased risk of **falls and fall-related injuries**
- Lack of **real-time monitoring** when caregivers are unavailable
- Difficulty maintaining **balanced nutrition**
- Inappropriate or unsafe **exercise routines**
- Fragmented care solutions that do not work together

🔴 **Existing systems** often focus on only one aspect (health tracking or emergency alerts) and fail to provide a **holistic, preventive, and proactive solution**.

---

## 💡 Proposed Solution

This project proposes a **Smart Elderly Care System** that integrates:

- **AI-based fall detection and prevention warnings using multi-camera (CCTV) feeds**
- **Personalized meal recommendation plans**
- **Safe exercise generation and monitoring tailored to physical conditions**

The system supports both **elder users** and **caregivers** through an intuitive interface and intelligent alerts.

---

## 🚀 Main Features

### 1️⃣ Meal Recommendation Plan 🥗

**Purpose:**  
Ensure proper nutrition tailored to elderly individuals, especially those with arthritis and limited mobility.

**Key Capabilities:**
- Personalized meal plans based on:
  - Age
  - Health conditions
  - Activity level
- Daily meal schedules (Breakfast, Lunch, Dinner)
- Hydration reminders
- Easy-to-follow recommendations

**Outcome:**  
✔ Improves nutrition  
✔ Supports joint health  
✔ Reduces caregiver burden  

---

### 2️⃣ Fall Detection & Prevention 🚨

**Purpose:**  
Detect falls in real time and **prevent falls before they happen**.

**How it Works:**
- Activities captured through **CCTV cameras**
- AI model identifies:
  - Normal activities
  - Unstable movements
  - Near-fall situations
  - Actual falls

**Prevention Warnings (Displayed On-Screen):**
- “Please walk slowly”
- “Sit down for a moment”
- “Use nearby support”

**Critical Alerts:**
- Immediate caregiver notification
- Live camera access
- Emergency response options

**Outcome:**  
✔ Faster response time  
✔ Reduced fall incidents  
✔ Increased safety and confidence  

---

### 3️⃣ Exercise Generator 🏃‍♂️

**Purpose:**  
Promote safe physical activity without increasing injury risk.

**Key Features:**
- Low-impact exercises designed for:
  - Arthritis
  - Limited mobility
- Exercises categorized by:
  - Difficulty level
  - Joint focus
- Duration-based routines
- Clear instructions and visuals

**Outcome:**  
✔ Maintains mobility  
✔ Reduces stiffness  
✔ Encourages healthy routine  

---

## 🏗 System Architecture (High-Level)
<img width="1246" height="822" alt="image" src="https://github.com/user-attachments/assets/a087d2f6-2339-4baa-b21e-1c76076f0a2a" />


---
## 📦 Project Dependencies
- 🖥️ Frontend:
Flutter – Cross-platform mobile application development (Android & iOS)
- 🔧 Backend:
 FastAPI (Python) – High-performance REST API framework
- 🧠 AI / Machine Learning:
 MoViNet – Video-based human activity recognition model,
 CNN – Image/video-based recognition for exercises and meal analysis,
 TensorFlow / PyTorch – Model training and inference
- 🎥 Video Processing:
 OpenCV – Video frame processing,
 MediaPipe (or OpenPose) – Pose estimation and keypoint extraction
🗄️ Database:
Firebase – Real-time database, authentication, and cloud data storage
- 🚨 Alerts & Notifications:
Firebase Cloud Messaging (FCM) – Push notifications,
SMS / Email APIs – Emergency alerts to caregivers


## 🎨 UI/UX Highlights

- Large fonts & high contrast (elder-friendly)
- Color-coded risk indicators (Normal / Warning / Critical)
- Simple navigation for caregivers
- Real-time alerts with minimal interaction required
- Accessibility-focused design

---

## 👥 Target Users

- Elderly individuals (especially with arthritis)
- Family caregivers
- Professional caregivers
- Healthcare support staff

---

## ✅ Conclusion

This system delivers a **comprehensive, preventive, and intelligent elderly care solution** by combining **AI-based fall detection**, **nutrition planning**, and **safe exercise generation**.  
It improves **quality of life**, enhances **caregiver efficiency**, and promotes **safe independent living** for the elderly.

---




