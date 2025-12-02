# Adaptive Learning for Highschool Students in Bosnia and Herzegovina

This repository contains a project for **adaptive learning** for highschool students in Bosnia and Herzegovina.  
The project was created as part of the course **"Machine Learning: Supervised Techniques"**.

---

## 📄 Project Description

The goal of this project is to explore adaptive learning techniques using **student answer data** and **question metadata**.  
We implemented two approaches:

1. **Random Forest** – predicts whether a student will answer the next question correctly.  
2. **Manual Bayesian Knowledge Tracing (BKT)** – models the student's **mastery progression** over time for each skill.

Even with a single student's data, BKT allows us to visualize **how mastery of different skills evolves** as the student practices.

---

## 🗂 Repository Structure

'''Adaptive-Learning-For-HighSchools/
│
├── Code/                  # All code files (scripts, notebooks)
│
├── Documentation/         # Documentation files (project description, methodology)
│   ├── README.md
|   └── Summary of related work
│
├── Dataset/               # All datasets
│   ├── EdNet-KT1
|   |    └── u1000.csv      # All students
│   └── questions.csv
│
├── Results/               # plots, model outputs, evaluation metrics
│
├── Presentations/         # Optional: slides for progress reviews
│   └── project_slides.pdf
│
└── requirements.txt       # Python dependencies'''
