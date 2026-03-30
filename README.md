# Unsupervised Learning: Customer Segmentation via PCA and K-Means Clustering

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
This repository contains a professional-standard tutorial on **Unsupervised Learning**. The project demonstrates how to identify hidden patterns in high-dimensional financial data without using pre-labeled targets. 

By combining **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering**, we turn 17 complex customer behaviors into 4 distinct, actionable market segments.

## 🚀 Key Technical Features
- **Data Engineering:** Handling missing values via forward-filling and robust feature scaling.
- **The Elbow Method:** An experimental "intermediate step" used to mathematically determine the optimal number of clusters ($k$).
- **Dimensionality Reduction (PCA):** Compressing 17-dimensional data into a 2D plane to allow for human-interpretable visualization.
- **Professional Workflow:** Automated directory management and high-resolution result generation.

## 📁 Project Structure
```text
Customer-Segmentation-Tutorial/
├── data/                   <-- kaggle dataset
├── output/                 <-- plots
├── main.py                 <-- the clean Python script version
├── notebook/               <-- the interactive tutorial notebook
├── report/                 <-- the report
├── README.md               
├── requirements.txt        <-- Dependencies
└── LICENSE                 <-- MIT License
```

## 📊 Visual Results
The tutorial generates two primary visualizations (saved in the `/output` folder):
1. **The Elbow Plot:** Demonstrates the "Inertia" reduction and the mathematical justification for choosing 4 clusters.
2. **PCA Cluster Map:** A professional 2D visualization showing how 9,000 customers are grouped into 4 distinct behavioral segments.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/sumanth589/Customer-Segmentation-Tutorial.git
   cd Customer-Segmentation-Tutorial
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data:**
   Download the [Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) from Kaggle and place `CC GENERAL.csv` into the `/data` folder.

## ♿ Accessibility & Inclusive Design
In compliance with the assignment rubric, this project emphasizes inclusivity:
- **High-Contrast Plots:** We use a distinct color palette (Red, Blue, Green, Orange) to ensure visibility for color-blind users.
- **Transparency (Alpha):** We use semi-transparent data points to allow users to see overlapping data density.
- **Human-Readable Code:** Comments are written in clear, simple English to ensure the tutorial functions as an effective teaching tool for everyone.

## ⚖️ Ethical AI: Privacy & Profiling
Unsupervised learning carries the risk of "unintended profiling." This tutorial addresses the ethical responsibility of data scientists to ensure that clusters are based on financial behavior and not on protected attributes that could lead to discriminatory banking practices.

## 📚 References
1. **Arthur, D., & Vassilvitskii, S. (2007).** *k-means++: The advantages of careful seeding.* (The foundation for modern K-Means).
2. **Pearson, K. (1901).** *On lines and planes of closest fit to systems of points in space.* (The foundational paper for PCA).
3. **Scikit-Learn Documentation:** *Clustering and Dimensionality Reduction.*

## 📄 License
This project is licensed under the **MIT License**.