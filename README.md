---

# Customer Segmentation using K-Means Clustering

This repository contains a complete machine learning project on **Customer Segmentation** using the **K-Means clustering algorithm**. The purpose of this project is to group customers based on their purchasing behavior and demographic attributes, enabling businesses to understand customer categories and build targeted marketing strategies.

This entire project was **independently developed by me**, including data preprocessing, exploratory analysis, clustering, visualization, and interpretation of customer groups.

---

## Project Overview

Customer segmentation is a common application of unsupervised machine learning. Using K-Means clustering, we can divide customers into meaningful groups such as high-value customers, low-engagement customers, budget shoppers, and more.

This project demonstrates how K-Means can identify hidden patterns in unlabeled data and help businesses personalize their approach for each customer segment.

---

## Key Features

* Fully implemented customer segmentation workflow
* K-Means clustering from scratch using Scikit-learn
* Clean preprocessing pipeline
* Visualizations for clusters and feature distributions
* Elbow Method and Silhouette Score used for optimal cluster selection
* Clear interpretation of each customer segment
* Built entirely by me as an individual data science project

---

## Dataset

The dataset used for this project is publicly available on Kaggle:

**Dataset Source:**
[https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

All rights belong to the original dataset creator. The dataset is used here only for educational and research purposes.

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## Project Workflow

### 1. Load and Inspect Data

* Import dataset using Pandas
* Review structure, missing values, and basic statistics

### 2. Data Preprocessing

* Handle missing values
* Select relevant features (e.g., Age, Annual Income, Spending Score)
* Scale features using StandardScaler for better clustering performance

### 3. Exploratory Data Analysis

* Distribution plots
* Correlation analysis
* Feature relationships visualized

### 4. Choosing the Number of Clusters

* Elbow Method
* Silhouette Score
* Decision on optimal number of clusters

### 5. Applying K-Means Clustering

* Fit the K-Means model
* Assign each customer to a cluster
* Visualize cluster boundaries and group centers

### 6. Interpreting Clusters

Each cluster is analyzed based on:

* Spending behavior
* Income level
* Age and lifestyle attributes

These insights help businesses identify segments such as:

* High-income, high-spending customers
* Low-income but high-engagement customers
* Budget-conscious customers
* Younger vs. older customer groups

---

## Project Structure

```
Customer-Segmentation-KMeans/
│
├── data/
│   └── Mall_Customers.csv        # Dataset (not included in repo)
│
├── notebook/
│   └── customer_segmentation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   └── visualization.py
│
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repository:

```
git clone https://github.com/yourusername/Customer-Segmentation-KMeans.git
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Run the notebook:

```
jupyter notebook
```

4. Open `customer_segmentation.ipynb` to explore the full project.

---

## Author

This project was **fully created and implemented by me**—from cleaning the data to designing the clustering model and interpreting the results.

---

## License

This project is intended for learning and educational purposes only.

---

