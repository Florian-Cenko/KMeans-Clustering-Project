# KMeans-Clustering-Project

This project applies **KMeans Clustering**, an **unsupervised learning algorithm**, to group universities into **two clusters: Private and Public**.

Although the dataset contains labels for private/public status, we **do not use them in the clustering process**, as KMeans is meant to work without labeled data.
The labels are only used after clustering to evaluate algorithm performance (for learning purposes).



## The Data
The dataset contains **777 observation**s with the following 18 variables:

- Private: Factor with levels Yes/No indicating private or public university
- Apps: Number of applications received
- Accept: Number of applications accepted
- Enroll: Number of new students enrolled
- Top10perc: % of new students from top 10% of high school class
- Top25perc: % of new students from top 25% of high school class
- F.Undergrad: Number of full-time undergraduates
- P.Undergrad: Number of part-time undergraduates
- Outstate: Out-of-state tuition
- Room.Board: Room and board costs
- Books: Estimated book costs
- Personal: Estimated personal spending
- PhD: % of faculty with Ph.D.’s
- Terminal: % of faculty with terminal degree
- S.F.Ratio: Student/faculty ratio
- perc.alumni: % of alumni who donate
- Expend: Instructional expenditure per student
- Grad.Rate: Graduation rate


## Objectives

- Perform **Exploratory Data Analysis (EDA)** to understand university distributions
- Implement **KMeans Clustering** to group universities into clusters
- Compare the clusters to actual labels (Private/Public) to evaluate performance
- Visualize cluster assignments and assess patterns in the data

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- PyCharm

## How to Run
1. Clone the repository:  
```bash
     git clone <your-repo-url>
```
2. Create and activate a virtual environment (Recommended):
```bash
     python -m venv venv
```
   - Linux/macOS:
     ```bash
          source venv/bin/activate
     ```
   - Windows:
     ```bash
          venv\Scripts\activate
     ```
3. Install dependencies:
 ```bash
      pip install <name of libraries>
```
3. Run the main Python script:
 ```bash
      python KMeans Clustering.py
```

## Notes
- KMeans is an **unsupervised algorithm**, so in real-world use, labels are usually **not available**
- Labels here are only used to **check clustering performance** after the fact
- Good practice for understanding **clustering, feature scaling, and visualization**
