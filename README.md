# Titanic Survival Prediction

## Project Overview

This project leverages machine learning to predict the survival of Titanic passengers based on demographic and travel information. Using a **Random Forest Classifier**, it analyzes features such as age, gender, class, and fare to determine survival outcomes.

---

## Dataset

The dataset contains data on 891 passengers, with the following columns:

| Column         | Description                                                               |
|----------------|---------------------------------------------------------------------------|
| `PassengerId`  | Unique identifier for each passenger                                     |
| `Survived`     | Target variable (1 = Survived, 0 = Did Not Survive)                      |
| `Pclass`       | Passenger class (1 = First, 2 = Second, 3 = Third)                       |
| `Name`         | Full name of the passenger                                               |
| `Sex`          | Gender of the passenger (`male`, `female`)                               |
| `Age`          | Age of the passenger                                                     |
| `SibSp`        | Number of siblings/spouses aboard                                        |
| `Parch`        | Number of parents/children aboard                                        |
| `Ticket`       | Ticket number                                                           |
| `Fare`         | Ticket fare                                                             |
| `Cabin`        | Cabin number (if available)                                             |
| `Embarked`     | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton)|

---


## Project Structure

```
Titanic-Survival-Prediction/
│
├── data/
│   └── Titanic-Dataset.csv         # Dataset
│
├── Notebooks/
│   └── EDA.ipynb 
├── SCripts/                  
│   ├── titanic.py.py   
├──.gitignore
├──LICENSE
├── README.md              # Project documentation
└── requirements.txt       # Required Python libraries
```

---

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

**Required Libraries**:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/dani6566/CODSOFT-Titanic-Survival-Prediction.git
   ```
---

## Future Work
1. Add an interactive dashboard for visualizing predictions.

---

Feel free to contribute by submitting issues or pull requests!
