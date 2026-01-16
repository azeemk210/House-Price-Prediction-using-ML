# House-Price-Prediction-using-ML

## Project Overview
This project aims to predict house prices using machine learning techniques. It involves data preprocessing, model training, evaluation, and deployment. The dataset includes various features that influence house prices, and the goal is to build a model that accurately predicts prices based on these features.

## Directory Structure
```
House-Price-Prediction-using-ML/
│
├── data/
│   ├── raw/                # Raw dataset files
│   │   ├── data_description.txt
│   │   ├── train.csv
│   │   ├── train.csv.dvc
│   │   ├── test.csv
│   │   └── test.csv.dvc
│   └── processed/          # Processed dataset files
│       └── train.csv
│
├── models/                 # Trained models and related files
│
├── src/                    # Source code for the project
│   ├── preprocess.py       # Data preprocessing scripts
│   ├── train.py            # Model training scripts
│   ├── evaluate.py         # Model evaluation scripts
│   └── models/             # Additional model-related scripts
│
├── dvc.yaml                # DVC pipeline configuration
├── LICENSE                 # License information
├── metrics.json            # Metrics generated during evaluation
└── README.md               # Project documentation (this file)
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- DVC (Data Version Control)
- Git
- Required Python packages (listed in `requirements.txt` if available)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/azeemk210/House-Price-Prediction-using-ML.git
   ```
2. Navigate to the project directory:
   ```bash
   cd House-Price-Prediction-using-ML
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
1. Ensure the raw data files are present in the `data/raw/` directory.
2. Use the preprocessing script to process the data:
   ```bash
   python src/preprocess.py
   ```

### Running the Project
- Train the model:
  ```bash
  python src/train.py
  ```
- Evaluate the model:
  ```bash
  python src/evaluate.py
  ```

### Version Control with DVC
- To pull the data files:
  ```bash
  dvc pull
  ```
- To push changes to the remote storage:
  ```bash
  dvc push
  ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [DVC](https://dvc.org/) for data version control
- [Kaggle](https://www.kaggle.com/) for providing datasets

## Contact
For any inquiries, please contact [Azeem](https://github.com/azeemk210).