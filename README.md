# Classification Project

## Objective
Perform classification to predict the earning category (<=50K or >50K) based on various demographic and work-related attributes. Draw inferences from the classification results.

## Data Description
The dataset contains information on individuals, including their demographic details, work-related attributes, and earnings. The goal is to classify individuals into earning categories based on these attributes.

### Columns
- **age**: Age of the individual
- **workclass**: Type of employment (e.g., Private, Self-emp-not-inc)
- **fnlwgt**: Final weight, a measure used in the dataset
- **education**: Highest level of education achieved
- **education-num**: Number of years of education
- **marital-status**: Marital status of the individual
- **occupation**: Type of occupation (e.g., Exec-managerial, Handlers-cleaners)
- **relationship**: Relationship status within a family (e.g., Husband, Not-in-family)
- **race**: Race of the individual
- **sex**: Gender of the individual
- **capital-gain**: Capital gains in the past year
- **capital-loss**: Capital losses in the past year
- **hours-per-week**: Number of hours worked per week
- **native-country**: Country of origin
- **earning**: Earning category (<=50K or >50K)

## Steps to Perform Classification
1. **Data Preprocessing**: Clean the data by handling missing values, encoding categorical variables, and scaling numerical features.
2. **Feature Selection**: Select relevant features that contribute to the classification task.
3. **Model Training**: Train various classification models (e.g., Logistic Regression, Decision Trees, Random Forest) on the preprocessed data.
4. **Model Evaluation**: Evaluate the models using metrics like accuracy, precision, recall, and F1-score.
5. **Hyperparameter Tuning**: Optimize the model parameters to improve performance.
6. **Analyze Results**: Draw inferences from the classification results and interpret the model's predictions.

## Inferences
- **Feature Importance**: Identify the most important features that influence the earning category.
- **Model Performance**: Compare the performance of different models and select the best one.
- **Insights**: Provide insights based on the classification results, such as identifying key factors that contribute to higher earnings.

## Usage
1. **Clone Repository**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/muzammiltariq95/income-classification.git
   ```
2. **Install Dependencies**: Install the required dependencies using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Analysis**: Execute the provided scripts to perform classification and analyze the results.

## Contributing
Contributions are welcome! Please read the contributing guidelines for more details.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
