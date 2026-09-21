# Software Bug Prediction System

An ML-based software bug prediction system that analyzes source-code metrics and predicts potential bug risk at the source-file level.

The system combines machine learning, SHAP explainability, FastAPI, and GitHub repository analysis to provide bug-risk predictions and recommendations.

## Features

- Machine-learning-based bug prediction
- Source-code metric analysis
- Feature selection and hyperparameter tuning
- Leakage-safe ML pipeline
- SHAP-based explainability
- Bug probability estimation
- Risk-level classification
- GitHub repository analysis
- Source-file-level risk prediction
- Highest-risk file identification
- Risk factors and recommendations
- Responsive web interface

## Workflow

```text
GitHub Repository
       ↓
Source File Analysis
       ↓
Software Metrics
       ↓
ML Prediction Pipeline
       ↓
Bug Probability
       ↓
Risk Level
       ↓
SHAP Explanation
       ↓
Risk Factors & Recommendations
````

## Software Metrics

The final model uses 12 selected metrics:

* `loc`
* `v_g`
* `ev_g`
* `iv_g`
* `l`
* `d`
* `i`
* `t`
* `lOComment`
* `lOBlank`
* `locCodeAndComment`
* `uniq_Op`

## Risk Levels

| Risk Level | Description                  |
| ---------- | ---------------------------- |
| LOW        | Lower predicted bug risk     |
| MEDIUM     | Moderate predicted bug risk  |
| HIGH       | Higher predicted bug risk    |
| CRITICAL   | Very high predicted bug risk |

## Technologies

* Python
* Scikit-learn
* SHAP
* Pandas
* NumPy
* FastAPI
* Uvicorn
* JavaScript
* HTML
* CSS
* GitHub API

## GitHub Repository Analysis

Users can enter a public GitHub repository URL to analyze its source files.

The system:

1. Validates the repository URL.
2. Identifies source files.
3. Extracts software metrics.
4. Runs the ML prediction pipeline.
5. Calculates bug probability.
6. Assigns a risk level.
7. Generates SHAP-based risk factors.
8. Displays recommendations for high-risk files.

## Validation

The system was tested using publicly available GitHub repositories. The complete pipeline successfully generated bug-risk predictions ranging from LOW to CRITICAL for different source files.

GitHub validation is considered functional/external validation because independent ground-truth defect labels were not available for the analyzed files.

## Security

GitHub authentication is handled using an environment variable:

```text
GITHUB_TOKEN=your_github_token
```

The `.env` file is excluded from version control using `.gitignore`.

## Running Locally

Install the required Python packages and start the FastAPI backend:

```bash
python -m uvicorn app:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Then open the frontend:

```text
frontend/index.html
```

## Future Enhancements

* Support for additional programming languages
* More source-code metrics
* Pull-request analysis
* CI/CD integration
* Continuous repository monitoring
* Cloud deployment

## Project Author

**Monisha Mohan**

Artificial Intelligence and Data Science

## Project Status

**Functional prototype** with ML prediction, SHAP explainability, FastAPI backend, GitHub repository analysis, and responsive frontend.

## License

This project is developed for academic and educational purposes.


