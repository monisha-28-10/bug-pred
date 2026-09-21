from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import shap
import requests
from pydantic import BaseModel

import re
import lizard

import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
github_headers = {
    "Accept": "application/vnd.github+json"
}

if GITHUB_TOKEN:
    github_headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Software Bug Prediction API",
    description="ML-based software bug prediction system",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD SAVED ARTIFACTS
# ============================================

final_pipeline = joblib.load(
    "final_bug_prediction_pipeline.pkl"
)

selected_features = joblib.load(
    "selected_features.pkl"
)

shap_background = joblib.load(
    "shap_background.pkl"
)

with open("model_config.json", "r") as f:
    config = json.load(f)

FINAL_THRESHOLD = config["classification_threshold"]


# ============================================
# EXTRACT PIPELINE COMPONENTS
# ============================================

log_transform = final_pipeline.named_steps["log_transform"]
scaler = final_pipeline.named_steps["scaler"]
model = final_pipeline.named_steps["model"]


# ============================================
# CREATE SHAP EXPLAINER
# ============================================

explainer = shap.LinearExplainer(
    model,
    shap_background
)


# ============================================
# INPUT DATA
# ============================================

class BugPredictionInput(BaseModel):

    loc: float
    v_g: float
    ev_g: float
    iv_g: float
    l: float
    d: float
    i: float
    t: float
    lOComment: float
    lOBlank: float
    locCodeAndComment: float
    uniq_Op: float


# ============================================
# RISK LEVEL
# ============================================

def get_risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.60:
        return "MEDIUM"

    elif probability < 0.80:
        return "HIGH"

    else:
        return "CRITICAL"


# ============================================
# EXACT NOTEBOOK EXPLANATION LOGIC
# ============================================

def explain_feature(feature, value):

    explanations = {

        "loc": (
            f"Large code size ({value:.0f} LOC) may indicate a large or "
            "complex module that requires additional review and testing."
        ),

        "t": (
            f"Very high estimated effort/time ({value:.2f}) indicates "
            "a potentially difficult and resource-intensive module."
        ),

        "uniq_Op": (
            f"High unique operator count ({value:.0f}) indicates "
            "greater operator diversity and potentially more complex logic."
        ),

        "iv(g)": (
            f"High independent-path complexity ({value:.0f}) indicates "
            "many independent execution paths that may require additional testing."
        ),

        "v(g)": (
            f"High cyclomatic complexity ({value:.0f}) indicates "
            "more complex control flow."
        ),

        "l": (
            f"Program length ({value:.2f}) contributes to the model's "
            "assessment of module complexity."
        ),

        "i": (
            f"High intelligence metric ({value:.2f}) contributed to "
            "the model's prediction."
        ),

        "lOBlank": (
            f"High blank-line count ({value:.0f}) contributes to the "
            "structural characteristics considered by the model."
        ),

        "lOComment": (
            f"Comment-line count ({value:.0f}) contributed to the "
            "model's prediction."
        ),

        "locCodeAndComment": (
            f"Lines containing code and comments ({value:.0f}) "
            "contributed to the model's prediction."
        ),

        "ev(g)": (
            f"High essential complexity ({value:.0f}) contributed to "
            "the model's assessment."
        )
    }

    return explanations.get(
        feature,
        f"{feature} contributed to the model's prediction."
    )


# ============================================
# EXACT NOTEBOOK RECOMMENDATION LOGIC
# ============================================

def get_recommendation(feature):

    recommendations = {

        "loc": (
            "Consider breaking the module into smaller functions or "
            "components and perform additional code review."
        ),

        "t": (
            "Review the module for unnecessary complexity and consider "
            "refactoring time-consuming or difficult sections."
        ),

        "uniq_Op": (
            "Review operator usage and simplify complex expressions "
            "where possible. Add focused unit tests for critical logic."
        ),

        "iv(g)": (
            "Increase testing coverage for independent execution paths "
            "and consider simplifying complex control flow."
        ),

        "v(g)": (
            "Review complex conditional and branching logic. Consider "
            "refactoring methods with excessive control-flow complexity."
        ),

        "l": (
            "Consider decomposing the module into smaller, more "
            "maintainable components."
        ),

        "i": (
            "Review the module's overall computational complexity and "
            "test critical functionality thoroughly."
        ),

        "lOBlank": (
            "Review the module structure and formatting, while focusing "
            "primarily on substantive code complexity."
        ),

        "lOComment": (
            "Review whether important logic is adequately documented "
            "and ensure comments accurately reflect the implementation."
        ),

        "locCodeAndComment": (
            "Review mixed code/comment lines for readability and "
            "maintainability."
        ),

        "ev(g)": (
            "Review essential control-flow complexity and add tests "
            "for the important logical paths."
        )
    }

    return recommendations.get(
        feature,
        "Perform additional code review and testing for this feature."
    )

# ============================================
# ROOT-CAUSE ANALYSIS
# ============================================

def get_root_cause(feature, value):

    root_causes = {

        "loc": (
            "Large module size may increase complexity "
            "and make the code harder to maintain."
        ),

        "v(g)": (
            "High cyclomatic complexity indicates complex "
            "control flow with many decision paths."
        ),

        "ev(g)": (
            "High essential complexity indicates difficult "
            "or poorly structured control flow."
        ),

        "iv(g)": (
            "High independent-path complexity indicates "
            "many execution paths requiring additional testing."
        ),

        "l": (
            "Program length contributes to the structural "
            "complexity of the module."
        ),

        "d": (
            "High difficulty indicates that the module may "
            "be difficult to understand or modify."
        ),

        "i": (
            "High program intelligence indicates increased "
            "computational or structural complexity."
        ),

        "t": (
            "High estimated effort/time suggests that the "
            "module may require substantial development effort."
        ),

        "lOComment": (
            "Comment-line characteristics may indicate "
            "documentation or maintainability issues."
        ),

        "lOBlank": (
            "Blank-line characteristics contribute to the "
            "structural properties of the module."
        ),

        "locCodeAndComment": (
            "Mixed code-and-comment lines may affect "
            "readability and maintainability."
        ),

        "uniq_Op": (
            "High unique-operator diversity can indicate "
            "more complex program logic."
        )
    }

    return root_causes.get(
        feature,
        "This feature contributed to the predicted defect risk."
    )

# ============================================
# HOME
# ============================================

@app.get("/")
def home():

    return {
        "message": "Software Bug Prediction API is running"
    }

def get_overall_recommendation(risk_level, risk_factors):

    if risk_level == "CRITICAL":
        return (
            "Immediate code review is recommended. "
            "Focus on the highest-impact contributing factors, "
            "refactor complex code, and increase unit and regression testing."
        )

    elif risk_level == "HIGH":
        return (
            "Additional code review and testing are recommended. "
            "Prioritize the features with the highest risk contributions."
        )

    elif risk_level == "MEDIUM":
        return (
            "Review the main contributing factors and consider "
            "targeted refactoring and additional testing."
        )

    else:
        return (
            "The predicted defect risk is low. "
            "Continue regular code review and testing practices."
        )


class GitHubRequest(BaseModel):
    github_url: str

# ============================================
# SOURCE CODE METRIC EXTRACTION
# ============================================

def calculate_basic_metrics(code, filename="github_source.js"):

    lines = code.splitlines()

    loc = 0
    blank_lines = 0
    comment_lines = 0
    code_comment_lines = 0

    in_block_comment = False

    for line in lines:

        stripped = line.strip()

        if not stripped:
            blank_lines += 1
            continue

        if in_block_comment:
            comment_lines += 1

            if "*/" in stripped:
                in_block_comment = False

            continue

        if stripped.startswith("/*"):
            comment_lines += 1

            if "*/" not in stripped[2:]:
                in_block_comment = True

            continue

        if (
            stripped.startswith("//")
            or stripped.startswith("#")
            or stripped.startswith("--")
        ):
            comment_lines += 1
            continue

        if "//" in line or "#" in line:
            code_comment_lines += 1

        loc += 1

    try:

        analysis = lizard.analyze_file.analyze_source_code(
            filename,
            code
        )

        print("=== LIZARD ANALYSIS ===")

        for function in analysis.function_list:
            print("Function:", function.name)
            print("NLOC:", function.nloc)
            print("Cyclomatic:", function.cyclomatic_complexity)
            print("Parameters:", function.parameter_count)
            print("----------------------")

        functions = analysis.function_list

        if functions:

            cyclomatic_values = [
                f.cyclomatic_complexity
                for f in functions
            ]

            cyclomatic = max(cyclomatic_values)

        else:
            cyclomatic = 1

    except Exception:

        analysis = None
        cyclomatic = 1

    metrics = {
        "loc": float(loc),

        "v(g)": float(cyclomatic),

        # NASA PROMISE-compatible proxy for GitHub analysis.
        # Exact ev(g) and iv(g) require McCabe flow-graph reduction.
        "ev(g)": float(cyclomatic),
        "iv(g)": float(cyclomatic),

        "lOComment": float(comment_lines),
        "lOBlank": float(blank_lines),
        "locCodeAndComment": float(code_comment_lines)
    }

    # Add Halstead metrics
    if analysis is not None:

        halstead = calculate_halstead_metrics(
            code,
            analysis
        )

        metrics.update(halstead)

    else:

        metrics.update({
            "l": 0.0,
            "d": 0.0,
            "i": 0.0,
            "t": 0.0,
            "uniq_Op": 0.0
        })

    return metrics

def calculate_halstead_metrics(code, analysis):

    operators = {
        "+", "-", "*", "/", "%", "**",
        "=", "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!", "&", "|", "^", "~",
        "<<", ">>", "++", "--",
        "+=", "-=", "*=", "/=", "%=",
        "&=", "|=", "^=", "<<=", ">>=",
        "?", ":", "=>",
        "return", "if", "else", "for", "while",
        "do", "switch", "case", "break", "continue",
        "function", "class", "new", "delete",
        "try", "catch", "finally", "throw",
        "import", "from", "export", "extends",
        "in", "of", "typeof", "instanceof"
    }

    # Remove comments
    code_no_comments = re.sub(
        r"//.*|/\*[\s\S]*?\*/|#.*",
        "",
        code
    )

    # Tokenize source code
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|"
        r"\d+(?:\.\d+)?|"
        r"==|!=|<=|>=|&&|\|\||\+\+|--|"
        r"\+=|-=|\*=|/=|%=|&=|\|=|\^=|"
        r"<<=|>>=|<<|>>|=>|\*\*|"
        r"[+\-*/%=&|^~!<>?:]",
        code_no_comments
    )

    operator_tokens = []
    operand_tokens = []

    for token in tokens:

        if token in operators:
            operator_tokens.append(token)

        elif re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*$",
            token
        ):
            operand_tokens.append(token)

        elif re.match(r"^\d", token):
            operand_tokens.append(token)

    # Base Halstead measures
    uniq_op = len(set(operator_tokens))
    uniq_opnd = len(set(operand_tokens))

    total_op = len(operator_tokens)
    total_opnd = len(operand_tokens)

    N = total_op + total_opnd
    vocabulary = uniq_op + uniq_opnd

    if N == 0 or vocabulary == 0:
        return {
            "l": 0.0,
            "d": 0.0,
            "i": 0.0,
            "t": 0.0,
            "uniq_Op": 0.0
        }

    # -----------------------------
    # NASA Halstead calculations
    # -----------------------------

    # Volume
    volume = N * np.log2(vocabulary)

    # Potential operator count
    mu1_prime = 2

    # Potential operand count
    if analysis.function_list:
        mu2_prime = max(
            f.parameter_count
            for f in analysis.function_list
        )
    else:
        mu2_prime = 0

    # Minimum vocabulary
    minimum_vocabulary = (
        mu1_prime + mu2_prime
    )

    if minimum_vocabulary > 0:

        minimum_volume = (
            minimum_vocabulary *
            np.log2(minimum_vocabulary)
        )

    else:
        minimum_volume = 0.0

    # Program length
    program_length = minimum_volume / N

    # Difficulty
    if program_length > 0:
        difficulty = 1 / program_length
    else:
        difficulty = 0.0

    # Intelligence
    intelligence = program_length * volume

    # Effort
    effort = volume / program_length if program_length > 0 else 0.0

    # Time
    time = effort / 18

    return {
        "l": float(program_length),
        "d": float(difficulty),
        "i": float(intelligence),
        "t": float(time),
        "uniq_Op": float(uniq_op)
    }

@app.post("/analyze-github")
def analyze_github(data: GitHubRequest):

    url = data.github_url.strip()

    if not url.startswith("https://github.com/"):
        return {
            "success": False,
            "message": "Invalid GitHub repository URL."
        }

    parts = url.rstrip("/").split("/")

    if len(parts) < 5:
        return {
            "success": False,
            "message": "Invalid GitHub repository URL."
        }

    username = parts[3]
    repository = parts[4]

    # Remove .git if the user enters it
    if repository.endswith(".git"):
        repository = repository[:-4]

    repo_api = f"https://api.github.com/repos/{username}/{repository}"

    try:

        # Check repository
        repo_response = requests.get(
            repo_api,
            headers=github_headers,
            timeout=10
        )

        if repo_response.status_code == 404:
            return {
                "success": False,
                "message": "GitHub repository not found or is not public."
            }

        if repo_response.status_code != 200:
            return {
                "success": False,
                "message": f"GitHub API error: {repo_response.status_code}"
            }

        repo_data = repo_response.json()

        default_branch = repo_data["default_branch"]

        # Get complete repository tree
        tree_url = (
            f"https://api.github.com/repos/"
            f"{username}/{repository}/git/trees/"
            f"{default_branch}?recursive=1"
        )

        tree_response = requests.get(
            tree_url,
            headers=github_headers,
            timeout=20
        )

        if tree_response.status_code != 200:
            return {
                "success": False,
                "message": "Unable to read repository files."
            }

        tree_data = tree_response.json()

        print("GitHub tree status:", tree_response.status_code)
        print("GitHub tree items:", len(tree_data.get("tree", [])))
        print("GitHub tree sample:", tree_data.get("tree", [])[:10])

        # Source-code file extensions
        source_extensions = (
            ".py",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".cs",
            ".php",
            ".rb",
            ".go",
            ".html",
            ".css"
        )

        source_files = []

        for item in tree_data.get("tree", []):

            if item.get("type") != "blob":
                continue

            file_path = item.get("path", "")

            if file_path.lower().endswith(source_extensions):
                source_files.append(file_path)


        # Read source-code contents
        source_contents = []

        for file_path in source_files:

            file_api_url = (
                f"https://api.github.com/repos/"
                f"{username}/{repository}/contents/{file_path}"
            )

            file_response = requests.get(
                file_api_url,
                headers=github_headers,
                timeout=10
            )

            if file_response.status_code != 200:
                continue

            file_data = file_response.json()

            import base64

            encoded_content = file_data.get("content", "")

            try:
                decoded_content = base64.b64decode(
                    encoded_content
                ).decode("utf-8", errors="ignore")
            except Exception:
                continue

            source_contents.append({
                "file": file_path,
                "content": decoded_content
            })


        # ============================================
        # CALCULATE GITHUB METRICS
        # ============================================

        github_metrics = []

        for source in source_contents:

            metrics = calculate_basic_metrics(
                source["content"],
                source["file"]
            )

            github_metrics.append({
                "file": source["file"],
                "metrics": metrics
            })

        # ---------------------------------------
        # Predict + Explain GitHub source files
        # ---------------------------------------

        github_predictions = []

        for item in github_metrics:

            metrics = item["metrics"]

            try:

                github_input = BugPredictionInput(
                    loc=metrics["loc"],
                    v_g=metrics["v(g)"],
                    ev_g=metrics["ev(g)"],
                    iv_g=metrics["iv(g)"],
                    l=metrics["l"],
                    d=metrics["d"],
                    i=metrics["i"],
                    t=metrics["t"],
                    lOComment=metrics["lOComment"],
                    lOBlank=metrics["lOBlank"],
                    locCodeAndComment=metrics["locCodeAndComment"],
                    uniq_Op=metrics["uniq_Op"]
                )

                result = predict_bug(github_input)

                github_predictions.append({
                    "file": item["file"],
                    "bug_probability": result["bug_probability"],
                    "prediction": result["prediction"],
                    "risk_level": result["risk_level"],
                    "risk_factors": result["risk_factors"],
                    "overall_recommendation": result[
                        "overall_recommendation"
                    ]
                })

            except Exception as e:

                github_predictions.append({
                    "file": item["file"],
                    "error": str(e)
                })
        if len(source_files) == 0:
            return {
                "success": False,
                "username": username,
                "repository": repository,
                "default_branch": default_branch,
                "source_file_count": 0,
                "message": "No supported source files found in this repository."
            }
        return {
            "success": True,
            "username": username,
            "repository": repository,
            "default_branch": default_branch,
            "total_files": len(tree_data.get("tree", [])),
            "source_files": source_files,
            "source_file_count": len(source_files),
            "source_contents": source_contents,
            "github_metrics": github_metrics,
            "github_predictions": github_predictions,
            "message": "Repository source code retrieved successfully."

        }

    except requests.RequestException:
        return {
            "success": False,
            "message": "Unable to connect to GitHub."
        }

# ============================================
# PREDICTION + EXPLANATION
# ============================================

@app.post("/predict")
def predict_bug(data: BugPredictionInput):

    # ----------------------------------------
    # Create DataFrame with EXACT model names
    # ----------------------------------------

    input_data = {

        "loc": data.loc,

        "v(g)": data.v_g,

        "ev(g)": data.ev_g,

        "iv(g)": data.iv_g,

        "l": data.l,

        "d": data.d,

        "i": data.i,

        "t": data.t,

        "lOComment": data.lOComment,

        "lOBlank": data.lOBlank,

        "locCodeAndComment": data.locCodeAndComment,

        "uniq_Op": data.uniq_Op
    }

    sample = pd.DataFrame([input_data])

    # Keep exact selected features
    sample = sample[selected_features]


    # ========================================
    # 1. PREDICTION
    # ========================================

    probability = final_pipeline.predict_proba(
        sample
    )[0, 1]

    prediction = int(
        probability >= FINAL_THRESHOLD
    )

    risk_level = get_risk_level(
        probability
    )


    # ========================================
    # 2. SHAP EXPLANATION
    # ========================================

    sample_log = log_transform.transform(
        sample
    )

    sample_scaled = scaler.transform(
        sample_log
    )

    sample_shap = explainer(
        sample_scaled
    )

    shap_values_sample = sample_shap.values[0]


    # ========================================
    # 3. CREATE EXPLANATION TABLE
    # ========================================

    explanation = pd.DataFrame({

        "Feature": selected_features,

        "Feature_Value": sample.iloc[0].values,

        "SHAP_Value": shap_values_sample
    })

    explanation["Abs_SHAP"] = np.abs(
        explanation["SHAP_Value"]
    )

    explanation = explanation.sort_values(
        "Abs_SHAP",
        ascending=False
    ).reset_index(drop=True)


    # ========================================
    # 4. TOP POSITIVE FACTORS
    # ========================================

    positive_factors = explanation[
        explanation["SHAP_Value"] > 0
    ].head(5)


    risk_factors = []

    for _, row in positive_factors.iterrows():

        risk_factors.append({

            "feature": row["Feature"],

            "value": float(
                row["Feature_Value"]
            ),

            "shap": float(
                round(row["SHAP_Value"], 3)
            ),

            "root_cause": get_root_cause(
                row["Feature"],
                row["Feature_Value"]
            ),

            "explanation": explain_feature(
                row["Feature"],
                row["Feature_Value"]
            ),

            "recommendation": get_recommendation(
                row["Feature"]
            )
        })

    # ============================================
    # CALCULATE RISK CONTRIBUTION
    # ============================================

    total_shap = sum(
        abs(factor["shap"])
        for factor in risk_factors
    )

    if total_shap > 0:

        for factor in risk_factors:

            factor["risk_contribution"] = round(
                (abs(factor["shap"]) / total_shap) * 100,
                2
            )

            del factor["shap"]

    # ========================================
    # 5. FINAL RESPONSE
    # ========================================

    overall_recommendation = get_overall_recommendation(
        risk_level,
        risk_factors
    )

    return {
        "bug_probability": float(
            round(probability * 100, 2)
        ),

        "risk_level": risk_level,

        "prediction": prediction,

        "risk_factors": risk_factors,

        "overall_recommendation": overall_recommendation
    }