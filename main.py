# ============================================================
# EXPERIMENT 5
# MODEL DEPLOYMENT AS AN API
# TASK 2 - FASTAPI REST API
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import time


# ============================================================
# 1. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CKD Prediction API",
    description="Machine Learning API for CKD prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

try:

    model = joblib.load("ckd_model.pkl")

    model_info = joblib.load("model_info.pkl")

    model_loaded = True

except Exception as e:

    model = None

    model_info = {}

    model_loaded = False

    print("Error loading model:", e)


# ============================================================
# 3. API REQUEST COUNTERS
# ============================================================

total_requests = 0

successful_predictions = 0

failed_requests = 0

total_response_time = 0.0


# ============================================================
# 4. INPUT DATA SCHEMA
# ============================================================

class PatientData(BaseModel):

    age: float = Field(
        ...,
        description="Patient age in years",
        example=48
    )

    bp: float = Field(
        ...,
        description="Blood pressure",
        example=80
    )

    sg: float = Field(
        ...,
        description="Specific gravity",
        example=1.020
    )

    al: float = Field(
        ...,
        description="Albumin level",
        example=1
    )

    bgr: float = Field(
        ...,
        description="Blood glucose random",
        example=121
    )

    bu: float = Field(
        ...,
        description="Blood urea",
        example=36
    )

    sc: float = Field(
        ...,
        description="Serum creatinine",
        example=1.2
    )

    hemo: float = Field(
        ...,
        description="Hemoglobin",
        example=15.4
    )

    htn: str = Field(
        ...,
        description="Hypertension: yes/no",
        example="no"
    )

    dm: str = Field(
        ...,
        description="Diabetes mellitus: yes/no",
        example="no"
    )


# ============================================================
# 5. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "service": "CKD Prediction API",

        "message":
            "Machine Learning API is running",

        "version":
            "1.0.0",

        "documentation":
            "/docs",

        "status":
            "active"

    }


# ============================================================
# 6. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_loaded":
            model_loaded

    }


# ============================================================
# 7. READY ENDPOINT
# ============================================================

@app.get("/ready")
def ready():

    if model_loaded:

        return {

            "ready": True,

            "message":
                "API is ready to accept predictions"

        }

    return {

        "ready": False,

        "message":
            "Model is not loaded"

    }


# ============================================================
# 8. MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_information():

    return {

        "model_name":
            model_info.get(
                "model_name",
                "Unknown"
            ),

        "model_version":
            model_info.get(
                "model_version",
                "Unknown"
            ),

        "algorithm":
            model_info.get(
                "algorithm",
                "Unknown"
            ),

        "accuracy":
            model_info.get(
                "accuracy_percentage",
                "Unknown"
            ),

        "number_of_features":
            model_info.get(
                "number_of_features",
                "Unknown"
            ),

        "features":
            model_info.get(
                "features",
                []
            )

    }


# ============================================================
# 9. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    global total_requests
    global successful_predictions
    global failed_requests
    global total_response_time

    start_time = time.time()

    total_requests += 1

    try:

        # ----------------------------------------------------
        # Convert API input into DataFrame
        # ----------------------------------------------------

        input_data = {

            "age": [patient.age],

            "bp": [patient.bp],

            "sg": [patient.sg],

            "al": [patient.al],

            "bgr": [patient.bgr],

            "bu": [patient.bu],

            "sc": [patient.sc],

            "hemo": [patient.hemo],

            "htn": [patient.htn],

            "dm": [patient.dm]

        }

        input_df = pd.DataFrame(input_data)


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]


        # ----------------------------------------------------
        # Prediction label
        # ----------------------------------------------------

        if int(prediction) == 1:

            result = "CKD"

        else:

            result = "Not CKD"


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]

            probability = round(
                float(
                    max(probabilities)
                ),
                4
            )


        # ----------------------------------------------------
        # Update metrics
        # ----------------------------------------------------

        successful_predictions += 1

        response_time = (
            time.time() - start_time
        )

        total_response_time += response_time


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {

            "success": True,

            "prediction": result,

            "prediction_code":
                int(prediction),

            "model_probability":
                probability,

            "response_time_seconds":
                round(
                    response_time,
                    4
                ),

            "message":
                "Prediction generated successfully. "
                "This is a model output and not a medical diagnosis."

        }


    except Exception as e:

        failed_requests += 1

        response_time = (
            time.time() - start_time
        )

        total_response_time += response_time

        return {

            "success": False,

            "error":
                str(e),

            "message":
                "Prediction could not be generated."

        }


# ============================================================
# 10. METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def metrics():

    if total_requests > 0:

        average_response_time = (
            total_response_time
            / total_requests
        )

    else:

        average_response_time = 0


    return {

        "total_requests":
            total_requests,

        "successful_predictions":
            successful_predictions,

        "failed_requests":
            failed_requests,

        "average_response_time_seconds":
            round(
                average_response_time,
                4
            )

    }


# ============================================================
# END OF API
# ============================================================