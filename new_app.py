# new_app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from app_v1_3 import compute_combined_score
import shutil

app = FastAPI()



@app.get("/")
def read_root():
    return {"message": "Welcome to the ATS Score API. Use /ats-score/ endpoint to get started."}


@app.post("/ats-score/")
async def ats_score(cv: UploadFile = File(...), jd: UploadFile = File(...)):
    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_cv:
            shutil.copyfileobj(cv.file, tmp_cv)
            cv_path = tmp_cv.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp_jd:
            shutil.copyfileobj(jd.file, tmp_jd)
            jd_path = tmp_jd.name

        # Run your model scoring
        score = compute_combined_score(cv_path, jd_path)

        return {"ats_score": score}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
