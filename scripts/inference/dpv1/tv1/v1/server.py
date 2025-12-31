import torch
import joblib
from fastapi import FastAPI

from scripts.training.dpv1.v1.train_full_pipeline import WinPredictionNet, PreprocessingPipeline
from scripts.inference.dpv1.tv1.v1.models import InputModel, OutputModel
from scripts.inference.dpv1.tv1.v1.infer_model import infer


app = FastAPI()

pipeline = joblib.load("data\\models\\dpv1\\v1\\preprocessing_pipeline.joblib")
model_data = torch.load("data\\models\\dpv1\\v1\\win_prediction_model.pt")
model = WinPredictionNet(input_size=model_data["input_size"])
model.load_state_dict(model_data["model_state_dict"])
model.eval()


@app.post("/infer")
async def infer_model(input_model: InputModel) -> OutputModel:

    output = infer(input_model, pipeline, model)
    return output
