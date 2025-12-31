import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


from scripts.training.dpv1.v1.train_full_pipeline import PreprocessingPipeline, ACTION_SPACE, WinPredictionNet
from scripts.inference.dpv1.tv1.v1.models import InputModel, OutputModel


def infer(input_model: InputModel, pipeline: PreprocessingPipeline, model: nn.Module) -> OutputModel:

    row = dict(input_model)
    input_data = pd.DataFrame([row])
    transformed_data = pipeline.transform(input_data)
    cloned_data = pd.concat([transformed_data]*3, ignore_index=True)
    cloned_data.loc[0, "chosen_ancient_policy_Tradition"] = 1
    cloned_data.loc[1, "chosen_ancient_policy_Progress"] = 1
    cloned_data.loc[2, "chosen_ancient_policy_Authority"] = 1

    input_tensor = torch.from_numpy(np.array(cloned_data).astype(np.float32))
    outputs = model(input_tensor)
    probabilities = outputs.cpu().detach().numpy()
    return OutputModel(
        tradition_score=probabilities[ACTION_SPACE.index("Tradition")],
        progress_score=probabilities[ACTION_SPACE.index("Progress")],
        authority_score=probabilities[ACTION_SPACE.index("Authority")],
        predicted_ancient_policy=ACTION_SPACE[np.argmax(probabilities)]
    )


if __name__ == "__main__":
    pipeline = joblib.load("data\\models\\dpv1\\v1\\preprocessing_pipeline.joblib")
    model_data = torch.load("data\\models\\dpv1\\v1\\win_prediction_model.pt")
    model = WinPredictionNet(input_size=model_data["input_size"])
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()
    sample = InputModel(
        civilization="The Maya",
        score=95,
        cities=1,
        population=5,
        territory=14,
        gold=56,
        gold_per_turn=7.0,
        happiness_percentage=100,
        science_per_turn=7,
        culture_per_turn=5,
        faith_per_turn=3,
        tourism_per_turn=0,
        technologies=3
    )
    output = infer(sample, pipeline, model)
    pass