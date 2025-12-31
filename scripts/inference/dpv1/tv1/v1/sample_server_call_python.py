import requests

from scripts.inference.dpv1.tv1.v1.models import InputModel


if __name__ == "__main__":
    response = requests.post("http://127.0.0.1:10000", json=InputModel(
        civilization="The Huns",
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
    ).model_dump())
    print(response.json())
