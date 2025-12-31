from typing import Literal
from pydantic import BaseModel


class InputModel(BaseModel):
    """
    Summary of a player, when he has to choose an ancient policy tree
    """
    civilization: Literal[
      "America",
      "Arabia",
      "Assyria",
      "Austria",
      "Babylon",
      "Brazil",
      "Byzantium",
      "Carthage",
      "China",
      "Denmark",
      "Egypt",
      "England",
      "Ethiopia",
      "France",
      "Germany",
      "Greece",
      "India",
      "Indonesia",
      "Japan",
      "Korea",
      "Mongolia",
      "Morocco",
      "Persia",
      "Poland",
      "Polynesia",
      "Portugal",
      "Rome",
      "Russia",
      "Siam",
      "Songhai",
      "Spain",
      "Sweden",
      "The Aztecs",
      "The Celts",
      "The Huns",
      "The Inca",
      "The Iroquois",
      "The Maya",
      "The Netherlands",
      "The Ottomans",
      "The Shoshone",
      "The Zulus",
      "Venice"
    ]
    score: int
    cities: int
    population: int
    territory: int
    gold: int
    gold_per_turn: float
    happiness_percentage: int
    science_per_turn: int
    culture_per_turn: int
    faith_per_turn: int
    tourism_per_turn: int
    technologies: int


class OutputModel(BaseModel):
    """
    Scores and chosen acient policy tree 
    """
    tradition_score: float
    progress_score: float
    authority_score: float
    predicted_ancient_policy: Literal["Tradition", "Progress", "Authority"]
