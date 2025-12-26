

from pydantic import BaseModel


class TurnData(BaseModel):
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
    policy_branches: dict[str, list[str]] = {}


class Player(BaseModel):
    id: int
    civilization: str
    turn_data_list: list[TurnData] = []


class GameData(BaseModel):
    id: str
    turn: int
    victoryPlayerID: int
    victoryType: str
    players: list[Player]


class GameDataList(BaseModel):
    games: list[GameData]
