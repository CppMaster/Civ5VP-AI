

from pydantic import BaseModel


class Player(BaseModel):
    id: int
    civilization: str


class GameData(BaseModel):
    id: str
    turn: int
    victoryPlayerID: int
    victoryType: str
    players: list[Player]
    

class GameDataList(BaseModel):
    games: list[GameData]