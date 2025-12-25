import sqlite3
import json
from pathlib import Path
import sys
from collections import defaultdict

from scripts.data_processing.v1.models import GameData, Player, TurnData


DB_PATH = Path(
    "data/raw/Vox Deorum/runs-tiny-4player/none-strategist-013/"
    "c25c5079-d85a-430f-a2ec-85d85630ad1f_1765137881454.db"
)


def load_game_data(db_path: Path) -> GameData:
    """Load game data from SQLite database and return GameData object."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Load GameMetadata (key-value pairs)
    metadata = {}
    for row in cur.execute("SELECT Key, Value FROM GameMetadata"):
        metadata[row["Key"]] = row["Value"]

    # Extract required fields from metadata
    game_id = metadata["gameId"]
    turn = int(metadata["turn"])
    victory_player_id = int(float(metadata["victoryPlayerID"]))  # Handle float like "2.0"
    victory_type = metadata["victoryType"]

    # Load PlayerInformations and create a mapping
    player_info_map = {}
    for row in cur.execute("SELECT Key, ID, Civilization, IsMajor FROM PlayerInformations"):
        if row["IsMajor"] == 1:
            player_key = int(row["Key"])
            player_info_map[player_key] = {
                "id": int(row["ID"]),
                "civilization": row["Civilization"]
            }

    # Load PlayerSummaries and group by player Key
    player_turn_data = defaultdict(list)
    for row in cur.execute("""
        SELECT Key, Turn, Score, Cities, Population, Territory, Gold, 
               GoldPerTurn, HappinessPercentage, SciencePerTurn, CulturePerTurn, 
               FaithPerTurn, TourismPerTurn, Technologies, PolicyBranches
        FROM PlayerSummaries
        ORDER BY Key, Turn
    """):
        player_key = int(row["Key"])
        if player_key not in player_info_map:
            continue  # Skip non-major players
        
        # Parse PolicyBranches (JSON string or empty)
        policy_branches = {}
        if row["PolicyBranches"]:
            try:
                policy_branches = json.loads(row["PolicyBranches"])
            except (json.JSONDecodeError, TypeError):
                policy_branches = {}
        
        # Create TurnData object
        turn_data = TurnData(
            score=int(row["Score"]) if row["Score"] is not None else 0,
            cities=int(row["Cities"]) if row["Cities"] is not None else 0,
            population=int(row["Population"]) if row["Population"] is not None else 0,
            territory=int(row["Territory"]) if row["Territory"] is not None else 0,
            gold=int(row["Gold"]) if row["Gold"] is not None else 0,
            gold_per_turn=float(row["GoldPerTurn"]) if row["GoldPerTurn"] is not None else 0.0,
            happiness_percentage=int(row["HappinessPercentage"]) if row["HappinessPercentage"] is not None else 0,
            science_per_turn=int(row["SciencePerTurn"]) if row["SciencePerTurn"] is not None else 0,
            culture_per_turn=int(row["CulturePerTurn"]) if row["CulturePerTurn"] is not None else 0,
            faith_per_turn=int(row["FaithPerTurn"]) if row["FaithPerTurn"] is not None else 0,
            tourism_per_turn=int(row["TourismPerTurn"]) if row["TourismPerTurn"] is not None else 0,
            technologies=int(row["Technologies"]) if row["Technologies"] is not None else 0,
            policy_branches=policy_branches
        )
        
        player_turn_data[player_key].append(turn_data)

    con.close()

    # Create Player objects with turn_data_list
    players = []
    for player_key, info in player_info_map.items():
        players.append(Player(
            id=info["id"],
            civilization=info["civilization"],
            turn_data_list=player_turn_data.get(player_key, [])
        ))

    # Create and return GameData object
    return GameData(
        id=game_id,
        turn=turn,
        victoryPlayerID=victory_player_id,
        victoryType=victory_type,
        players=players
    )


def main() -> None:
    """Main entry point."""
    try:
        game_data = load_game_data(DB_PATH)
        print("GameData loaded successfully:")
        print(f"  ID: {game_data.id}")
        print(f"  Turn: {game_data.turn}")
        print(f"  Victory Player ID: {game_data.victoryPlayerID}")
        print(f"  Victory Type: {game_data.victoryType}")
        print(f"  Players :({len(game_data.players)})")
        for player in game_data.players:
            print(f"    - ID: {player.id}, Civilization: {player.civilization}, Turns: {len(player.turn_data_list)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

