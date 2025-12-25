import sqlite3
from pathlib import Path
import sys


from scripts.data_processing.v1.models import GameData, Player


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
    victory_player_id = int(metadata["victoryPlayerID"])  # Handle float like "2.0"
    victory_type = metadata["victoryType"]

    # Load PlayerInformations
    players = []
    for row in cur.execute("SELECT ID, Civilization FROM PlayerInformations"):
        if row["IsMajor"] == 0:
            continue
        players.append(Player(
            id=int(row["Key"]),
            civilization=row["Civilization"]
        ))

    con.close()

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
        print(f"  Players ({len(game_data.players)}):")
        for player in game_data.players:
            print(f"    - ID: {player.id}, Civilization: {player.civilization}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

