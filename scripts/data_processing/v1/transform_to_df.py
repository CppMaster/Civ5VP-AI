import json
from pathlib import Path
import pandas as pd
from scripts.data_processing.v1.models import GameDataList


def load_game_data_list(json_path: Path) -> GameDataList:
    """Load GameDataList from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return GameDataList(**data)


def find_last_index_turn_without_policies(turn_data_list) -> int:
    """Find the last TurnData entry (chronologically) where policy_branches is empty."""
    last_turn_index_without_policies = None
    for index, turn_data in enumerate(turn_data_list):
        if not turn_data.policy_branches:  # Empty dict or None
            last_turn_index_without_policies = index
        else:
            return last_turn_index_without_policies
    return last_turn_index_without_policies


def create_player_dataframe(game_data_list: GameDataList) -> pd.DataFrame:
    """Create a DataFrame with player information and statistics from the last turn without policy_branches."""
    rows = []
    
    for game in game_data_list.games:
        for player in game.players:
            # Find the last turn without policy_branches
            last_turn_index_without_policies = find_last_index_turn_without_policies(player.turn_data_list)
            last_turn_data = player.turn_data_list[last_turn_index_without_policies]
            chosen_ancient_policy = list(player.turn_data_list[last_turn_index_without_policies + 1].policy_branches.keys())[0]
            
            # Determine if player won
            won = player.id - 1 == game.victoryPlayerID
            
            # Create row with player info and turn statistics
            row = {
                'game_id': game.id,
                'player_id': player.id,
                'civilization': player.civilization,
                'won': won,
                'victory_type': game.victoryType,
                'score': last_turn_data.score,
                'cities': last_turn_data.cities,
                'population': last_turn_data.population,
                'territory': last_turn_data.territory,
                'gold': last_turn_data.gold,
                'gold_per_turn': last_turn_data.gold_per_turn,
                'happiness_percentage': last_turn_data.happiness_percentage,
                'science_per_turn': last_turn_data.science_per_turn,
                'culture_per_turn': last_turn_data.culture_per_turn,
                'faith_per_turn': last_turn_data.faith_per_turn,
                'tourism_per_turn': last_turn_data.tourism_per_turn,
                'technologies': last_turn_data.technologies,
                'chosen_ancient_policy': chosen_ancient_policy
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    """Main entry point."""
    json_path = Path("data/processed/game_data_list.json")
    
    # Load the game data list
    print(f"Loading game data from {json_path}...")
    game_data_list = load_game_data_list(json_path)
    print(f"Loaded {len(game_data_list.games)} games")
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = create_player_dataframe(game_data_list)
    print(f"Created DataFrame with {len(df)} rows")
    print(f"\nDataFrame columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    df.to_csv("data/processed/game_data.csv")


if __name__ == "__main__":
    df = main()

