"""Process all .db files recursively from a directory and save GameDataList to JSON.

Usage:
    python scripts/process_all_games.py [directory_path]

If no directory is provided, defaults to:
    data/raw/Vox Deorum/runs-tiny-4player
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.data_processing.v1.load_game_data import load_game_data
from scripts.data_processing.v1.models import GameDataList


def find_db_files(directory: Path) -> list[Path]:
    """Recursively find all .db files in the given directory."""
    db_files = list(directory.rglob("*.db"))
    return sorted(db_files)


def process_all_games(directory: Path, output_file: Path) -> None:
    """Process all .db files in directory and save to JSON."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    db_files = find_db_files(directory)
    
    if not db_files:
        print(f"No .db files found in {directory}")
        return

    print(f"Found {len(db_files)} database file(s)")
    
    games = []
    errors = []
    
    for i, db_file in enumerate(db_files, 1):
        print(f"[{i}/{len(db_files)}] Processing: {db_file.relative_to(directory)}")
        try:
            game_data = load_game_data(db_file)
            games.append(game_data)
            print(f"  ✓ Loaded game {game_data.id} (turn {game_data.turn})")
        except Exception as e:
            error_msg = f"Error processing {db_file}: {e}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
    
    # Create GameDataList
    game_data_list = GameDataList(games=games)
    
    # Save to JSON without indent
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(game_data_list.model_dump(), f, ensure_ascii=False, indent=None)
    
    print(f"\n✓ Successfully processed {len(games)} game(s)")
    print(f"✓ Saved to: {output_file}")
    
    if errors:
        print(f"\n⚠ {len(errors)} error(s) occurred:")
        for error in errors:
            print(f"  - {error}")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        # Default directory - .db files are in data/raw
        directory = Path("data/raw/Vox Deorum/runs-tiny-4player")
    
    output_file = Path("game_data_list.json")
    
    try:
        process_all_games(directory, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

