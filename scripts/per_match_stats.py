import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Default location for the Türkiye Süper Lig player data. This can be overridden
# via the --data-path CLI argument.
DATA_PATH = Path(__file__).resolve().parent.parent / "YSA_Proje_Veri"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed_data"

ROLE_CONFIGS = {
    "goalkeeper": {
        "positions": ["GK"],
        "metrics": [
            "Performance_CrdY",
            "Performance_CrdR",
            "Performance_Gls",
            "Performance_Ast",
            "Per90_xG",
        ],
    },
    "defender": {
        "positions": ["DF"],
        "metrics": [
            "Performance_Gls",
            "Performance_Ast",
            "Performance_G-PK",
            "Expected_npxG",
            "Expected_xAG",
            "Progression_PrgC",
            "Progression_PrgP",
            "Progression_PrgR",
        ],
    },
    "midfielder": {
        "positions": ["MF"],
        "metrics": [
            "Performance_Gls",
            "Performance_Ast",
            "Performance_G+A",
            "Performance_G-PK",
            "Expected_xAG",
            "Expected_npxG+xAG",
            "Progression_PrgP",
            "Progression_PrgR",
        ],
    },
    "forward": {
        "positions": ["FW"],
        "metrics": [
            "Performance_Gls",
            "Performance_Ast",
            "Performance_G+A",
            "Performance_G-PK",
            "Expected_xG",
            "Expected_npxG",
            "Expected_npxG+xAG",
            "Per90_xG",
        ],
    },
}


def _clean_category(value: str) -> str:
    value = value.replace("\ufeff", "").strip()
    if not value or value.startswith("Unnamed"):
        return ""
    return value


def _merge_headers(category_row: List[str], header_row: List[str]) -> List[str]:
    headers: List[str] = []
    current_category = ""
    for category, name in zip(category_row, header_row):
        category_clean = _clean_category(category)
        column_name = (name or "").strip()

        if category_clean:
            current_category = category_clean
        if not column_name or column_name.startswith("Unnamed"):
            continue

        if current_category:
            if current_category == "Per 90 Minutes":
                headers.append(f"Per90_{column_name}")
            else:
                headers.append(f"{current_category}_{column_name}")
        else:
            headers.append(column_name)
    return headers


def _to_number(value: str) -> Optional[float]:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"nan", "matches"}:
        return None
    # Some rows include stray timestamps or text; ignore those.
    try:
        return float(cleaned)
    except ValueError:
        return None


def _read_file(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            category_row = next(reader)
            header_row = next(reader)
        except StopIteration:
            return []
        headers = _merge_headers(category_row, header_row)

        for row in reader:
            if not row or not row[0].strip():
                continue
            if row[0].strip().lower() in {"overall", "home/away", "süper lig"}:
                continue
            data = {key: value for key, value in zip(headers, row) if value}
            data["Team"] = path.stem.split(".")[0].split(" - ")[0]
            yield data


def load_players(path: Path) -> List[Dict[str, str]]:
    if path.is_dir():
        source_files = sorted(path.glob("*Sayfa1.csv"))
        players: List[Dict[str, str]] = []
        for file_path in source_files:
            players.extend(list(_read_file(file_path)))
        return players
    return list(_read_file(path))


def _matches_role(position: str, expected: List[str]) -> bool:
    normalized = {part.strip().upper() for part in position.split(",") if part.strip()}
    for pos in expected:
        if pos.upper() in normalized:
            return True
    return False


def build_per_match_rows(players: List[Dict[str, str]], role: str, config: Dict[str, List[str]]):
    role_rows = []
    for player in players:
        position = player.get("Pos") or player.get("Position") or ""
        if not position or not any(char.isalpha() for char in position):
            continue
        if not _matches_role(position, config["positions"]):
            continue

        mp_value = _to_number(player.get("MP") or player.get("Playing Time_MP"))
        if not mp_value or mp_value <= 0:
            continue

        output_row = {
            "Player": player.get("Player", ""),
            "Team": player.get("Team", ""),
            "Pos": position,
            "Matches": mp_value,
        }

        for metric in config["metrics"]:
            metric_value = _to_number(player.get(metric))
            column_name = f"{metric}_per_match"
            if metric_value is None:
                output_row[column_name] = ""
            else:
                output_row[column_name] = round(metric_value / mp_value, 3)
        role_rows.append(output_row)
    return role_rows


def write_role_csv(role: str, rows: List[Dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"turkey_{role}_per_match.csv"
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = ["Player", "Team", "Pos", "Matches"]
    metric_keys = [key for key in rows[0].keys() if key not in fieldnames]
    fieldnames.extend(metric_keys)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate per-match role-based stats for Türkiye Süper Lig players.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the Türkiye league player CSV file or folder containing *Sayfa1.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Destination folder for the generated per-match CSV files.",
    )
    args = parser.parse_args()

    players = load_players(args.data_path)
    if not players:
        raise SystemExit(f"No player data found at {args.data_path}")

    for role, config in ROLE_CONFIGS.items():
        rows = build_per_match_rows(players, role, config)
        path = write_role_csv(role, rows, args.output_dir)
        print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
