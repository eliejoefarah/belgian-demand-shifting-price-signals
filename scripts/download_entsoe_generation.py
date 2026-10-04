import json
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient


# Load API token from JSON file
TOKEN_FILE = Path("entsoe_tokens.json")

with TOKEN_FILE.open("r", encoding="utf-8") as f:
    config = json.load(f)

api_key = config["api_key"]

# Create ENTSO-E client
client = EntsoePandasClient(api_key=api_key)

# Query period
start = pd.Timestamp("2023-01-01", tz="Europe/Brussels")
end = pd.Timestamp("2024-01-01", tz="Europe/Brussels")

# Query generation mix
mix = client.query_generation(
    "BE",
    start=start,
    end=end,
    psr_type=None,
)

# Save results
Path("data").mkdir(parents=True, exist_ok=True)
mix.to_csv("data/entsoe_be_2023_generation.csv")

print("Generation data saved to data/entsoe_be_2023_generation.csv")
