import json
import re

input_file = "pins_details.jsonl"
output_file = "updated_file.jsonl"

# Function to update JSON objects
def update_json(obj):
    # Remove exact release date, keep only the year
    if "release_date" in obj and isinstance(obj["release_date"], str):
        match = re.search(r"\d{4}", obj["release_date"])  # Extract year
        obj["release_date"] = match.group(0) if match else obj["release_date"]
    
    # Rename "Edition" field to a simpler format if necessary
    if "edition" in obj and isinstance(obj["edition"], str):
        obj["edition"] = re.sub(r"Limited Edition\s+", "LE ", obj["edition"])  # Convert format

    return obj

# Process the JSONL file line by line
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        obj = json.loads(line.strip())  # Parse each line as JSON
        updated_obj = update_json(obj)  # Apply transformations
        outfile.write(json.dumps(updated_obj) + "\n")  # Write updated JSON back

print("JSONL file updated successfully!")
