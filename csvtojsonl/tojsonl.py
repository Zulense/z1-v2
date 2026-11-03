import csv
import json

def convert_csv_to_jsonl(csv_filepath, jsonl_filepath):
    with open(csv_filepath, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        with open(jsonl_filepath, mode='w', encoding='utf-8') as jsonl_file:
            for row in reader:
                # Convert numeric fields to appropriate types (float/int)
                try:
                    row['aesthetic score'] = float(row['aesthetic score'])
                    row['motion score'] = float(row['motion score'])
                    row['temporal consistency score'] = float(row['temporal consistency score'])
                    row['frame'] = int(row['frame'])
                    row['fps'] = float(row['fps'])
                    row['seconds'] = float(row['seconds'])
                except ValueError:
                    # Fallback to string if a value is missing or corrupted
                    pass 
                
                # Dump the row dictionary to a JSON string and write as a new line
                jsonl_file.write(json.dumps(row) + '\n')

# Execute the conversion
convert_csv_to_jsonl('./dataset.csv', './dataset.jsonl')
print("Conversion complete!")