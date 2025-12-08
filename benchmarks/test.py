import json

FILE = "captions_test2017_results.json"

try:
    print(f"Reading {FILE}...")
    with open(FILE, 'r') as f:
        data = json.load(f)

    print(f"✅ JSON Syntax is valid.")
    print(f"   Total entries: {len(data)}")

    # Check first item structure
    if isinstance(data, list) and "image_id" in data[0] and "caption" in data[0]:
        print(f"✅ Structure looks correct: List of Dictionaries.")
    else:
        print(f"❌ Structure FAIL. Must be a list of dicts with 'image_id' and 'caption'.")
        
    # Check types
    if isinstance(data[0]['image_id'], int):
        print(f"✅ image_id is Integer (Correct).")
    else:
        print(f"❌ image_id is {type(data[0]['image_id'])} (Must be int).")

    # Check for empty captions
    empty_caps = sum([1 for x in data if not x['caption'].strip()])
    if empty_caps > 0:
        print(f"⚠️ WARNING: Found {empty_caps} empty captions. This might crash CodaLab.")
    else:
        print(f"✅ No empty captions found.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: Your JSON file is corrupted.\n{e}")