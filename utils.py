import os
import json
import gdown
import requests

def list_json_files(directory):
  json_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith(".json"):
        json_files.append(os.path.join(root, file))
    print(f"Found {len(json_files)} JSON files in {directory}:")
    print(json_files)
  return json_files

def data_count_json_files(directory):
  count = 0
  for i, file in enumerate(directory):
    with open(file, 'r') as fp:
        count = 0
        for line in fp:
          count += 1
    if i == 1:
      print(f"{count} user review")
    else:
      print(f"{count} products")
    count = 0


def list_datasets(directory,row_count:int=2,mode:int=0):
  pprint = lambda x: print(json.dumps(x, indent=2)) if isinstance(x, dict) else print(x)

  # Validate inputs
  if not isinstance(directory, (list, tuple)) or len(directory) < 2:
    print("`directory` must be a list/tuple with at least two file paths")
    return

  if mode == 0:
    file_to_open = directory[0]
  elif mode == 1:
    file_to_open = directory[1]
  else:
    print("Mode can be 0 or 1")
    return

  # Print up to `row_count` rows (if row_count <= 0, treat as 1)
  try:
    rows_to_show = max(1, int(row_count))
  except Exception:
    rows_to_show = 1

  shown = 0
  with open(file_to_open, 'r', encoding='utf-8') as fp:
    for line in fp:
      if shown >= rows_to_show:
        break
      line = line.strip()
      if not line:
        continue
      try:
        parsed = json.loads(line)
        pprint(parsed)
      except Exception:
        # If the line isn't JSON, print raw
        pprint(line)
      shown += 1

def read_datasets(file_path):
  with open(file_path[0], 'r') as fp:
    reviews = []
    for line in fp:
        reviews.append(json.loads(line.strip()))

  with open(file_path[1], 'r') as fp:
      meta_product_review = []
      for line in fp:
          meta_product_review.append(json.loads(line.strip()))
  print(f"Total reviews: {len(reviews)}")
  print(f"Total products: {len(meta_product_review)}")
  return reviews,meta_product_review

def drive_downloader(file_id, destination):
  gdown.download(id=file_id, output=destination, quiet=False)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)