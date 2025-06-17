import os
import csv
import shutil
from urllib.parse import urlparse, parse_qs

CSV_INPUT = 'video_titles.csv'
DEST_ROOT = 'false-movies'

def extract_video_id(url):
    try:
        parsed = urlparse(url)
        if parsed.hostname in ['www.youtube.com', 'youtube.com'] and parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        elif 'youtu.be' in parsed.hostname:
            return parsed.path.strip('/')
    except Exception as e:
        print(f"Błąd przy wyciąganiu ID z {url}: {e}")
    return None

def move_false_videos(csv_file, destination_root):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            zawiera = row['zawiera'].strip().lower()
            if zawiera == 'false':
                url = row['url']
                video_id = extract_video_id(url)
                if not video_id:
                    print(f"Nie udało się wyciągnąć ID z url: {url}")
                    continue

                original_path = row['ścieżka']
                # original_path = row['ścieżka'].replace('frames', 'movies', 1)
                if not os.path.isdir(original_path):
                    print(f"Ścieżka nie istnieje: {original_path}")
                    continue

                found = False
                for folder in os.listdir(original_path):
                    if folder.endswith(f"_{video_id}"):
                        source = os.path.join(original_path, folder)

                        try:
                            relative_path = os.path.relpath(original_path, start='frames')
                        except ValueError:
                            print(f"Ścieżka {original_path} nie znajduje się wewnątrz 'frames/'")
                            continue

                        dest_dir = os.path.join(destination_root, relative_path)
                        os.makedirs(dest_dir, exist_ok=True)

                        dest = os.path.join(dest_dir, folder)
                        print(f"Przenoszenie {source} → {dest}")
                        shutil.move(source, dest)
                        found = True
                        break

                if not found:
                    print(f"Nie znaleziono folderu kończącego się na _{video_id} w {original_path}")

if __name__ == "__main__":
    move_false_videos(CSV_INPUT, DEST_ROOT)
