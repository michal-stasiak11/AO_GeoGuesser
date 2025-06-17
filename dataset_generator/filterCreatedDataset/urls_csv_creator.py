import os
import csv
from yt_dlp import YoutubeDL

ROOT_DIR = 'frames'
MERGED_FILE = 'mergedUrls.txt'
CSV_OUTPUT = 'video_titles.csv'

def find_movie_urls(root_dir):
    urls = []
    for root, dirs, files in os.walk(root_dir):
        if 'movieUrls.txt' in files:
            file_path = os.path.join(root, 'movieUrls.txt')
            print(f'Zapisuje url z {file_path}\n')
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    url = line.strip()
                    if url:
                        urls.append((url, root))
    return urls

def merge_urls(urls, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        print(f'Zapisuje wszystkie url d {output_file}\n')
        for url, path in urls:
            f.write(f"{url} | {path}\n")

def extract_video_title(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'force_generic_extractor': False,
        'cookiesfrombrowser': ('firefox', 'h5vvucb9.default-release')
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get('title')
        except Exception as e:
            print(f"Błąd przy pobieraniu tytułu z {url}: {e}")
            return None

def check_contains(title, path_parts):
    title_lower = title.lower()
    return any(part.lower() in title_lower for part in path_parts)

def process_merged_file(input_file, output_csv):
    processed_entries = set()

    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as existing:
            reader = csv.reader(existing)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    processed_entries.add((row[1], row[2]))

    with open(input_file, 'r', encoding='utf-8') as f, open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if os.stat(output_csv).st_size == 0:
            writer.writerow(['id', 'url', 'ścieżka', 'tytuł', 'zawiera'])

        for idx, line in enumerate(f):
            if '|' not in line:
                print(f'Nie znaleziono ścieżki dla {line}')
                continue
            url, path = map(str.strip, line.strip().split('|'))

            if (url, path) in processed_entries:
                print(f'Pominięto: {url} | {path}')
                continue

            print(f'Tworzenie wpisu do csv dla url: {url} , path: {path}')
            title = extract_video_title(url)
            if not title:
                print(f'Nie znaleziono tytułu dla: {url} , path: {path}')
                continue
            path_parts = [p for p in path.split(os.sep) if p not in ['frames', '']]
            zawiera = check_contains(title, path_parts)
            print(f'Zapisuje: {url} , path: {path}, title: {title}, zawiera: {zawiera}')
            writer.writerow([idx + 1, url, path, title, zawiera])
            processed_entries.add((url, path))


if __name__ == "__main__":
    urls = find_movie_urls(ROOT_DIR)
    merge_urls(urls, MERGED_FILE)
    process_merged_file(MERGED_FILE, CSV_OUTPUT)
    print("Plik CSV zapisany jako:", CSV_OUTPUT)
