import csv
from collections import defaultdict
from pathlib import Path

csv_file = 'video_titles.csv'

country_counts = defaultdict(lambda: defaultdict(int))

with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('zawiera', '').strip().lower() != 'true':
            continue

        path = Path(row['ścieżka'])
        try:
            continent = path.parts[1] 
            country = path.parts[2]   
            country_counts[continent][country] += 1
        except IndexError:
            print(f"Niepoprawna ścieżka: {row['ścieżka']}")

result = []
for continent, countries in country_counts.items():
    for country, count in countries.items():
        result.append((continent, country, count))

result.sort(key=lambda x: (x[0], -x[2]))

output_file = 'countries_with_movies_count.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"{'Kontynent':10} | {'Kraj':15} | {'Liczba filmików'}\n")
    f.write("-" * 45 + "\n")
    for continent, country, count in result:
        f.write(f"{continent:10} | {country:15} | {count}\n")

print(f"Wyniki zapisano w pliku: {output_file}")
