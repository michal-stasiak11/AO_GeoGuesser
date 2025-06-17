import csv

INPUT_CSV = 'video_titles.csv'
OUTPUT_CSV = 'video_titles_false_only.csv'

def filter_false_entries(input_csv, output_csv):
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            if row['zawiera'].strip().lower() == 'false':
                writer.writerow(row)

    print(f"Zapisano wpisy z zawiera=False do: {output_csv}")

if __name__ == "__main__":
    filter_false_entries(INPUT_CSV, OUTPUT_CSV)
