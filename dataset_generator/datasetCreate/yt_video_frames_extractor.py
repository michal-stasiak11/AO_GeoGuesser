import os
import random
import re
import requests
import cv2
from yt_dlp import YoutubeDL
import time
from datasetCreate.locations import city_to_continent, city_to_country

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

def get_first_unused_youtube_video_url(query, used_urls):
    attempt = 0
    while True:
        attempt += 1
        search_url = f'https://www.youtube.com/results?search_query={query.replace(" ", "+")}&sp=EgIQAQ%253D%253D'
        response = requests.get(search_url, headers=HEADERS)
        if response.status_code != 200:
            print(f'Błąd pobierania strony YouTube dla zapytania "{query}"')
            time.sleep(1)
            continue

        html = response.text
        video_ids = re.findall(r'\"videoId\":\"([a-zA-Z0-9_-]{11})\"', html)
        if not video_ids:
            print(f'Nie znaleziono żadnego filmu dla "{query}"')
            return None

        for video_id in video_ids:
            url = f'https://www.youtube.com/watch?v={video_id}'
            if url not in used_urls:
                return url

        print(f'[{attempt}] Wszystkie znalezione URL-e były już użyte. Ponowne szukanie')
        time.sleep(1)


def download_video(url, download_folder='videos', cookies_file='cookies.txt'):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    with YoutubeDL({
        'quiet': True,
        'no_warnings': True,
        'cookiesfrombrowser': ('firefox', 'h5vvucb9.default-release')
        # 'cookiefile': cookies_file
    }) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        duration = info_dict.get('duration', 0)
        video_id = info_dict.get('id')
        formats = info_dict.get('formats', [])

    def select_best_video_only_format(formats):
        preferred_exts = ['mp4', 'mkv', 'avi']
        preferred_codecs = ['avc1', 'h264', 'vp9']
        compatible = [
            fmt for fmt in formats
            if fmt.get('vcodec') != 'none' and fmt.get('acodec') == 'none'
            and fmt.get('ext') in preferred_exts
            and any(codec in fmt.get('vcodec', '').lower() for codec in preferred_codecs)
        ]
        if compatible:
            return max(compatible, key=lambda f: f.get('height', 0))['format_id']

        video_only = [fmt for fmt in formats if fmt.get('vcodec') != 'none' and fmt.get('acodec') == 'none']
        return video_only[0]['format_id'] if video_only else 'bestvideo'

    selected_format_id = select_best_video_only_format(formats)

    if duration < 60:
        start_time = 0
        end_time = duration
    else:
        start_time = max((duration // 2) - 30, 0)
        end_time = min(start_time + 60, duration)

    ydl_opts = {
        'format': selected_format_id,
        'outtmpl': os.path.join(download_folder, f'{video_id}.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'download_ranges': lambda info_dict, ydl: [{'start_time': start_time, 'end_time': end_time}],
        # 'cookiefile': cookies_file
        'cookiesfrombrowser': ('firefox', 'h5vvucb9.default-release')
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)

    return filename, video_id

def extract_random_frames(video_path, output_folder, frames_count=20, continent='Unknown', country='Unknown', city='Unknown'):
    print(f'Odczytywanie pliku video: {video_path}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Nie można otworzyć pliku video: {video_path}')
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f'Brak klatek w pliku video: {video_path}')
        cap.release()
        return []

    actual_frames_to_extract = min(frames_count, frame_count)
    selected_frames = random.sample(range(frame_count), actual_frames_to_extract)
    selected_frames.sort()

    saved_files = []
    for i, frame_no in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(
                output_folder,
                f'{continent}_{country}_{city}_frame_{i+1}_frame{frame_no}.jpg'
            )
            cv2.imwrite(filename, frame)
            saved_files.append(filename)
    cap.release()
    return saved_files


def main():
    errorLogPath = 'errors.txt'
    if not os.path.exists(errorLogPath):
        with open(errorLogPath, 'w', encoding='utf-8') as f:
            pass
    while(True):
        for city in city_to_country:
            continent = city_to_continent.get(city, 'Unknown')
            country = city_to_country.get(city, 'Unknown')
            base_folder = os.path.join('frames', continent, country, city)
            os.makedirs(base_folder, exist_ok=True)

            url_log_path = os.path.join(base_folder, 'movieUrls.txt')
            if not os.path.exists(url_log_path):
                with open(url_log_path, 'w', encoding='utf-8') as f:
                    pass

            with open(url_log_path, 'r', encoding='utf-8') as f:
                used_urls = set(line.strip() for line in f.readlines())

            print(f'Pobieram nowy, nieużywany film dla {city}...')
            video_url = get_first_unused_youtube_video_url(city + ' countryside dashcam drive', used_urls)
            # video_url = get_first_unused_youtube_video_url(city + ' dashcam drive', used_urls)
            if video_url is None:
                continue
            print(f'Znaleziono: {video_url}')

            try:
                video_path, video_id = download_video(video_url)
            except Exception as e:
                print(f'Błąd pobierania video: {e}')
                with open(errorLogPath, 'a', encoding='utf-8') as error_file:
                    error_file.write(f'Miasto: {city}, URL: {video_url}, Błąd: {str(e)}\n')
                if 'ffmpeg exited with code' in str(e):
                    with open(url_log_path, 'a', encoding='utf-8') as f:
                        f.write(video_url + '\n')
                continue

            subfolder_number = len([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
            frame_folder = os.path.join(base_folder, f'filmik_{subfolder_number+1}_{video_id}')
            saved_frames = extract_random_frames(video_path, frame_folder, continent=continent, country=country, city=city)
            if saved_frames:
                print(f'Zapisano {len(saved_frames)} klatek w {frame_folder}')
                with open(url_log_path, 'a', encoding='utf-8') as f:
                    f.write(video_url + '\n')
            else:
                print(f'Nie udało się zapisać klatek dla {city}.')

            if os.path.exists(video_path):
                os.remove(video_path)


if __name__ == '__main__':
    main()
