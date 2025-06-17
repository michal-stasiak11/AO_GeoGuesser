import os
import re
from yt_dlp import YoutubeDL
import shutil

FRAMES_ROOT = 'frames'
MOVIES_ROOT = 'movies'

def download_video(url, download_folder='videos'):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    with YoutubeDL({
        'quiet': True,
        'no_warnings': True,
        'cookiesfrombrowser': ('firefox', 'h5vvucb9.default-release')
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
        'cookiesfrombrowser': ('firefox', 'h5vvucb9.default-release')
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)

    return filename, video_id

def extract_video_id_from_foldername(folder_name):
    match = re.search(r'_([\w-]{11})$', folder_name)
    return match.group(1) if match else None

def process_all_folders_with_video_ids(root):
    for dirpath, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            video_id = extract_video_id_from_foldername(dirname)
            if not video_id:
                continue

            folder_full_path = os.path.join(dirpath, dirname)
            relative_path = os.path.relpath(folder_full_path, FRAMES_ROOT)
            output_folder = os.path.join(MOVIES_ROOT, relative_path)

            output_filename = f"{dirname}_video.mp4"
            output_file = os.path.join(output_folder, output_filename)

            if os.path.exists(output_file):
                print(f"Film już istnieje: {output_file}")
                continue

            url = f"https://www.youtube.com/watch?v={video_id}"
            try:
                print(f"Pobieranie: {url}")
                os.makedirs(output_folder, exist_ok=True)
                temp_folder = 'temp_download'
                video_path, _ = download_video(url, temp_folder)
                shutil.move(video_path, output_file)
                shutil.rmtree(temp_folder, ignore_errors=True)
                print(f"Zapisano do: {output_file}")
            except Exception as e:
                print(f"Błąd pobierania filmu z {url}: {e}")

if __name__ == "__main__":
    process_all_folders_with_video_ids(FRAMES_ROOT)
