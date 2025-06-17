# datasetCreate
* yt_video_frames_extractor.py - szuka filmików na YouTube na podstawie lokalizacji zdefiniowanych w pliku locations.py. Dla każdego filmiku pobiera kilka klatek (frames) i zapisuje je w odpowiedniej lokalizacji katalogowej (np. frames/Europe/Poland/Warszawa/...). Zapisuje również wykorzystane URL-e do dalszego użycia.

* yt_movies_downloader.py - pobiera pełne filmiki z YouTube wykorzystując wcześniej zapisane URL-e z yt_video_frames_extractor.py.

# filterCreatedDataset
Sprawdza, czy lokalizacja filmu (na podstawie struktury katalogów lub metadanych) jest zgodna z jego tytułem.

Filmik z katalogu Europe/Poland/Warszawa powinien zawierać w tytule słowa takie jak Europe, Poland lub Warszawa.

* mergedUrls.txt – zbiór wszystkich URL-i użytych w datasetcie.
* urls_csv_creator.py – tworzy CSV z informacjami o URL-ach i przypisanych lokalizacjach.
* filter_video_csv.py – tworzy nowy plik CSV zawierający tylko błędne URL
* extract_false_movies.py – na podstawie pliku CSV z błędnymi URL usuwa niepoprawne filmiki/klatki

# countDatasetMovies
* country_movies_conter.py - alicza liczbę pobranych filmów dla każdego kraju i sortuje dane po kontynencie i ilości filmików.

# wygenerowany dataset
https://drive.google.com/drive/folders/1rV315dzRW7APN6o12Ts0tW0PqgKcIorZ?usp=drive_link