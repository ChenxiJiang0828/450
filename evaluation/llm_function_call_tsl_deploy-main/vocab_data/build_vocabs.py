# -*- coding: utf-8 -*-
from collections import OrderedDict
from pathlib import Path

BASE_DIR = Path("./vocab_data")
MOVIE_FILE = BASE_DIR / "movie.txt"
SERIES_FILE = BASE_DIR / "series.txt"
SONG_FILE = BASE_DIR / "song.txt"
VIDEO_OUTPUT = BASE_DIR / "vocab_video.txt"
SONG_OUTPUT = BASE_DIR / "vocab_song.txt"


def read_items(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def unique_keep_order(items: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(items))


def sort_by_length_desc(items: list[str]) -> list[str]:
    return sorted(items, key=lambda x: (-len(x), x))


def write_items(path: Path, items: list[str]) -> None:
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def main() -> None:
    movie_items = read_items(MOVIE_FILE)
    series_items = read_items(SERIES_FILE)
    song_items = read_items(SONG_FILE)

    video_items = unique_keep_order(movie_items + series_items)
    song_items = unique_keep_order(song_items)

    sorted_video_items = sort_by_length_desc(video_items)
    sorted_song_items = sort_by_length_desc(song_items)

    write_items(VIDEO_OUTPUT, sorted_video_items)
    write_items(SONG_OUTPUT, sorted_song_items)

    print(f"video: input={len(movie_items) + len(series_items)}, unique={len(sorted_video_items)}, output={VIDEO_OUTPUT}")
    print(f"song: input={len(song_items)}, unique={len(sorted_song_items)}, output={SONG_OUTPUT}")


if __name__ == "__main__":
    main()
