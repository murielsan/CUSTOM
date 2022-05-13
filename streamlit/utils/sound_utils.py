import sys
from pathlib import Path, PurePath

from .communications import soundsQueue


class SoundHelper:

    # File format
    file_format = "wav"
    # Sounds folder
    sounds_folder_name = "sounds"
    # Sounds path
    sounds_path = PurePath(sys.path[0]).parent.joinpath(sounds_folder_name)
    # Filename template
    file_name = ""

    def __init__(self):
        # Max number of files to keep
        self.rotate_files = 50
        self.current_file = 0
        try:
            Path(self.sounds_path).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        finally:
            # Empty folder
            [f.unlink() for f in Path(self.sounds_path).glob("*") if f.is_file()]

    def save_sound(self, buffer):
        self.file_name = (
            f"{self.sounds_path}/sound{self.current_file}.{self.file_format}"
        )

        buffer.export(self.file_name, format=self.file_format)

        # Update count
        if (self.current_file + 1) >= self.rotate_files:
            self.current_file = 0
        else:
            self.current_file += 1

        # Put the file in the queue
        soundsQueue.put(self.file_name)

        return self.file_name
