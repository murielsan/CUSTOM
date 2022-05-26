"""
sound utils
-----------

Functions useful for processing sounds
"""
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
    mode = None

    def __init__(self, mode="wav"):
        """Prepares the folder to store the files"""
        print(f"Initialize sound helper, {mode} mode")
        self.mode = mode

        if self.mode == "wav":
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

    def queue_sound(self, sound):
        """Generic function, decides what to do with received sound"""
        if self.mode == "wav":
            self.save_sound(sound)
        else:
            self.stream_sound(sound)

    def save_sound(self, buffer):
        """Stores and queue a pydub AudioSegment object as sound file"""
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

    def stream_sound(self, buffer):
        """Queue pydub AudioSegment file"""
        print(
            f"Buffer received ({buffer.duration_seconds}):\nChannels: {buffer.channels}\nSample Width: {buffer.sample_width}, \nFrame rate: {buffer.frame_rate}"
        )
        soundsQueue.put(buffer.set_channels(1))  # Pass only 1 channel
