#! /usr/bin/env python3
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import config_rona
from playsound import playsound
from gtts import gTTS

text2say = "Text to say"

tts = gTTS(text=text2say,
        lang="en",
        slow=False)

name = "name_of_file.mp3"
full_path = config_rona.path_audio_files + name
tts.save(full_path)
playsound(full_path, block=True)