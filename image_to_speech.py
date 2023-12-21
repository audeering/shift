import numpy as np
import subprocess
import cv2

from moviepy.editor import *
import soundfile

# yt-dlp https://www.youtube.com/watch?v=mEplIrAP-z4

# https://superuser.com/questions/583393/how-to-extract-subtitle-from-video-using-ffmpeg
from mimic3_tts.__main__ import (CommandLineInterfaceState,
                                 get_args,
                                 initialize_args,
                                 initialize_tts,
                                 # print_voices,
                                 # process_lines,
                                 shutdown_tts,
                                 OutputNaming,
                                 # process_line
                                 )

import csv
import json
import io
import typing
import wave
import time
import soundfile
import numpy as np
import re

from unidecode import unidecode

#___________________________________________________________________________________________________
#   VIDEO FROM IMAGE with CAPTIONS
#
# UPLOAD to: Simaviro: Documents General WORK PACKAGES WP1 ContentRepository ANBPR_ROMANIA TTSvideos
# __________________________________________________________________________________________________

# TO DONLOAD SRT for youtub
# yt-dlp --write-sub --sub-lang en --convert-subs "srt" https://www.youtube.com/watch?v=F1Ib7TAu7eg&list=PL4x2B6LSwFewdDvRnUTpBM7jkmpwouhPv&index=2

# _voice = 'en_US/vctk_low#p330'
# _voice = 'en_US/cmu-arctic_low#lnh' #en_US/vctk_low#p249'  # 'en_US/vctk_low#p282'
# _voice = ''en_US/vctk_low#p351''
# _voice = 'en_US/vctk_low#p351'  # avoid 318 it does the ghhhhhh
# _voice = 'en_US/m-ailabs_low#judy_bieber'  # Nice voice for ('Arta culinara romaneasca - Groza Irina [phIF0NxgwlQ].mkv' 'Arta culinara romaneasca - Groza Irina [phIF0NxgwlQ].en-GB.srt'),
# _voice = 'en_UK/apope_low'
# _voice = 'en_US/m-ailabs_low#mary_ann'
# _voice = 'en_US/vctk_low#p351'
# _voice = 'en_US/hifi-tts_low#92'
# voice_str = f'_{_voice.replace("/", "")}'



with open('voices.json', 'r') as f:
    VOICES = json.load(f)['voices']


def process_line(
    line: str,
    state: CommandLineInterfaceState,
    line_id: str = "",
    line_voice: typing.Optional[str] = None,
):
    assert state.result_queue is not None
    args = state.args

    if state.tts:
        # Local TTS
        from mimic3_tts import SSMLSpeaker

        assert state.tts is not None

        args = state.args

        if line_voice:
            if line_voice.startswith("#"):
                # Same voice, but different speaker
                state.tts.speaker = line_voice[1:]
            else:
                # Different voice
                state.tts.voice = line_voice

        if args.ssml:
            results = SSMLSpeaker(state.tts).speak(line)
        else:
            state.tts.begin_utterance()

            # TODO: text language
            state.tts.speak_text(line)
            # If we only call speak_text only once then maybe after process line
            # we can take the output samples in a list

            results = state.tts.end_utterance()

    return results


def process_lines(state: CommandLineInterfaceState, wav_path=None):

    args = state.args

    result_idx = 0
    for line in state.texts:
        line_voice: typing.Optional[str] = None
        line_id = ""
        line = line.strip()

        if args.output_naming == OutputNaming.ID:
            # Line has the format id|text instead of just text
            with io.StringIO(line) as line_io:
                reader = csv.reader(line_io, delimiter=args.csv_delimiter)
                row = next(reader)
                line_id, line = row[0], row[-1]
                if args.csv_voice:
                    line_voice = row[1]

        # process_line(line, state, line_id=line_id, line_voice=line_voice)
        # result_idx += 1
        generator_4 = process_line(line, state, line_id=line_id, line_voice=line_voice)
        resK = next(generator_4)  # this generator_4 is list of 1 lement has to call next to get AudioResult hence audiobuffer

        result_idx += 1

        # import pdb; pdb.set_trace()

    # print('\nARRive at All2 Audio writing\n\n\n\n')

    # -------------------------------------------------------------------------
    with io.BytesIO() as wav_io:

        wav_file_play: wave.Wave_write = wave.open(wav_io, "wb")
        with wav_file_play:
            wav_file_play.setframerate(state.sample_rate_hz)
            wav_file_play.setsampwidth(state.sample_width_bytes)
            wav_file_play.setnchannels(state.num_channels)
            # wav_file_play.writeframes(state.all_audio)  # WRITES state.all_audio
            wav_file_play.writeframes(resK.audio_bytes)  # state.all_audio)

        # print(wav_io.getvalue(), len(wav_io.getvalue()), np.array(wav_io.getvalue()).shape, 'IN wav_file wirte')

        with open(wav_path, 'wb') as wav_file:  # .wav
            wav_file.write(wav_io.getvalue())
            wav_file.seek(0)

# image/descriptions provided by other SHIFT tool or Human curator


STATIC_FRAME = 'assets/image_from_T31.jpg'

DESCRIPTIONS = [
    [('Inference 1. This is an old photograph of a city street with people walking on it. '
         'There are buildings on either side of the street, and a horse and carriage in the foreground. '
         'The people are dressed in Victorian clothing. The photograph is black and white.'),
     'en_US/m-ailabs_low#judy_bieber'
     ],
    [('Inference 2. The image shows a street scene in an old city, with people walking on the sidewalk and carriages driving by.'
         'The buildings on either side of the street are tall and ornate, with balconies and ornate facades.'
         'There is a horse-drawn carriage in the foreground, and a group of people standing on the sidewalks '
         'in front of buildings. The sky is overcast, and there is a sense of movement and activity in the scene.'),
    'en_US/vctk_low#p318', #'en_US/hifi-tts_low#92'
     ],
    [('Inference 3. This is an old photograph of a city street with people walking on it.'
                         'There are buildings on either side of the street, including a church in the background.'
                         'The people in the photograph are dressed in Victorian Era clothing, involving top hats and long coats.'
                         'There are also horse-drawn carriages on the street. The overall atmosphere of the photograph is one of a bustling city.'
                         'source National Association of Librarians and Public Libraries of Romania.'),
    'en_US/cmu-arctic_low#lnh'
     ]
]

SILENT_VIDEO = '_silent_video.mp4'
AUDIO_TRACK = '_audio_track.wav'
OUT_FILE = f'IM2VID.mp4'

# SILENT CLIP
# ==
clip_silent = ImageClip(STATIC_FRAME).set_duration(5)  # as long as the audio - TTS first
clip_silent.write_videofile(SILENT_VIDEO, fps=24)

total = [np.zeros(22050)]
for _text_, _voice in DESCRIPTIONS:

    voice_str = f'_{_voice.replace("/", "")}'

    _text_ = _text_.replace('Knjaževac', 'Kneeazevachy')
    _text_ = _text_.replace('fate', 'feyta')
    _text_ = _text_.replace('i.e. ', 'that is ')
    _text_ = _text_.replace(' cm ', ' centimeter ')
    _text_ = _text_.replace('.', ',').replace('sculpture', 'Skullptur').replace('figure', 'feegurr')
    _text_ = unidecode(_text_)

    print(_text_)

    # rate = VOICES[_voice]['rate']
    # rate = min(max(.77, len(_text_) / 46), .841) #1.44)  # 1.24 for bieber
    rate = .77 + .4 * np.random.rand()
    print(rate)
    volume = int(40 * np.random.rand() + 58)

    _text_ = re.sub(r"""
       [,.;@#?!&$]+  # Accept one or more copies of punctuation
       \ *           # plus zero or more copies of a space,
       [,.;@#?!&$ ]*  #  --- perhaps even more punctuation plus spaces
       """,
       ", ",          # and replace it with a single space
       _text_, flags=re.VERBOSE)
    _text_ = _text_[:-2] + '.'

    # print('____\n', _text_, '\n__')
    txt = ('<speak>'
            f'<prosody volume=\'{volume}\'>'
            f'<prosody rate=\'{rate}\'>'
            f'<voice name=\'{_voice}\'>'
            f'<s>{_text_}</s>'
            '</voice>'
            '</prosody>'
            '</prosody>'
        '</speak>')
    args = get_args()
    args.ssml = True
    args.text = [txt]  # ['aa', 'bb'] #txt
    args.interactive = False

    # args.noise_scale = .667 #VOICES[_voice]['noise_scale']
    # args.noise_w = .00 # VOICES[_voice]['noise_w']

    state = CommandLineInterfaceState(args=args)
    initialize_args(state)
    initialize_tts(state)
    process_lines(state, wav_path='_tmp.wav')
    shutdown_tts(state)
    time.sleep(1.7)  # some wav files don't have time to be written

    # # predict emotion of synthesized speech

    x, fs = soundfile.read('_tmp.wav')
    print(fs)

    total.append(x)
    total.append(np.zeros(fs))  # 1s silence

total = np.concatenate(total)
soundfile.write(AUDIO_TRACK, total, 22050)
subprocess.call(
    ["ffmpeg",
        "-y",
        "-i",
        SILENT_VIDEO,
        "-i",
        AUDIO_TRACK,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        " 1:a:0",
        OUT_FILE])