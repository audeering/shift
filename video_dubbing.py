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

# from pathlib import Path
# from config import VOICES
import csv
import json
import io
import os
import typing
import wave
import time
import audb
import audresample
import soundfile
import numpy as np
import re
import srt
import subprocess
import permissive_dict
import torch

from unidecode import unidecode

# ======================================== TTS
frame_tts = np.zeros((104, 1920, 3), dtype=np.uint8)
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (240, 74)  # w,h
fontScale              = 2
fontColor              = (255, 255, 255)
thickness              = 4
lineType               = 2
cv2.putText(frame_tts, 'TTS',
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    thickness,
    lineType)
# cv2.imshow('i', frame_tts); cv2.waitKey(); cv2.destroyAllWindows()
# ====================================== NATIVE VOICE
frame_orig = np.zeros((104, 1920, 3), dtype=np.uint8)
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (101, 74)  # w,h
fontScale              = 2
fontColor              = (255, 255, 255)
thickness              = 4
lineType               = 1000
cv2.putText(frame_orig, 'ORIGINAL VOICE',
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    thickness,
    lineType)
#___________________________________________________________________________________________________
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
_voice = 'en_US/hifi-tts_low#92'
voice_str = f'_{_voice.replace("/", "")}'



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



# ===========================================================
# ==
# MKV 2 MP4 - loadable by moviepy
# ffmpeg -y -i Distaff\ \[qVonBgRXcWU\].mkv -c copy -c:a aac Distaff_qVonBgRXcWU.mp4
video_file, srt_file = ['assets/Head_of_fortuna.mp4', 
                        'assets/head_of_fortuna_en.srt'
                    ]
vf = VideoFileClip(video_file)
# vf = vf.subclip(10, 14)

args = permissive_dict.PermissiveDict()

prefix = video_file[:-4]  # , file_type = video_file.split('.')  # I think prefix can even be different that vdieoname
audio_track = prefix + voice_str + '_.wav'
with open(srt_file, "r") as f:
    s = f.read()
subtitles = [[j.content, j.start.total_seconds(), j.end.total_seconds()] for j in srt.parse(s)]
print(subtitles)


MAX_LEN = int(subtitles[-1][2] + 17) * 22050  # 17 extra seconds fail-safe for long-last-segment
print("TOTAL LEN SAMPLES ", MAX_LEN, '\n====================')
total = np.zeros(MAX_LEN, dtype=np.float32)
is_tts = np.zeros(MAX_LEN, dtype=np.float32)

previous_segment_end = 0
# previous_segment_start = 0
for k, (_text_, orig_start, orig_end) in enumerate(subtitles):
    # orig_start += 1.7  # offset of TTS start so some nativ voice is heard
    # replace Knajevac > Verbatim phonemes
    _text_ = _text_.replace('Knjaževac', 'Kneeazevachy')
    _text_ = _text_.replace('fate', 'feyta')
    _text_ = _text_.replace('i.e. ', 'that is ')
    _text_ = _text_.replace(' cm ', ' centimeter ')
    _text_ = _text_.replace('.', ',').replace('sculpture', 'Skullptur').replace('figure', 'feegurr')
    _text_ = unidecode(_text_)
    print(_text_, orig_start, orig_end)

    # rate = VOICES[_voice]['rate']
    rate = min(max(.74, len(_text_) / 46), .841) #1.44)  # 1.24 for bieber
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
    # args.noise_w = .0001 # VOICES[_voice]['noise_w']

    state = CommandLineInterfaceState(args=args)
    initialize_args(state)
    initialize_tts(state)
    process_lines(state, wav_path='_tmp.wav')
    shutdown_tts(state)
    time.sleep(1.7)  # some wav files don't have time to be written

    # # predict emotion of synthesized speech

    x, fs = soundfile.read('_tmp.wav')
    print(fs)

    
    orig_start = int(orig_start * 22050)
    long_tts = (orig_start + len(x)) / 22050 - orig_end
    if long_tts > 0:   # TTS wants to finish beyond orig_end -> tell = tell-diff/2
        # MAX ALLOWED SHIFT to the LEFT is previous_segment_end
        tts_start = min(previous_segment_end, orig_start - int(long_tts))
    else:
        tts_start = orig_start
    


    # MASK 0 = native 1 = TTS
    # from previous tts start until current tts start
    #  


    previous_segment_end = tts_start + len(x)
    # ==
    free_samples = len(is_tts[tts_start:previous_segment_end])
    w = np.tanh(.001 * np.arange(free_samples))
    w *= w[::-1]  # symmetric tanh
    is_tts[tts_start:previous_segment_end] = w * (0 if k == 0 else 1)  # 0 .9 1 11 1111111111 .9
    # is_tts[tts_start:previous_segment_end] = w * (k % 2)  # only use TTS in odd sub
    total[tts_start:previous_segment_end] = x
    # previous_segment_start = tts_start
    # perhaps
soundfile.write(audio_track, total, 22050)

# ############################################################
# Extract native wav from video for fading
# ffmpeg -i Sandra\ Kotevska\,\ Painting\ Rose\ bush\,\ mixed\ media\,\ 2017.\ \[NMzC_036MtE\].mkv -f mp3 -ar 22050 -vn out44.wa
native_track = audio_track[:-4] + '_NATIVE_.wav'
print(f'\n__________________ {native_track=}')
subprocess.call(
    ["ffmpeg",
         "-y",  # https://stackoverflow.com/questions/39788972/ffmpeg-overwrite-output-file-if-exists
         "-i",
         video_file, #BUNICA\ NESTIUTA\ Adriana\ Andrei\ \[tmo2UbKYAqc\].webm 
         "-f",
         "mp3",
         "-ar",
         "22050",
         "-vn",
         native_track,
         ])
x_native, _ = soundfile.read(native_track)
x_native = x_native[:, 0]  # stereo




# PAD == SHORTEST OF TTS / NATIVE
if len(x_native) > len(total):
    total = np.pad(total, (0, max(0, x_native.shape[0] - total.shape[0])))
    is_tts = np.pad(total, (0, max(0, x_native.shape[0] - total.shape[0])))  # pad 0 means set label to native
else:  # pad native to len of is_tts & total
    x_native = np.pad(x_native, (0, max(0, total.shape[0] - x_native.shape[0])))
print(total.shape, is_tts.shape, x_native.shape, 'SHAPES')
# ==


class MovingAveragePrediction():
    def __init__(self):
        self._snr = 0
        self._rt60 = 0

    def inpaint_frame0(self, get_frame, t):
        im = np.copy(get_frame(t))

        ix = int(t * 22050)
        
        if is_tts[ix] > .5:  # mask is 1 thus tts else native
            frame = frame_tts
        else:
            frame = frame_orig
        h, w, _ = frame.shape
        # im[-h:, -w:, :] = (.4 * im[-h:, -w:, :] + .6 * frame_orig).astype(np.uint8)
        offset_h = 24
        im[offset_h:h+offset_h, :w, :] = (.4 * im[offset_h:h+offset_h, :w, :] + .6 * frame).astype(np.uint8)

        # im2 = np.concatenate([im, frame_tts], 0)
        # cv2.imshow('t', im2); cv2.waitKey(); cv2.destroyAllWindows()
        return im #np.concatenate([im, frane_ttts], 0)

# ALTERNATING AUDIO NATIVE .. TTS .. NATV .. TTS         
ma = MovingAveragePrediction()
final = vf.fl(ma.inpaint_frame0)

audio_track2 = 'l3.wav'
soundfile.write(audio_track2,
                (is_tts * total + (1-is_tts) * x_native)[:, None],
                22050)
# audioclip = AudioArrayClip(array=(is_tts * total + (1-is_tts) * x_native)[:, None],  # Nx1 not N,
#                 fps=vf.fps)
# final.audio = audioclip
video_file_inpaint = 'l3.mp4'

final.write_videofile(video_file_inpaint)
out_file = f'dubbed__{video_file[:-4].replace("/", "_")}__{_voice.replace("/","_")}.mp4'
# ====================================================
# soundfile.write(audio_track2, x, 22050)  # WRITE FINAL
# ========================================================

subprocess.call(
    ["ffmpeg",
        "-y",
        "-i",
        video_file_inpaint,
        "-i",
        audio_track2,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        " 1:a:0",
        out_file])