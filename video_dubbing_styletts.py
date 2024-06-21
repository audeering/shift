import numpy as np
import subprocess
import cv2
from moviepy.editor import *
import soundfile
import msinference

# SSH AGENT
#   eval $(ssh-agent -s)
#   ssh-add ~/.ssh/id_ed25519_github2024
#
#   git remote set-url origin git@github.com:audeering/shift   
# ==

# yt-dlp https://www.youtube.com/watch?v=mEplIrAP-z4

# https://superuser.com/questions/583393/how-to-extract-subtitle-from-video-using-ffmpeg
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
import transformers  # for transformers.PretrainedConfig
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



# MKV 2 MP4 - loadable by moviepy
# ffmpeg -y -i Distaff\ \[qVonBgRXcWU\].mkv -c copy -c:a aac Distaff_qVonBgRXcWU.mp4
video_file, srt_file = ['assets/Head_of_fortuna.mp4', 
                        'assets/head_of_fortuna_en.srt'
                    ]
vf = VideoFileClip(video_file)
# vf = vf.subclip(10, 14)

# args = permissive_dict.PermissiveDict()







#___________________________________________ NATIVE
prefix = video_file[:-4]  # , file_type = video_file.split('.')  # I think prefix can even be different that vdieoname
audio_track = prefix + '_17_04_styletts2_.wav'

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
         "24000", #"22050",
         "-vn",
         native_track,
         ])
x_native, _ = soundfile.read(native_track)
x_native = x_native[:, 0]  # stereo
#_______________________________________ NATIVE



with open(srt_file, "r") as f:
    s = f.read()
subtitles = [[j.content, j.start.total_seconds(), j.end.total_seconds()] for j in srt.parse(s)]
print(subtitles)


MAX_LEN = int(subtitles[-1][2] + 17) * 24000  # 17 extra seconds fail-safe for long-last-segment
print("TOTAL LEN SAMPLES ", MAX_LEN, '\n====================')



# ================================================= TGT
TGT_WAV = 'assets/wavs/en_US_m-ailabs_elliot_miller.wav'
# ===========================================================



previous_segment_end = 0
# ==================================================================================== START TTS
pieces = []
for k, (_text_, orig_start, orig_end) in enumerate(subtitles):




    # orig_start += 1.7  # offset of TTS start so some nativ voice is heard
    # replace Knajevac > Verbatim phonemes
    _text_ = _text_.replace('Knjaževac', 'Knazevach')
    _text_ = _text_.replace('fate', 'feyta')
    _text_ = _text_.replace('i.e. ', 'that is ')
    _text_ = _text_.replace(' cm ', ' centimeter ')
    _text_ = _text_.replace('.', ',').replace('sculpture', 'Skullptur').replace('figure', 'feegurr')
    _text_ = unidecode(_text_)
    print(_text_, orig_start, orig_end)




    # process_lines(state, wav_path='_tmp.wav')
    # style tts
    pieces.append(msinference.inference(_text_,
                                    msinference.compute_style(TGT_WAV), 
                                    alpha=0.3, 
                                    beta=0.7, 
                                    diffusion_steps=7, 
                                    embedding_scale=1)
                )  # print(f'{pieces[-1].shape=}')  # (52750, )
# ============================================================================================== END TTS    
x = np.concatenate(pieces, 0)
    # x = audresample.resample(x.astype(np.float32), 24000, 22050)  # reshapes (64,) -> (1,64)
    
total = x  # 24000 hz resample to mix with video?

print(total.shape)

soundfile.write(audio_track, total, 24000)  # 340000 sounds cool






# PAD SHORTEST of  TTS / NATIVE
if len(x_native) > len(total):
    total = np.pad(total, (0, max(0, x_native.shape[0] - total.shape[0])))

else:  # pad native to len of is_tts & total
    x_native = np.pad(x_native, (0, max(0, total.shape[0] - x_native.shape[0])))
print(total.shape, x_native.shape, 'PADDED TRACKS')
# ==

num = x_native.shape[0]
is_tts = .5 + .5 * np.tanh(4*(np.linspace(-10, 10, num) + 9.4)) #np.ones_like(x_native)  # fade heaviside


def inpaint_frame0(get_frame, t):
    '''blend banner saying if now plays tts or native voice'''
    im = np.copy(get_frame(t))

    ix = int(t * 24000)
    
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



# FUSE AUDIO -> NATIVE .. TTS .. NATV .. TTS         

final = vf.fl(inpaint_frame0)

audio_track2 = 'l3.wav'
soundfile.write(audio_track2,
                (is_tts * total + (1-is_tts) * x_native)[:, None],
                24000)
# audioclip = AudioArrayClip(array=(is_tts * total + (1-is_tts) * x_native)[:, None],  # Nx1 not N,
#                 fps=vf.fps)
# final.audio = audioclip
video_file_inpaint = 'l3.mp4'

final.write_videofile(video_file_inpaint)
out_file = f'dub_styletts2_BMN_{video_file[:-4].replace("/", "_")}.mp4'
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
