
# -*- coding: utf-8 -*-
import numpy as np
import soundfile
import argparse
import audresample
from moviepy.editor import *
import text_utils
import msinference
import srt
import subprocess
import cv2
from pathlib import Path
from permissive_dict import PermissiveDict
from types import SimpleNamespace
# StyleTTS 2 HTTP Streaming API by @fakerybakery - Copyright (c) 2023 mrfakename. All rights reserved.
# Docs: API_DOCS.md
# To-Do:
# * Support voice cloning
# * Implement authentication, user "credits" system w/ SQLite3
import io
import os
import hashlib
import threading
import markdown
import re
import json

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
import soundfile



Path('./flask_cache').mkdir(parents=True, exist_ok=True)

# SSH AGENT
#   eval $(ssh-agent -s)
#   ssh-add ~/.ssh/id_ed25519_github2024
#
#   git remote set-url origin git@github.com:audeering/shift
# ==


def tts_multi_sentence(precomputed_style_vector=None,
                       text=None,
                       voice=None):
    '''create 24kHZ np.array with tts

       precomputed_style_vector :   required if en_US or en_UK in voice, so
                                    to perform affective TTS.
       text : string
       voice: string or None (falls to styleTTS)
       '''
    if ('en_US/' in voice) or ('en_UK/' in voice) or (voice is None):
        assert precomputed_style_vector is not None, 'For affective TTS, style vector is needed.'
        x = []
        for _sentence in text:
            x.append(msinference.inference(_sentence,
                        precomputed_style_vector,
                                    alpha=0.3,
                                    beta=0.7,
                                    diffusion_steps=7,
                                    embedding_scale=1))
        x = np.concatenate(x)
        # N = x.shape[0]
        # x = .9 * x * (np.abs(np.sin(np.linspace(-np.pi, np.pi, x.shape[0]) / .07)))
        return x
    # NON AFFECTIVE mimic3
    #
    #
    # if called via video dubbing text has to be list of single sentence
    text_utils.store_ssml(text=text,
                          voice=voice)
    ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > _tmp.wav', shell=True)
    ps.wait()
    x, fs = soundfile.read('_tmp.wav')
    return audresample.resample(x.astype(np.float32), 24000, fs)[0, :]  # reshapes (64,) -> (1,64)





















# voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
# voices = {}
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)



app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def index():
    with open('API_DOCS.md', 'r') as f:
        return markdown.markdown(f.read())







@app.route("/api/v1/static", methods=['POST'])
def serve_wav():
    # https://stackoverflow.com/questions/13522137/in-flask-convert-form-post-
    #                      object-into-a-representation-suitable-for-mongodb
    r = request.form.to_dict(flat=False)

    
    
    

    # Physically Save Client Files
    for filename, obj in request.files.items():
        obj.save(f'flask_cache/{filename.replace("/","")}')
        
    print('Saved all files on Server Side\n\n') 
    # # Args - pass in request
    
    # audio_file = request.files['h_voice']
    # audio_file.save('_tmp_srv.wav')
    # # waveform, _ = soundfile.read(file=io.BytesIO(audio_bytes), dtype='float32')
    # x, _ = soundfile.read('_tmp_srv.wav')
    # print(x.shape, x[:4])

    # args.text = args.get("text")
    # args.image = args.get('image')
    # args.video = args.get('video')
    # args.native = args.get('native')
    # args.voice = args.get('voice')
    # args.affective = args.get('affective')
    # args.out_file = args.get('out_file')

    # print('\nMAKE Args=\n', args)
    args = SimpleNamespace(text=None if r.get('text') is None else 'flask_cache/' + r.get('text')[0],  # ['sample.txt']
                           video=None if r.get('video') is None else 'flask_cache/' + r.get('video')[0],
                           image=None if r.get('image') is None else 'flask_cache/' + r.get('image')[0], #flask_cache/' + request.data.get("image"),
                           voice=r.get('voice')[0],
                           native=None if r.get('native') is None else 'flask_cache/' + r.get('native')[0],
                           affective = r.get('affective')[0],
                           out_file = 'flask_cache/' + ('out6' if r.get('out_file')[0] is None else r.get('out_file')[0])
                                  )
    # print('\n==RECOMPOSED as \n',request.data,request.form,'\n==')
    

    print(args, 'ENTER Script')
    do_video_dub = True if args.text.endswith('.srt') else False

    SILENT_VIDEO = '_silent_video.mp4'
    AUDIO_TRACK = '_audio_track.wav'

    if do_video_dub:
        print('==\nFound .srt : {args.txt}, thus Video should be given as well\n\n')
        with open(args.text, "r") as f:
            s = f.read()
        text = [[j.content, j.start.total_seconds(), j.end.total_seconds()] for j in srt.parse(s)]
        assert args.video is not None
        native_audio_file = '_tmp.wav'
        subprocess.call(
            ["ffmpeg",
                "-y",  # https://stackoverflow.com/questions/39788972/ffmpeg-overwrite-output-file-if-exists
                "-i",
                args.video,
                "-f",
                "mp3",
                "-ar",
                "24000",  # "22050 for mimic3",
                "-vn",
                native_audio_file])
        x_native, _ = soundfile.read(native_audio_file)  # reads mp3
        x_native = x_native[:, 0]  # stereo
        # ffmpeg -i Sandra\ Kotevska\,\ Painting\ Rose\ bush\,\ mixed\ media\,\ 2017.\ \[NMzC_036MtE\].mkv -f mp3 -ar 22050 -vn out44.wa
    else:
        with open(args.text, 'r') as f:
            t = ''.join(f)
        text = [t] if len(t) < 170 else text_utils.split_into_sentences(t)

    # ====STYLE VECTOR====

    precomputed_style_vector = None
    if args.native:  # Voice Cloning
        try:
            precomputed_style_vector = msinference.compute_style(args.native)
        except soundfile.LibsndfileError:  # Fallback - internal voice
            print('\n  Could not voice clone audio:', args.native, 'fallback to video or Internal TTS voice.\n')
        if do_video_dub:  # Clone voice via Video
            native_audio_file = args.video.replace('.', '').replace('/', '')
            native_audio_file += '__native_audio_track.wav'
            soundfile.write('tgt_spk.wav',
                np.concatenate([
                    x_native[:int(4 * 24000)]], 0).astype(np.float32), 24000)  # 27400?
            precomputed_style_vector = msinference.compute_style('tgt_spk.wav')

    # NOTE: style vector may be None

    if precomputed_style_vector is None:
        if 'en_US' in args.voice or 'en_UK' in args.voice:
            _dir = '/' if args.affective else '_v2/'
            precomputed_style_vector = msinference.compute_style(
                'assets/wavs/style_vector' + _dir + args.voice.replace(
                    '/', '_').replace(
                    '#', '_').replace(
                    'cmu-arctic', 'cmu_arctic').replace(
                    '_low', '') + '.wav')
    print('\n  STYLE VECTOR \n', precomputed_style_vector)
    # ====SILENT VIDEO====

    if args.video is not None:
        # banner
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
        #     cv2.imshow('i', frame_tts); cv2.waitKey(); cv2.destroyAllWindows()
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
        #====SILENT VIDEO EXTRACT====
        # DONLOAD SRT from youtube
        #
        #     yt-dlp --write-sub --sub-lang en --convert-subs "srt" https://www.youtube.com/watch?v=F1Ib7TAu7eg&list=PL4x2B6LSwFewdDvRnUTpBM7jkmpwouhPv&index=2
        #
        #
        # .mkv ->.mp4 moviepy loads only .mp4
        #
        #     ffmpeg -y -i Distaff\ \[qVonBgRXcWU\].mkv -c copy -c:a aac Distaff_qVonBgRXcWU.mp4
        #           video_file, srt_file = ['assets/Head_of_fortuna.mp4', 
        #                         'assets/head_of_fortuna_en.srt']
        #
        video_file = args.video
        vf = VideoFileClip(video_file)
        try:
            # inpaint banners if native voice
            num = x_native.shape[0]
            is_tts = .5 + .5 * np.tanh(4*(np.linspace(-10, 10, num) + 9.4))  # np.ones_like(x_native)  # fade heaviside
            def inpaint_banner(get_frame, t):
                '''blend banner - (now plays) tts or native voic
                '''
                im = np.copy(get_frame(t))

                ix = int(t * 24000)

                if is_tts[ix] > .5:  # mask is 1 thus tts else native
                    frame = frame_tts
                else:
                    frame = frame_orig
                h, w, _ = frame.shape
                # im[-h:, -w:, :] = (.4 * im[-h:, -w:, :] + .6 * frame_orig).astype(np.uint8)
                offset_h = 24
                im[offset_h:h + offset_h, :w, :] = (.4 * im[offset_h:h + offset_h, :w, :] + .6 * frame).astype(np.uint8)

                # im2 = np.concatenate([im, frame_tts], 0)
                # cv2.imshow('t', im2); cv2.waitKey(); cv2.destroyAllWindows()
                return im  # np.concatenate([im, frane_ttts], 0)
        except UnboundLocalError:  # args.native == False
            def inpaint_banner(get_frame, t):
                im = np.copy(get_frame(t))
                frame = frame_tts
                h, w, _ = frame.shape
                offset_h = 24
                im[offset_h:h + offset_h, :w, :] = (.4 * im[offset_h:h+offset_h, :w, :] + .6 * frame).astype(np.uint8)
                return im
        vf = vf.fl(inpaint_banner)
        vf.write_videofile(SILENT_VIDEO)

        # ==== TTS .srt ====

        if do_video_dub:
            OUT_FILE = args.out_file + '_video_dub.mp4'
            subtitles = text
            MAX_LEN = int(subtitles[-1][2] + 17) * 24000  # 17 extra seconds fail-safe for long-last-segment
            print("TOTAL LEN SAMPLES ", MAX_LEN, '\n====================')
            pieces = []
            for k, (_text_, orig_start, orig_end) in enumerate(subtitles):

                # PAUSES ?????????????????????????


                pieces.append(tts_multi_sentence(text=[_text_],
                                                 precomputed_style_vector=precomputed_style_vector,
                                                 voice=args.voice)
                              )
            total = np.concatenate(pieces, 0)
            # x = audresample.resample(x.astype(np.float32), 24000, 22050)  # reshapes (64,) -> (1,64)
            # PAD SHORTEST of  TTS / NATIVE
            if len(x_native) > len(total):
                total = np.pad(total, (0, max(0, x_native.shape[0] - total.shape[0])))

            else:  # pad native to len of is_tts & total
                x_native = np.pad(x_native, (0, max(0, total.shape[0] - x_native.shape[0])))
            # print(total.shape, x_native.shape, 'PADDED TRACKS')
            soundfile.write(AUDIO_TRACK,
                            # (is_tts * total + (1-is_tts) * x_native)[:, None],
                            (.64 * total + .27 * x_native)[:, None],
                            24000)
        else:  # Video from plain (.txt)
            OUT_FILE = args.out_file + '_video_from_txt.mp4'
            x = tts_multi_sentence(text=text,
                               precomputed_style_vector=precomputed_style_vector,
                               voice=args.voice
                               )
            soundfile.write(AUDIO_TRACK, x, 24000)

    # IMAGE 2 SPEECH

    if args.image is not None:

        STATIC_FRAME = args.image  # 'assets/image_from_T31.jpg'
        OUT_FILE = args.out_file + '_image_to_speech.mp4'

        # SILENT CLIP

        clip_silent = ImageClip(STATIC_FRAME).set_duration(5)  # as long as the audio - TTS first
        clip_silent.write_videofile(SILENT_VIDEO, fps=24)

        x = tts_multi_sentence(text=text,
                               precomputed_style_vector=precomputed_style_vector,
                               voice=args.voice
                               )
        soundfile.write(AUDIO_TRACK, x, 24000)
    if args.video or args.image:
        # write final output video
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
        print(f'\noutput video is saved as {OUT_FILE}')
        return 0

    # Fallback: No image nor video provided - do only tts
    x = tts_multi_sentence(text=text,
                           precomputed_style_vector=precomputed_style_vector, 
                           voice=args.voice)
    OUT_FILE = args.out_file + '.wav'
    soundfile.write(OUT_FILE, x, 24000)


    

    # audios = [msinference.inference(text, 
    #                                 msinference.compute_style(f'voices/{voice}.wav'), 
    #                                 alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1)]
    # # for t in [text]:
    # output_buffer = io.BytesIO()
    # write(output_buffer, 24000, np.concatenate(audios))
    # response = Response(output_buffer.getvalue())
    # response.headers["Content-Type"] = "audio/wav"

    # 
    # https://stackoverflow.com/questions/67591467/flask-shows-typeerror-send-from-directory-missing-1-required-positional-argum
    response = send_from_directory('flask_cache/', path=OUT_FILE.split('/')[-1])
    response.headers['my-custom-header'] = 'my-custom-status-0'
    return response

    
if __name__ == "__main__":
    app.run("0.0.0.0")
