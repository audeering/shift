# -*- coding: utf-8 -*-
import numpy as np
import soundfile
import text_utils
import msinference
import re
import srt
import subprocess
import cv2
import markdown
import nltk

from pathlib import Path
from types import SimpleNamespace
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from moviepy.editor import *

nltk.download('punkt')
nltk.download('punkt_tab')
Path('./flask_cache').mkdir(parents=True, exist_ok=True)

# SSH AGENT
#   eval $(ssh-agent -s)
#   ssh-add ~/.ssh/id_ed25519_github2024
#
#   git remote set-url origin git@github.com:audeering/shift
# ==


def tts_multi_sentence(precomputed_style_vector=None,
                       text=None,
                       voice=None,
                       speed=None):
    '''24 kHZ tts'''
    
    if ('en_US/' in voice) or ('en_UK/' in voice) or (voice is None):
        
        assert precomputed_style_vector is not None, 'For affective TTS, style vector is needed.'
        
        if isinstance(text, str) and len(text) > 100:
            text = text_utils.split_into_sentences(text)  # split to short sentences (~200 phonemes max)
        else:
            text = [text]  # list of D sentences
        
        # STYLETTS2
        
        x = []
        for _sentence in text:

            x.append(msinference.inference(_sentence,
                        precomputed_style_vector,
                                    alpha=0.3,
                                    beta=0.7,
                                    diffusion_steps=7,
                                    embedding_scale=1))
        return np.concatenate(x)
    # FOREIGN MMS-TTS
    
    x = msinference.foreign(text=text,
                            lang=voice,  # voice = 'romanian', 'serbian' 'hungarian'
                            speed=speed)
    
    return x

def _resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    '''https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py'''
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def index():
    with open('README.md', 'r') as f:
        return markdown.markdown(f.read())


    

@app.route("/", methods=['GET', 'POST', 'PUT'])
def serve_wav():
    # https://stackoverflow.com/questions/13522137/in-flask-convert-form-post-
    #                      object-into-a-representation-suitable-for-mongodb
    r = request.form.to_dict(flat=False)

    # Physically Save Client Files - DELTE / of name ?
    for f, obj in request.files.items():
        
        obj.save(f'flask_cache/{f[-6:]}')
        
    args = SimpleNamespace(
        text      = None if r.get('text')  is None else 'flask_cache/' + r.get('text' )[0][-6:],
        video     = None if r.get('video') is None else 'flask_cache/' + r.get('video')[0][-6:],
        image     = None if r.get('image') is None else 'flask_cache/' + r.get('image')[0][-6:],
        native    = None if r.get('native') is None else 'flask_cache/' + r.get('native')[0][-6:],
        affective =       r.get('affective')[0],
        voice     =       r.get('voice')[0],
        speed     = float(r.get('speed')[0])  # For Non-English MMS TTS
                          )  # alpha_num('/folder1/folder2/file.txt')
    # print('\n==RECOMPOSED as \n',request.data,request.form,'\n==')
    

    print(args, 'ENTER Script')
    do_video_dub = True if args.text.endswith('srt') else False

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
        text = re.sub(' +', ' ', t)  # delete spaces
        
        
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
            
    # NOTE: style vector is None for FOREIGN LANGS
    
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
        
        print(f'\n______________________________\n'
              f'Gen Banners for TTS/Native Title {frame_tts.shape=} {frame_orig.shape=}'
              f'\n______________________________\n')
        # ====SILENT VIDEO EXTRACT====
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
        
        # GET 1st FRAME to OBTAIN frame RESOLUTION
        h, w, _ = vf.get_frame(0).shape
        frame_tts = _resize(frame_tts, width=w)
        frame_orig = _resize(frame_orig, width=w)
        h, w, _ = frame_orig.shape
        
        try:
            
            # inpaint banner to say if native voice
            num = x_native.shape[0]
            is_tts = .5 + .5 * np.tanh(4*(np.linspace(-10, 10, num) + 7.4))  # fade heaviside
            
            def inpaint_banner(get_frame, t):
                '''blend banner - (now plays) tts or native voic
                '''
                
                im = np.copy(get_frame(t))  # pic
                

                ix = int(t * 24000)

                if is_tts[ix] > .5:     # mask == 1 => tts / mask == 0 -> native
                    frame = frame_tts   # rename frame to rsz_frame_... because if frame_tts is mod
                                        # then is considered a "local variable" thus the "outer var"
                                        # is not observed by python raising referenced before assign
                else:
                    frame = frame_orig
                
                # im[-h:, -w:, :] = (.4 * im[-h:, -w:, :] + .6 * frame_orig).astype(np.uint8)
                
                

                offset_h = 24
                
                
                print(f'  > inpaint_banner() HAS NATIVE:  {frame.shape=} {im.shape=}\n\n\n\n')
                
                
                
                im[offset_h:h + offset_h, :w, :] = (.4 * im[offset_h:h + offset_h, :w, :] 
                                                    + .6 * frame).astype(np.uint8)
                
                # im2 = np.concatenate([im, frame_tts], 0)
                # cv2.imshow('t', im2); cv2.waitKey(); cv2.destroyAllWindows()
                return im  # np.concatenate([im, frane_ttts], 0)
            
        except UnboundLocalError:  # args.native == False
            
            def inpaint_banner(get_frame, t):

                im = np.copy(get_frame(t))
                offset_h = 24
                im[offset_h:h + offset_h, :w, :] = (.4 * im[offset_h:h+offset_h, :w, :] 
                                                    + .6 * frame_tts).astype(np.uint8)
                return im
        vf = vf.fl(inpaint_banner)
        vf.write_videofile(SILENT_VIDEO)

        # ==== SYNC .srt ====
        
        if do_video_dub:
            OUT_FILE = './flask_cache/tmp.mp4' #args.out_file + '_video_dub.mp4'
            subtitles = text
            MAX_LEN = int(subtitles[-1][2] + 17) * 24000
            total = np.zeros(MAX_LEN, dtype=np.float32)
            # 17 extra seconds fail-safe for long-last-segment
            
            previous_segment_end = 0
            pieces = []
            for k, (_text_, orig_start, orig_end) in enumerate(subtitles):
                
                x = tts_multi_sentence(
                    text=_text_,
                    precomputed_style_vector=precomputed_style_vector,
                    voice=args.voice,
                    speed=args.speed)

                # PAUSES BETWEEN SUBTITLE SEGMENTS

                orig_start = int(orig_start * 24000)
                long_tts = (orig_start + len(x)) / 24000 - orig_end
                if long_tts > 0:   # TTS wants to finish beyond native_end -> tell = tell-diff/2
                    # MAX ALLOWED SHIFT to the LEFT is previous_segment_end
                    tts_start = min(previous_segment_end, orig_start - int(long_tts))
                else:
                    tts_start = max(orig_start, previous_segment_end)  # to assure that we don't cut a long
                    # unfinished previous segment we should use as tts_start
                    # the max(orig_start, previous_segment_end)
                previous_segment_end = tts_start + len(x)
                total[tts_start:previous_segment_end] = x
                
                # --



            
            # PAD SHORTEST of  TTS / NATIVE
            if len(x_native) > len(total):
                
                total = np.pad(total, (0, max(0, x_native.shape[0] - total.shape[0])))

            else:
                
                shorter_n = x_native.shape[0]
                longer_n = total.shape[0]
                x_native = np.pad(x_native, (0, max(0, longer_n - shorter_n)))
                is_tts = np.pad(is_tts, (0, max(0, longer_n - shorter_n)))
                
            # print(total.shape, x_native.shape, is_tts.shape, 'PADDED TRACKS')
            soundfile.write(AUDIO_TRACK,
                            (is_tts * total + (1-is_tts) * x_native)[:, None],
                            # (.64 * total + .27 * x_native)[:, None],
                            24000)
        else:  # Video from plain (.txt)
            OUT_FILE = './flask_cache/tmp.mp4' #args.out_file + '_video_from_txt.mp4'
            x = tts_multi_sentence(text=text,
                               precomputed_style_vector=precomputed_style_vector,
                               voice=args.voice,                            speed=args.speed)
            soundfile.write(AUDIO_TRACK, x, 24000)

    # IMAGE 2 SPEECH

    if args.image is not None:
        
        STATIC_FRAME = args.image  # 'assets/image_from_T31.jpg'
        OUT_FILE = './flask_cache/tmp.mp4' #args.out_file + '_image_to_speech.mp4'

        # SILENT CLIP

        clip_silent = ImageClip(STATIC_FRAME).set_duration(5)  # as long as the audio - TTS first
        clip_silent.write_videofile(SILENT_VIDEO, fps=24)

        x = tts_multi_sentence(text=text,
                               precomputed_style_vector=precomputed_style_vector,
                               voice=args.voice,                            speed=args.speed
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

        print(f'\nIM2SPeech: output video is saved as {OUT_FILE}')
        
    else:
        
        # Fallback: No image nor video provided - do only tts
        x = tts_multi_sentence(text=text,
                            precomputed_style_vector=precomputed_style_vector, 
                            voice=args.voice,
                            speed=args.speed)
        
        OUT_FILE = './flask_cache/tmp.wav' #args.out_file + '.wav'
        soundfile.write(OUT_FILE, x, 24000)

        print(f'\nFALLBACK: output video is saved as {OUT_FILE}')
    

    # audios = [msinference.inference(text, 
    #                                 msinference.compute_style(f'voices/{voice}.wav'), 
    #                                 alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1)]
    # # for t in [text]:
    # output_buffer = io.BytesIO()
    # write(output_buffer, 24000, np.concatenate(audios))
    # response = Response(output_buffer.getvalue())
    # response.headers["Content-Type"] = "audio/wav"
    # https://stackoverflow.com/questions/67591467/
    #            flask-shows-typeerror-send-from-directory-missing-1-required-positional-argum
    response = send_from_directory('flask_cache/', path=OUT_FILE.split('/')[-1])
    response.headers['suffix-file-type'] = OUT_FILE.split('/')[-1]
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")
