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


def main(args):
    '''
    1. Check text if .srt / .txt
    2. Precompute Style Vector (if native voice cloning or preselect)
    3. extract silent video & inpaint
    4. make tts & mix w. native audio
    5. output video .mp4
    6. If no video nor image: Fallback to TTS -> output .wav
    '''

    do_video_dub = True if args.text.endswith('.srt') else False

    SILENT_VIDEO = '_silent_video.mp4'
    AUDIO_TRACK = '_audio_track.wav'

    if do_video_dub:
        print('==\nVideo dubbing from timestamp subtitles: {args.txt}\n\n')
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


def command_line_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--affective',
        help="Select Emotional or non-emotional variant of Available voices: https://audeering.github.io/shift/",
        action='store_false',
    )
    parser.add_argument(
        '--device',
        help="Device ID",
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--text',
        help="Text to be synthesized.",
        default='sample.txt',
        type=str,
    )
    parser.add_argument(
        '--native',
        help="""
        --native: (without argument) a flag to do voice cloning using the speech from --video,
        --native my_voice.wav:  Voice cloning from user provided audio""",
        # nargs='?',
        # const=None,
        default=False)
    parser.add_argument(
        '--voice',
        help="TTS voice - Available voices: https://audeering.github.io/shift/",
        default="en_US/m-ailabs_low#judy_bieber", #'en_US/cmu-arctic_low#lnh',
        type=str,
    )
    parser.add_argument(
        '--image',
        help="If provided is set as background for output video, see --text",
        type=str,
    )
    parser.add_argument(
        '--video',
        help="Video file for video translation. Voice cloned from the video",
        type=str,
    )
    parser.add_argument(
        '--out_file',
        help="Output file name.",
        type=str,
        default='out'
    )
    return parser


def cli():
    parser = command_line_args()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()
