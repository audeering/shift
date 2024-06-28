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


def tts_multi_sentence(precomputed_style_vector=None,
                       text=None,
                       voice=None):
    '''create 24kHZ np.array with tts

       precomputed_style_vector :   required if en_US or en_UK in voice, so
                                    to perform affective TTS.
       text : string
       voice: string or None (falls to styleTTS)
       '''
    if 'en_US/' or 'en_UK/' in voice or voice is None:
        assert precomputed_style_vector is not None, 'For affective TTS, style vector is needed.'
        if len(text) > 170:  # long text -> split sentences
            x = []
            for _sentence in text_utils.split_into_sentences(text):
                x.append(msinference.inference(_sentence,
                            precomputed_style_vector,
                                        alpha=0.3,
                                        beta=0.7,
                                        diffusion_steps=7,
                                        embedding_scale=1))
            return np.concatenate(x)  # @24000 Hz - match the StyleTTS (because mimic3 synth 22050)
        return msinference.inference(text,
                            precomputed_style_vector,
                                        alpha=0.3,
                                        beta=0.7,
                                        diffusion_steps=7,
                                        embedding_scale=1)  # single sentence @24000 Hz
    # NON AFFECTIVE i.e. mimic3
    # using ps to call mimic3 because samples dont have time to be written in stdout buffer
    _ssml = text_utils.build_ssml(text=text,
                                  voice=voice)
    with open('_tmp_ssml.txt', 'w') as f:
        f.write(_ssml)
    ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > _tmp.wav', shell=True)
    ps.wait()
    x, fs = soundfile.read('_tmp.wav')
    return audresample.resample(x.astype(np.float32), 24000, fs)[0, :]  # reshapes (64,) -> (1,64)


def main(args):
    '''1. Parse text
       2. Precompute Style Vector (native voice cloning or internal selection)
       3. Run
       4. Use cases: Image2speech, TTS basic, Video dubbing, Silentvideo2Speech
    '''

    do_video_dub = True if args.text.endswith('.srt') else False
    
    if do_video_dub:
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
                "24000", #"22050",
                "-vn",
                native_audio_file,
                ])
        x_native, _ = soundfile.read(native_audio_file)  # reads mp3
        x_native = x_native[:, 0]  # stereo
        # ffmpeg -i Sandra\ Kotevska\,\ Painting\ Rose\ bush\,\ mixed\ media\,\ 2017.\ \[NMzC_036MtE\].mkv -f mp3 -ar 22050 -vn out44.wa
    else:
        # split inside tts_multi_sentence ? if we call mimic3 there is no need for this split
        # this is only for StyleTTS2 with long text 
        # it may happend that a sub is as well long (allow this risk)
        # actually split in sentences only if text is very long 
        # check this in tts_multisentence
        with open(args.text, 'r') as f:
            text = ''.join(f)
        # text = text_utils.split_into_sentences(text)  # DO IN TTS() if text is vvery long
        




    if args.native is not None:
        if do_video_dub:
            native_audio_file = args.video.replace('.','').replace('/','')
            native_audio_file += '__native_audio_track.wav'
            soundfile.write('tgt_spk.wav', 
                np.concatenate([
                    x_native[:int(4*24000)],    
                                ], 0).astype(np.float32), 24000)  # 27400?
            precomputed_style_vector = msinference.compute_style('tgt_spk.wav')
        else:  # not video dub
            try:
                precomputed_style_vector = msinference.compute_style(args.native)
            except ValueError:  # fallback - internal voice
                pass
    # Global FALLBACK - Interal Voice
    if 'en_US' in args.voice or 'en_UK' in args.voices:
        _dir = '/' if args.affect else '_v2/'
        precomputed_style_vector = msinference.compute_style(
            'assets/wavs/style_vector' + _dir + args.voice.replace(
                                '/', '_').replace('#', '_').replace(
                                'cmu-arctic', 'cmu_arctic').replace(
                                '_low', '') + '.wav')
    else:
        precomputed_style_vector = None


    # Got precomputed_style_vector & do_video_dubbing

    # E X E C USE CASES  16h40 27 Jun 2024

    if args.video is not None:
        if do_video_dub:
            # -- video_dubbing_styletts.py
            # frame for video dubbing
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
            # video_file, srt_file = ['assets/Head_of_fortuna.mp4', 
            #                         'assets/head_of_fortuna_en.srt'
            #                     ]
            video_file = args.video
            # srt_file = args.text  # parsed already?
            vf = VideoFileClip(video_file)
            # vf = vf.subclip(10, 14)

            # args = permissive_dict.PermissiveDict()


            # do we have already extracted the native audio track of video ?
         

            subtitles = text


            MAX_LEN = int(subtitles[-1][2] + 17) * 24000  # 17 extra seconds fail-safe for long-last-segment
            print("TOTAL LEN SAMPLES ", MAX_LEN, '\n====================')



            previous_segment_end = 0
            
            pieces = []
            for k, (_text_, orig_start, orig_end) in enumerate(subtitles):
                # # orig_start += 1.7  # offset of TTS start so some nativ voice is heard
                # # replace Knajevac > Verbatim phonemes
                # _text_ = _text_.replace('Knjaževac', 'Knazevach')
                # _text_ = _text_.replace('fate', 'feyta')
                # _text_ = _text_.replace('i.e. ', 'that is ')
                # _text_ = _text_.replace(' cm ', ' centimeter ')
                # _text_ = _text_.replace('.', ',').replace('sculpture', 'Skullptur').replace('figure', 'feegurr')
                # _text_ = unidecode(_text_)
                pieces.append(tts_multi_sentence(text=_text_,
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
            print(total.shape, x_native.shape, 'PADDED TRACKS')
            # ==

            num = x_native.shape[0]
            is_tts = .5 + .5 * np.tanh(4*(np.linspace(-10, 10, num) + 9.4)) #np.ones_like(x_native)  # fade heaviside


            def inpaint_frame0(get_frame, t):
                '''blend banner saying if (now plays) tts or native voice
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
                im[offset_h:h+offset_h, :w, :] = (.4 * im[offset_h:h+offset_h, :w, :] + .6 * frame).astype(np.uint8)

                # im2 = np.concatenate([im, frame_tts], 0)
                # cv2.imshow('t', im2); cv2.waitKey(); cv2.destroyAllWindows()
                return im #np.concatenate([im, frane_ttts], 0)



            # FUSE AUDIO -> NATIVE .. TTS .. NATV .. TTS         

            final = vf.fl(inpaint_frame0)

            audio_track2 = 'l3.wav'
            soundfile.write(audio_track2,
                            # (is_tts * total + (1-is_tts) * x_native)[:, None],
                            (.64 * total + .27 * x_native)[:, None],
                            24000)
            video_file_inpaint = 'l3.mp4'
            final.write_videofile(video_file_inpaint)
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
                    args.out_file + '.mp4'])
        # -- END video_dubbing_styletts.py
        return 0
    else:  # silent video
        print('\n VIDEO SILENT not implented')
        raise NotImplementedError


    # IMAGE 2 SPEECH

    if args.image is not None:

        STATIC_FRAME = args.image  #'assets/image_from_T31.jpg'
        SILENT_VIDEO = '_silent_video.mp4'
        AUDIO_TRACK = '_audio_track.wav'
        OUT_FILE = args.out_file + '.mp4' #f'image_to_speech.mp4'

        # SILENT CLIP

        clip_silent = ImageClip(STATIC_FRAME).set_duration(5)  # as long as the audio - TTS first
        clip_silent.write_videofile(SILENT_VIDEO, fps=24)


        x = tts_multi_sentence(text=text,
                               precomputed_style_vector=precomputed_style_vector,
                               voice=args.voice
                               )
        soundfile.write(AUDIO_TRACK, x, 24000)

        # -- end tts

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
        print('\nImage2speech - output video is saved as {OUT_FILE}')
        return 0
    # ========================
    # DEFAULTSD TO BASIC TTS=============================
    x = tts_multi_sentence(text=text, precomputed_style_vector=precomputed_style_vector, voice=args.voice)
    soundfile.write(args.out_file + '.wav', x)
    return 'BASIC TTS'












    

def command_line_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--affect',
        help="Use emotional variant of Available voices: https://audeering.github.io/shift/",
        action='store_true',
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
        help="wav from which to find emotion and automatically choose best TTS voice.",
        type=str,
    )
    parser.add_argument(
        '--voice',
        help="TTS voice - Available voices: https://audeering.github.io/shift/",
        default='en_US/cmu-arctic_low#lnh',
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
