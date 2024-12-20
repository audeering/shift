# FOR EACH VOICE -> create .wav file per chapter & full audiobook.wav from assets/INCLUSION_IN_MUSEUMS_audiobook.docx
#
# Chapters
#
#   ROOT_DIR/voice/voxstr_CHAPTER_0.wav
#     ..
#   ROOT_DIR/voice/voxstr_CHAPTER_10.wav 
#   ROOT_DIR/voice/voxstr_full_book.wav
#
# Full AudioBook
#
#   ROOT_DIR/full_audiobook_all_voices.wav

import cv2
import subprocess
import numpy as np
import soundfile
import docx  # pip install python-docx

from pathlib import Path
from moviepy.editor import *

FS = 24000
ROOT_DIR = './tts_audiobooks/voices/'
Path(ROOT_DIR).mkdir(parents=True,
                     exist_ok=True)
voices = [
        # 'en_US/hifi-tts_low#9017' ,
        'en_US/m-ailabs_low#mary_ann',
        'en_US/cmu-arctic_low#jmk',
        # 'en_US/cmu-arctic_low#eey',
        # 'en_UK/apope_low'
        ]  # select any voice from - https://audeering.github.io/shift/

d = docx.Document('assets/INCLUSION_IN_MUSEUMS_audiobook.docx')  # slightly changed from the original .docx to be audible as by adding extra 'by them from this of etc.'

last_paragraph_was_silence = False  # to know to add silence only once after only at the 1st empty paragraph we detect

chapter_counter = 0  # assure chapters start with CHAPTER: ONCE UPON A TIME

youtube_video_parts = []  # audiobook .mp4 from each voice

for vox in voices:

    # string (map for assets/)
    
    vox_str = vox.replace(
                '/', '_').replace(
                '#', '_').replace(
                'cmu-arctic', 'cmu_arctic').replace(
                '_low', '').replace('-','')
                
    # create dir for chapter_x.wav & audiobook.wav - for this voice vox
    
    Path(ROOT_DIR + vox_str + '/').mkdir(parents=True,
                                         exist_ok=True)
                
                    
    print(vox)
 
    # for new voice start list of audio tiles making up the 1st chapter of book
 
    total = []
    chapter = []
    
    for para in d.paragraphs:  #[:41]:
        t = para.text
        
        
        
        
        # start new chapter
        
        if t.startswith('CHAPTER:'):
            
            
            
            # silence for end chapter
            
            chapter.append(np.zeros(int(.1 * FS), 
                                    dtype=np.float32))
                
            # chapter.wav
            
            audio = np.concatenate(chapter)
            
            soundfile.write(
                ROOT_DIR + vox_str + f'/{vox_str}_chapter_{chapter_counter}.wav',
                audio,
                FS)  # 27400?
            
            # fill AUDIO of this chapter into total (for complete audiobook)
            
            total.append(audio)
            
            # new chapter
            
            chapter = []
            
            chapter_counter += 1
            
            
            
            
                
        # If paragraph is non empty -> TTS
                
        if len(t) > 2 and t[0] != '{' and t[-1] != '}' and 'Figure' not in t:
            
            # place paragraph text to .txt for tts.py
            
            with open('_tmp.txt', 'w') as f:
                f.write(t.lower())  # WARNING! cast to lower otherwise accesibiliTy is pronounces accessibili..tay
            
            
            print(t,'\n_____________________________\n')
            
            # TTS
            
            subprocess.run(
                [
                "python",
                "tts.py",
                "--text", 
                "_tmp.txt", #t,         # paragraph text tts and append to voice_chapter.wav
                # "--affect",
                #'--image', '_tmp_banner.png',
                # '--scene', 'calm sounds of castle',
                '--voice', vox,
                '--out_file', '_tmp'  # save on _tmp load audio and concat to total
                    ])
            
            audio, _fs = soundfile.read('out/_tmp.wav')
            print('CHAPTER\n\n\n\n____', audio.shape,'____\n')
            chapter.append(audio)
            
            # flag
            
            last_paragraph_was_silence = False
            
        # append silence if empty paragraph (e.g. end of Section)
            
        else:
            
            if not last_paragraph_was_silence:  # skip multiple empty pargraphs - silence is added only once
                
                chapter.append(np.zeros(int(.1 * FS), 
                                        dtype=np.float32))
                
                last_paragraph_was_silence = True
                
    # save full .wav audiobook - for this voice
    
    soundfile.write(
                    ROOT_DIR + vox_str + '/' + f'{vox_str}_full_audiobook.wav',
                    np.concatenate(total),
                    FS)  # 27400?



    
    # pic TTS voice
    
    voice_pic = np.zeros((768, 1024, 3), dtype=np.uint8)

    shift_logo = cv2.imread('assets/shift_banner.png')

    voice_pic[:100, :400, :] = shift_logo[:100, :400, :]

    # voice name
    # frame_tts = np.zeros((104, 1920, 3), dtype=np.uint8)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 640)  # w,h
    fontScale              = 2
    fontColor              = (69, 74, 74)
    thickness              = 4
    lineType               = 2
    # voice
    cv2.putText(voice_pic, vox, #'en_US/m-ailabs_low#mary_ann',
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)
    # =
    cv2.putText(voice_pic, 'TTS voice =',
        (0, 500),
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)
    STATIC_FRAME = '_tmp.png'
    cv2.imwrite(STATIC_FRAME, voice_pic)
    
    
    # MoviePy silence video
    
    
    SILENT_VIDEO = '_tmp.mp4'

    # SILENT CLIP

    clip_silent = ImageClip(STATIC_FRAME).set_duration(5)  # as long as the audio - TTS first
    clip_silent.write_videofile(SILENT_VIDEO, fps=24)



  
  
    # fuse vox_full_audiobook.wav & SILENT_VIDEO -> TO FINALLY CONCATENATE into YouTube Video

    # write final output video
    subprocess.call(
        ["ffmpeg",
            "-y",
            "-i",
            SILENT_VIDEO,
            "-i",
            ROOT_DIR + vox_str + '/' + f'{vox_str}_full_audiobook.wav',
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            " 1:a:0",
            ROOT_DIR + vox_str + '/' + f'{vox_str}_full_audiobook.mp4',       #  OUT_FILE
            ])
        
    youtube_video_parts.append(ROOT_DIR + vox_str + '/' + f'{vox_str}_full_audiobook.mp4')
# Final vid for YouTube

with open('_youtube_video_parts.txt', 'w') as f:
    _str = 'file ' + ' \n file '.join(youtube_video_parts)
    f.write(_str)
    
# # list of audiobooks of single vox
# # --
# # $ cat mylist.txt
# # file '/path/to/file1'
# # file '/path/to/file2'
# # file '/path/to/file3'

youtube_video_file = 'audiobook_shift_youtube.mp4'

# ffmpeg -f concat -i video_parts.txt -c copy output.mp4
subprocess.call(
            ["ffmpeg",
                "-y",  # https://stackoverflow.com/questions/39788972/ffmpeg-overwrite-output-file-if-exists
                "-safe",
                "0",  # https://stackoverflow.com/questions/38996925/ffmpeg-concat-unsafe-file-name
                "-f",
                "concat", # https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg
                "-i",
                '_youtube_video_parts.txt',
                "-c",
                "copy",
                youtube_video_file]
            )