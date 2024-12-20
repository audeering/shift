# INCLUSION_IN_MUSEUMS_audiobook.docx
#
# FOR EACH VOICE -> create .wav file per chapter & full audiobook.wav from this .docx
#
# INDIVIDUAL CHAPTERS
#
#   ROOT_DIR/voice/voxstr_CHAPTER_0.wav
#     ..
#   ROOT_DIR/voice/voxstr_CHAPTER_10.wav 
#   ROOT_DIR/voice/voxstr_full_book.wav
#
# FULL BOOK
#
#   ROOT_DIR/full_audiobook_all_voices.wav
import subprocess
import numpy as np
import soundfile
import docx  # pip install python-docx
from pathlib import Path
FS = 24000
ROOT_DIR = './tts_audiobooks/voices/'
Path(ROOT_DIR).mkdir(parents=True,
                     exist_ok=True)
voices = [
        'en_US/hifi-tts_low#9017' ,
        # 'en_US/cmu-arctic_low#jmk',
        # 'en_US/cmu-arctic_low#eey',
        'en_UK/apope_low'
        ]  # select any voice from - https://audeering.github.io/shift/

d = docx.Document('assets/INCLUSION_IN_MUSEUMS_audiobook.docx')  # slightly changed from the original .docx to be audible as by adding extra 'by them from this of etc.'

last_paragraph_was_silence = False  # to know to add silence only once after only at the 1st empty paragraph we detect

chapter_counter = 0  # assure chapters start with CHAPTER: ONCE UPON A TIME

for vox in voices:

    # string (map for assets/)
    
    vox_str = vox.replace(
                '/', '_').replace(
                '#', '_').replace(
                'cmu-arctic', 'cmu_arctic').replace(
                '_low', '')
                
    # create dir for chapter_x.wav & audiobook.wav - for this voice vox
    
    Path(ROOT_DIR + vox_str + '/').mkdir(parents=True,
                                         exist_ok=True)
                
                    
    print(vox)
 
    # for new voice start list of audio tiles making up the 1st chapter of book
 
    total = []
    chapter = []
    
    for para in d.paragraphs[:100]:
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
    
    
# to make an image cv2 text vox (not vox str)
# 
    
   
   
   
   



# CONCAT  ffmpeg   VIDEO PER VOICE TO  SINGLE VIDEO   
   
   
# if each voice makes a pic then we can create a video via ffmpeg for each voice full book .wav and associated pic and via ffmpeg concat all parts to yt video    
                
# # To make one wav per voice per chapter we have to open new file every chapter by finding 'CHAPTER:' at start of para string
# #

# # create ROOTDIR by path
# #
# #
# #
# #
# #
# # 
        
# # call tts to only make .wav per chapter & concat or  also to make vid with still Image ?
# # we have to make Still Image with voice name or we have to screenshot from the book
# # then on kdenlive add pic for every voice;
# #
# # we will not know which voice is per timestep so we have to have a pic per voice perhaps made here
# #
# # So this script creates just audiobook / voice & pic of voice
        
# # make berlin wavs
# import subprocess

# ROOT_DIR = 'audiobooks/voices'



# for vox in voices:
#     vox_str = voice.replace(
#                 '/', '_').replace(
#                 '#', '_').replace(
#                 'cmu-arctic', 'cmu_arctic').replace(
#                 '_low', '')
#                 )
                    
#     print(vox, text_file)
#     subprocess.run(
#             [
#              "python",
#              "tts.py",
#              "--text", 
#              t,         # paragraph text tts and append to voice_chapter.wav
#              # "--affect",
#              #'--image', '_tmp_banner.png',
#              # '--scene', 'calm sounds of castle',
#              '--voice', vox,
#              '--out_file', '_tmp'  # save on _tmp load audio and concat to total
#                 ])

        
# # Final vid for YouTube


# # list of audiobooks of single vox
# # --
# # $ cat mylist.txt
# # file '/path/to/file1'
# # file '/path/to/file2'
# # file '/path/to/file3'


# with open('_single_voice_audiobooks.txt', 'w') as f:
#     _str = 'file ' + ' \n file '.join(list_of_single_voice_audiobooks)
#     f.write(_str)
    

    


# youtube_video_file = 'audiobook_shift_youtube.mp4'

# # ffmpeg -f concat -i video_parts.txt -c copy output.mp4
# subprocess.call(
#             ["ffmpeg",
#                 "-y",  # https://stackoverflow.com/questions/39788972/ffmpeg-overwrite-output-file-if-exists
#                 "-f",
#                 "concat", # https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg
#                 "-i",
#                 "_single_voice_audiobooks.txt",
#                 "-c",
#                 "copy",
#                 youtube_video_file]
#             )