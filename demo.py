import subprocess
import soundfile
import sox

## VOICES mimic3
# 
# /scratch/dkounadis/.envs/.tts/lib/python3.8/site-packages/mimic3_tts/

spk1 = 'en_US/vctk_low#p236'
rate1 = 1.24

spk2 = 'en_UK/apope_low'
rate2 = 1.64

pitch_semitones = -4

text = ('<speak>'
'<prosody volume=\'64\'>'
f'<prosody rate=\'{rate1}\'>'
f'<voice name=\'{spk1}\'>'
'<s>'
'A an exemplary voice.'
'</s>'
'</voice>'
'</prosody>'
'</prosody>'
f'<prosody rate=\'{rate2}\'>'
f'<voice name=\'{spk2}\'>'
'<s>'
'.Another pleasant voice.'
'</s>'
'</voice>'
'</prosody>'
'</speak>')

with open('_tmp_ssml.txt', 'w') as f:
    f.write(text)

raw_tts = 'test.wav'
ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > {raw_tts}', shell=True)
ps.wait()

x, fs = soundfile.read(raw_tts)
tfm = sox.Transformer()
tfm.pitch(pitch_semitones)
x_shift = tfm.build_array(
    input_array=x,
    sample_rate_in=fs)

soundfile.write(f'test_pitch.wav', x_shift, fs)
