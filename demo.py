import subprocess
import soundfile
import msinference



my_text = "Metamorphosis of cultural heritage to augmented hypermedia for accessibility and inclusion."
_voice = 'en_US/vctk_low#p276' # https://audeering.github.io/shift/
affect = True                  # False = Non-Affective voices
out_wav = f'example_{affect=}.wav'


if affect:

    # Mimic-3

    reference_wav = '_spk.wav'
    rate = 4  # high speed sounds nice when used as speaker-reference audio for 2nd stage (StyleTTS2)
    _ssml = (
        '<speak>'
        f'<prosody volume=\'24\'>'
            f'<prosody rate=\'{rate}\'>'
            f'<voice name=\'{_voice}\'>'
            f'<s>Sweet dreams are made of this, ... !!! I travel the world and the seven seas.</s>'
            '</voice>'
            '</prosody>'
            '</prosody>')
    _ssml += '</speak>'
    with open('_tmp_ssml.txt', 'w') as f:
        f.write(_ssml)
    ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > {reference_wav}', shell=True)
    ps.wait()  # using ps to call mimic3 because samples dont have time to be written in stdout buffer

    # StyleTTS2

    x = msinference.inference(my_text,
                            msinference.compute_style(reference_wav),
                            alpha=0.3,
                            beta=0.7,
                            diffusion_steps=7,
                            embedding_scale=1)
    soundfile.write(out_wav, x, 24000)



else:



    # Non Affective TTS
     
    rate = .84
    _ssml = (
        '<speak>'
        f'<prosody volume=\'94\'>'
            f'<prosody rate=\'{rate}\'>'
            f'<voice name=\'{_voice}\'>'
            f'<s>\'{my_text}\'</s>'
            '</voice>'
            '</prosody>'
            '</prosody>')
    _ssml += '</speak>'
    with open('_tmp_ssml.txt', 'w') as f:
        f.write(_ssml)
    ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > {out_wav}', shell=True)
    ps.wait()

