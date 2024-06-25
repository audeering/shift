import soundfile
import msinference


text = "Metamorphosis of cultural heritage to augmented hypermedia for accessibility and inclusion."
voice ='en_US/vctk_low#p276'  # For available voices https://audeering.github.io/shift/

voice = voice.replace('/', '_').replace('#', '_').replace('cmu-arctic', 'cmu_arctic').replace('_low', '')
affect = True


# StyleTTS2

style_vector = 'style_vector/' if affect else ''
x = msinference.inference(text,
          msinference.compute_style('assets/wavs/' + style_vector + voice + '.wav'),
                          alpha=0.3,
                          beta=0.7,
                          diffusion_steps=7,
                          embedding_scale=1)
soundfile.write(f'demo_{affect=}.wav', x, 24000)
