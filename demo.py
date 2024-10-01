import soundfile
import msinference


text = 'Metamorphosis of cultural heritage to augmented hypermedia for accessibility and inclusion.'
voice = 'en_US/vctk_low#p276'  # see available voices -> https://audeering.github.io/shift/
affect = True  # False ~ high volume & clarity
style = '' if affect else 'v2/'


# StyleTTS2

x = msinference.inference(text,
          msinference.compute_style(
            'assets/wavs/style_vector/' + style + voice.replace(
                '/', '_').replace('#', '_').replace(
                    'cmu-arctic', 'cmu_arctic').replace(
                        '_low', '') + '.wav'),
                          alpha=0.3,
                          beta=0.7,
                          diffusion_steps=7,
                          embedding_scale=1)
soundfile.write(f'demo_{affect=}.wav', x, 24000)
