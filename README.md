# Shift TTS System

Affective Speech Synthesis via [mimic3](https://pypi.org/project/mycroft-mimic3-tts/) and [Speech emotion recognition](https://github.com/audeering/w2v2-how-to).

### Installation

```
virtualenv --python=python3 ~/.envs/.my_env
source ~/.envs/.my_env/bin/activate
cd shift/
pip install -r requirements.txt
```

### Use

```python

# TTS
python tts.py --text sample.txt

# TTS - voice selection - Available Voices
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann"

# TTS - clone native voice's emotion
python tts.py --text sample.txt --native_voice assets/native_voice.wav
```

Output wav is saved as `out.wav`

##

# Available voices - english

<a href="https://audeering.github.io/shift/">Listen to available voices and visualize their emotion!</a>

# Switch Native voice with English TTS

Native voice video

[![Native voice ANBPR video](assets/native_video_thumb.png)](https://www.youtube.com/watch?v=tmo2UbKYAqc)

##

Same video where Native voice is replaced with English TTS voice with similar emotion


[![Same video w. Native voice replaced with English TTS](assets/tts_video_thumb.png)](https://www.youtube.com/watch?v=geI1Vqn4QpY)


## Review demo - SHIFT

https://www.youtube.com/watch?v=bpt7rOBENcQ

Generate your own dubbed video:

```python
python video_dub.py

```