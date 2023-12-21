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

[![Review demo SHIFT](assets/review_demo_thumb.png)](https://www.youtube.com/watch?v=bpt7rOBENcQ)

Generate dubbed video:

```python
python video_dubbing.py  # generate dubbed video from native video & subtitles

```


## Joint Application of D3.1 & D3.2

[![Captions To Video](assets/caption_to_video_thumb.png)](https://youtu.be/wWC8DpOKVvQ)

From an image with caption(s) create a video:

```python

python image_to_speech.py
```
