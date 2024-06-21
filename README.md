# SHIFT - Affective Video/Text to Speech

Affective TTS via [mimic3](https://pypi.org/project/mycroft-mimic3-tts/) and [StyleTTS2](https://github.com/yl4579/StyleTTS2) and [Speech emotion recognition](https://github.com/audeering/w2v2-how-to).

### Installation

```
virtualenv --python=python3 ~/.envs/.my_env
source ~/.envs/.my_env/bin/activate
cd shift/
pip install -r requirements.txt
```

### Usage

```python
# Output is saved as out.wav or out.mp4

# TTS

# 1.
python tts.py --text sample.txt

# 2. - see Available Voices (Affective)
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann"

# 3. - see Available Voices
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann" --noaffect

# 4. - voice cloning
python tts.py --text sample.txt --native_voice assets/native_voice.wav

# Image2Speech

# 5. - All TTS args can be used here!
python tts.py --text sample.txt --image assets/image_from_T31.jpg

# Video2Speech

# 6. - Video Dubbing / Voice Cloning
python tts.py --text assets/head_of_fortuna_en.srt --video assets/head_of_fortuna_en.mp4
```



# Listen Available voices

<a href="https://audeering.github.io/shift/">Listen to available voices and visualize their emotion!</a>

# Examples

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