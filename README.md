# SHIFT - TTS System

This repository implements the Video & Text to Affective Speech system for the [SHIFT Project](https://shift-europe.eu/). The system transforms text & silent videos/images into expressive, affective speech, enhancing accessibility and engagement for cultural heritage (CH) institutions.

##

Video Text to Affective Speech system is based on [StyleTTS2](https://github.com/yl4579/StyleTTS2)
for English, with optional support for non-affective TTS of [other langauges](https://github.com/MycroftAI/mimic3-voices) via the [mimic3](https://pypi.org/project/mycroft-mimic3-tts/) system.

**Available Voices**

<a href="https://audeering.github.io/shift/">Listen available voices!</a>

**Other Languages (Non-emotional)**

Although this system builds Affective TTS only for English. For other languages, use any of the [Mimic3 Voices](https://github.com/MycroftAI/mimic3-voices), it will automatically call Mimic3 for non-English voices.

## Installation

```
virtualenv --python=python3 ~/.envs/.my_env
source ~/.envs/.my_env/bin/activate
cd shift/
pip install -r requirements.txt
```

### Functions

Output is saved in `out.wav/.mp4`.

**Text 2 Speech**

```python
# Basic TTS - See Available Voices
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann" --affective

# voice cloning
python tts.py --text sample.txt --native assets/native_voice.wav
```

**Image 2 Video**

```python
# Make video narrating an image - All above TTS args apply also here!
python tts.py --text sample.txt --image assets/image_from_T31.jpg
```

**Video 2 Video**

```python
# Video Dubbing - from time-stamped subtitles (.srt)
python tts.py --text assets/head_of_fortuna_en.srt --video assets/head_of_fortuna.mp4

# Video narration - from text description (.txt)
python tts.py --text assets/head_of_fortuna_GPT.txt --video assets/head_of_fortuna.mp4
```

## Examples

Native voice video

[![Native voice ANBPR video](assets/native_video_thumb.png)](https://www.youtube.com/watch?v=tmo2UbKYAqc)

##

Same video where Native voice is replaced with English TTS voice with similar emotion


[![Same video w. Native voice replaced with English TTS](assets/tts_video_thumb.png)](https://www.youtube.com/watch?v=geI1Vqn4QpY)


## Video Dubbing

[![Review demo SHIFT](assets/review_demo_thumb.png)](https://www.youtube.com/watch?v=bpt7rOBENcQ)

Generate dubbed video:


```python
python tts.py --text assets/head_of_fortuna_en.srt --video assets/head_of_fortuna.mp4

```


## Joint Application of D3.1 & D3.2

[![Captions To Video](assets/caption_to_video_thumb.png)](https://youtu.be/wWC8DpOKVvQ)

From an image with caption(s) create a video:

```python

python tts.py --text sample.txt --image assets/image_from_T31.jpg
```
