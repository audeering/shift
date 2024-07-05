# SHIFT - Affective Video & Text to Speech

TTS via [mimic3](https://pypi.org/project/mycroft-mimic3-tts/) for `speaker_style_vector` creation, free of recording and environmental noise artifacts, followed by [StyleTTS2](https://github.com/yl4579/StyleTTS2).

### Installation

```
virtualenv --python=python3 ~/.envs/.my_env
source ~/.envs/.my_env/bin/activate
cd shift/
pip install -r requirements.txt
```

### Usage

Output is saved as `out.wav` or `out.mp4`.

**Text 2 Speech**

```python
#
python tts.py --text sample.txt

# see Available Voices (Affective)
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann"

# See Available Voices
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann" --noaffect

# voice cloning
python tts.py --text sample.txt --native assets/native_voice.wav
```

**Image 2 Speech**

```python
# Image Narration - All above TTS args apply also here!
python tts.py --text sample.txt --image assets/image_from_T31.jpg
```

**Video 2 Speech**

```python
# Video Dubbing - from time-stamped subtitles (.srt)
python tts.py --text assets/head_of_fortuna_en.srt --video assets/head_of_fortuna_en.mp4

# Video Storytell - from text description (.txt)
python tts.py --text assets/head_of_fortuna_GPT.txt --video assets/head_of_fortuna_en.mp4
```



## Available voices - Emotion

<a href="https://audeering.github.io/shift/">Listen to available voices!</a>

## Examples

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