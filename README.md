# SHIFT TTS System

Affective TTS tool for [SHIFT Horizon](https://shift-europe.eu/) using [this phenomenon](https://huggingface.co/dkounadis/artificial-styletts2/discussions/2). Synthesize speech from text or subtitles `.srt` and overlays it to videos.
  - Has [134 build-in voices](https://audeering.github.io/shift/) tuned for [StyleTTS2](https://github.com/yl4579/StyleTTS2) for English
  - Supports [foreing langauges](https://github.com/audeering/shift/blob/main/Utils/all_langs.tsv) via [mms-tts](https://huggingface.co/spaces/mms-meta/MMS))
  - A Beta Version of this tool for TTS & audio soundscape is [build here](https://huggingface.co/dkounadis/artificial-styletts2)

### Available Voices

<a href="https://audeering.github.io/shift/">Listen to available voices!</a>

## Install

```
virtualenv --python=python3 ~/.envs/.my_env
source ~/.envs/.my_env/bin/activate
cd shift/
pip install -r requirements.txt
```

Demo. TTS output saved as `out.wav`

```
CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=./hf_home CUDA_VISIBLE_DEVICES=0 python demo.py
```

## API

Start Flask server

```
CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=./hf_home CUDA_VISIBLE_DEVICES=0 python api.py
```

## Inference

The following needs `api.py` to be already running `e.g. on tmux session`

**Text 2 Speech**

```python
# Basic TTS - See Available Voices
python tts.py --text sample.txt --voice "en_US/m-ailabs_low#mary_ann" --affective

# voice cloning
python tts.py --text sample.txt --native assets/native_voice.wav
```

**Native voice 2 (english, affective) TTS**

```
python tts.py --voice "en_US/m-ailabs_low#mary_ann"  --video assets/anbpr.webm --text assets/anbpr.en.srt
```

[![Native voice > TTS (en)](assets/native_video_thumb.png)](https://youtu.be/9tecQ6amHaY)

**Native voice 2 (romanian) TTS**

```
python tts.py --voice romanian --video assets/anbpr.webm --text assets/anbpr.ro.srt
```

[![Native voice > TTS (ro)](assets/tts_video_thumb.png)](https://youtu.be/6bYcD2IZvoU)


**Native voice 2 (serbian) TTS**

[![Review demo SHIFT](assets/review_demo_thumb.png)](https://www.youtube.com/watch?v=bpt7rOBENcQ)

Generate dubbed video:

```python
# Video Dubbing - from time-stamped subtitles (.srt)
python tts.py --voice serbian --text assets/head_of_fortuna_en.srt --video assets/head_of_fortuna.mp4

# Video narration - from text description (.txt)
python tts.py --text assets/head_of_fortuna_GPT.txt --video assets/head_of_fortuna.mp4
```

**Image 2 Video**

```python
# Make video narrating an image
python tts.py --text sample.txt --image assets/image_from_T31.jpg --voice en_US/cmu-arctic_low#jmk
```

[![Captions To Video](assets/caption_to_video_thumb.png)](https://youtu.be/wWC8DpOKVvQ)


