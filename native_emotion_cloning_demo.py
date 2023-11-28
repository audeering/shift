# -*- coding: utf-8 -*-
import re
import numpy as np
import soundfile
import json
import pandas as pd
import urllib.request
import zipfile
import os
import onnxruntime
import argparse
import audresample
import subprocess

def start_emotion_inference_session(tmp_zip='_onnx_.zip'):
    '''1. Download audEERINGs emotion recognition

          https://github.com/audeering/w2v2-how-to

       2. Extract onnx from zip

       3. Find Input / Output node names of onnx (Optional)

       4. Start inferece session - onnxRuntime


    '''
    _path_ = '_onnx_/'

    onnx_file = _path_ + 'model.onnx/model.onnx'

    if not os.path.isfile(tmp_zip):
        urllib.request.urlretrieve(
            'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip', tmp_zip)

    if not os.path.isfile(onnx_file):
        with zipfile.ZipFile(tmp_zip, 'r') as zf:
            m = zf.infolist()
            for member in m:
                zf.extract(member, onnx_file)
            # member_names = [m.filename for m in members]
 

    # ---- find onnx node names
    # import onnx
    # model = onnx.load(onnx_file)
    # output =[node.name for node in model.graph.output]
    # input_all = [node.name for node in model.graph.input]
    # input_initializer =  [node.name for node in model.graph.initializer]
    # net_feed_input = list(set(input_all)  - set(input_initializer))
    # print('Inputs: ', net_feed_input)
    # print('Outputs: ', output)
    # ----


    sess = onnxruntime.InferenceSession(onnx_file,
           # providers=[('CUDAExecutionProvider',
           #             {'device_id': dev.index})],
           providers=['CPUExecutionProvider'])
    return sess



# == text preprocessor

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'



def split_into_sentences(text):
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]

    https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def build_ssml(text):
    spk = find_nearest_voice(text['sentiment'])
    _s = '<speak>'
    for sentence in split_into_sentences(text['content']):
        rate = min(max(.87, len(sentence) / 76), 1.14) #1.44)  # 1.24 for bieber
        # print('\n__=__\n', sentence, '\n__=__\n')
        print(rate, len(sentence) / 76)
        volume = int(74 * np.random.rand() + 24)
        # text = ('<speak>'
        _s += f'<prosody volume=\'{volume}\'>'   # THe other voice does not have volume
        _s += f'<prosody rate=\'{rate}\'>'
        _s += f'<voice name=\'{spk}\'>'
        _s += '<s>'
        _s += sentence
        _s += '</s>'
        _s += '</voice>'
        _s += '</prosody>'
        _s += '</prosody>'

    # # ========================= DIFFERENT SENTECE WITH MOD RATE
    # f'<prosody rate=\'{rate2}\'>'
    # f'<voice name=\'{spk2}\'>'
    # '<s>'
    # '.Another pleasant voice.'
    # '</s>'
    # '</voice>'
    # '</prosody>'

    _s += '</speak>'

    with open('_tmp_ssml.txt', 'w') as f:
        f.write(_s)



def find_nearest_voice(sentiment):
    with open('voices.json', 'r') as f:
        raw = json.load(f)['voices']

    # load voices.json
    vox = []
    emo = []
    for k, v in raw.items():
        vox.append(k)
        emo.append(v["emotion"])
        # print(k)

    df = pd.DataFrame(index=vox, data=emo, columns=['arousal', 'dominance', 'valence'])

    ix = df.iloc[(df['arousal'] - sentiment).abs().argsort()[:1]]  # sentiment = .4

    print(ix.index.item(), 'AA')
    return ix.index.item()





# =====================================================


def main(args):
    sess = start_emotion_inference_session()

    x, fs = soundfile.read(args.native_voice)
    x = x[:, 0]  # only need mono
    x = audresample.resample(x.astype(np.float32),
                                         16000,
                                         fs)  # - Emotion Recognition needs 16kHz

    print(f'recognizing emotion from wav={args.native_voice} ..')

    valence = sess.run(['hidden_states', 'logits'], {'signal': x})[1][0, 2]
    # [1][2]

    print(valence, '\n')

    text = {'content': ('Knjаževаc, a small town in Serbia, has a rich history dating back to the prehistoric era.'
                   'The town was inhabited by various tribes, including the Triballi, Moesi, Thracians, and Timachi.'
                   'In the 1st century AD, the Romans conquered the region, and during the Migration Period,'
                   'the Avars, Huns, and Slavs passed through,'
                   ),
        'sentiment': valence}  # valence used as sentiment


    build_ssml(text)
    raw_tts = 'emocloned_example.wav'
    ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > {raw_tts}', shell=True)
    ps.wait()





def command_line_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--device',
        help="Device ID",
        type=str,
        default='cpu',
    )

    parser.add_argument(
        '--native_voice',
        help="wav (~4 seconds) used to detect emotion, to choose optimal TTS voice",
        default='assets/native_voice_FOR_EMOTION_CLONING.wav',
        type=str,
    )

    parser.add_argument(
        '--few_classes',
        help="set if scene prediction is done without sound event prediction",
        action='store_true',
    )

    return parser


def cli():
    parser = command_line_args()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()
