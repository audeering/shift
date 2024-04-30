# 1. generate_config  -> saves prime voices.json
# 2. update_config_with_emotion_centroid.py
# 3. voice_table.py -> Update REadme

# AGENT
# https://serverfault.com/questions/672346/straight-forward-way-to-run-ssh-agent-and-ssh-add-on-login-via-ssh


# 1. generate voices.json (without emotion ) - very fast
import json
import matplotlib.pyplot as plt

N_PIX = 9
GLOBAL_FILE = 'voices_s.json'

def pixel_to_cube(adv=None):
    # x = .0000001 * np.ones((N_PIX, N_PIX, N_PIX))
    # ar, do, va = np.array([.5, .5, .5]) -.5
    ar, do, va = np.array(adv)-.5


    _grid = np.linspace(-1, 1, N_PIX)
    _grid = (_grid[:, None, None] - ar) ** 2 + (_grid[None, None, :] - do) ** 2 + (_grid[None, :, None] - va) ** 2
    # this has mean .5 for the gaussian so has to subtract the mean
    # [x, y, z]*1*[x-xm, y-ym, z-zm] = (x-xm)**2 + (y-ym)**2 + (z..)
    gauss = np.exp(-_grid*40)  # is a pmf
    # gauss /= gauss.sum()
    # x = (x + gauss).clip(0, 1)
    return gauss




def explode(data):
    '''replicate 16 x 16 x 16 cube to edges array 31 x 31 x 31'''
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def visual_cube(_h, png_file='cube3D.png', fig_title=''):
    '''_h = cuda tensor (N_PIX, N_PIX, N_PIX)'''
    filled = np.ones((N_PIX, N_PIX, N_PIX), dtype=bool)

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    # x[0::2, :, :] += 0.2
    # y[:, 0::2, :] += 0.2
    # z[:, :, 0::2] += 0.2   # val offset
    x[1::2, :, :] += 1 #0.95   # arousal size
    y[:, 1::2, :] += 1 #0.95   # dom size
    z[:, :, 1::2] += 1 #0.95  # val size

    ax = plt.figure().add_subplot(projection='3d')
    # HEAT = np.random.rand(N_PIX, N_PIX, N_PIX)

    # _h = (100 * _h).clip(0, 1)

    f_2 = np.ones([2 * N_PIX - 1,
                   2 * N_PIX - 1,
                   2 * N_PIX - 1, 4], dtype=np.float64)
    f_2[:, :, :, 3] = explode(_h)
    cm = plt.get_cmap('cool') #gist_rainbow' coolwarm') #
    f_2[:, :, :, :3] = cm(f_2[:, :, :, 3])[..., :3]
    # so now we want the color to be the attribute not the alpha

    f_2[:, :, :, 3] = f_2[:, :, :, 3].clip(.014, .7)
    # f_2[:, :, :, 3] *= (f_2[:, :, :, 3] > .24)  # deflate transparent almost zero pixels unclutter

    print(f_2.shape, 'f_2 AAAA')
    ecolors_2 = f_2
    # ecolors_2[:, :, :, :3] = 0
    N_TICK = 7
    ax.voxels(x, y, z, filled_2, facecolors=f_2, edgecolors=.006 * ecolors_2)
    ax.set_aspect('equal')
    ax.set_zticklabels([f'{n/N_TICK:.2f}' for n in ax.get_zticks()])
    ax.set_zlabel('valence', fontsize=11, labelpad=-160)
    # ax.zaxis.set_label_coords(-10.1,-1.02, -5)
    ax.set_xticklabels([f'{n/N_TICK:.2f}' for n in ax.get_xticks()])
    ax.set_xlabel('arousal', fontsize=11, labelpad=7)
    ax.set_yticklabels([f'{n/N_TICK:.2f}' for n in ax.get_yticks()], rotation=275)
    ax.set_ylabel('dominance', fontsize=11, labelpad=10)
    ax.set_title(fig_title, color='b', fontsize=10)
    plt.locator_params(nbins=N_TICK)
    plt.savefig(png_file, dpi=300, format=png_file.split('.')[-1], bbox_inches='tight')
    plt.close()
    # plt.show()













NOISE_SCALE = .667
NOISE_W = .9001 #.8 #.90001  # default .8 in __main__.py @ L697    IGNORED DUE TO ARTEfACTS - FOR NOW USE default

a = [
    'p239',
    'p236',
    'p264',
    'p250',
    'p259',
    'p247',
    'p261',
    'p263',
    'p283',
    'p274',
    'p286',
    'p276',
    'p270',
    'p281',
    'p277',
    'p231',
    'p238',
    'p271',
    'p257',
    'p273',
    'p284',
    'p329',
    'p361',
    'p287',
    'p360',
    'p374',
    'p376',
    'p310',
    'p304',
    'p340',
    'p347',
    'p330',
    'p308',
    'p314',
    'p317',
    'p339',
    'p311',
    'p294',
    'p305',
    'p266',
    'p335',
    'p334',
    'p318',
    'p323',
    'p351',
    'p333',
    'p313',
    'p316',
    'p244',
    'p307',
    'p363',
    'p336',
    'p312',
    'p267',
    'p297',
    'p275',
    'p295',
    'p288',
    'p258',
    'p301',
    'p232',
    'p292',
    'p272',
    'p278',
    'p280',
    'p341',
    'p268',
    'p298',
    'p299',
    'p279',
    'p285',
    'p326',
    'p300',
    's5',
    'p230',
    'p254',
    'p269',
    'p293',
    'p252',
    'p345',
    'p262',
    'p243',
    'p227',
    'p343',
    'p255',
    'p229',
    'p240',
    'p248',
    'p253',
    'p233',
    'p228',
    'p251',
    'p282',
    'p246',
    'p234',
    'p226',
    'p260',
    'p245',
    'p241',
    'p303',
    'p265',
    'p306',
    'p237',
    'p249',
    'p256',
    'p302',
    'p364',
    'p225',
    'p362']

print(len(a))

b = {}

for row in a:
    b[f'en_US/vctk_low#{row}'] = {'rate': 1.14,
                                  'pitch': 1.04,
                                  'noise_w': NOISE_W,
                                  'noise_scale': NOISE_SCALE}

# print(b)

# 00000000 arctic


a = ['awb'
        'rms',
        'slt',
        'ksp',
        'clb',
        'aew',
        'bdl',
        'lnh',
        'jmk',
        'rxr',
        'fem',
        'ljm',
        'slp',
        'ahw',
        'axb',
        'aup',
        'eey',
        'gka']


for row in a:
    b[f'en_US/cmu-arctic_low#{row}'] = {'rate': 1.14,
                                        'pitch': 1.04,
                                        'noise_w': NOISE_W,
                                        'noise_scale': NOISE_SCALE}


# HIFItts

a = ['9017',
     '6097',
     '92']

for row in a:
    b[f'en_US/hifi-tts_low#{row}'] = {'rate': 1.14,
                                      'pitch': 1.04,
                                      'noise_w': NOISE_W,
                                      'noise_scale': NOISE_SCALE}


a = [
    'elliot_miller',
    'judy_bieber',
    'mary_ann']

for row in a:
    b[f'en_US/m-ailabs_low#{row}'] = {'rate': 1.04,
                                      'pitch': 1.04,
                                      'noise_w': NOISE_W,
                                      'noise_scale': NOISE_SCALE}

# LJspeech single speaker

b[f'en_US/ljspeech_low'] = {'rate': 1.14,
                            'pitch': 1.04,
                            'noise_w': NOISE_W,
                            'noise_scale': NOISE_SCALE}

# en_UK apope - only speaker

b[f'en_UK/apope_low'] = {'rate': 1.14,
                        'pitch': 1.04,
                        'noise_w': NOISE_W,
                        'noise_scale': NOISE_SCALE}

# all_voices_str = 'VOICES = ' + json.dumps(b, indent=4)
with open(GLOBAL_FILE, 'w') as f:
    json.dump({'voices': b}, f, indent=4)

# -- END 1.

# 2. TTS Harvard sentences & pred centroid emotion -- for every voice -- save figs in assets -- DOES NOT SAVE wavs/voice
# mean emotion of each voice should be generated by running generate_mean_emotion.py

from pathlib import Path
# from config import VOICES
import csv
import json
import io
import os
import typing
import wave
import time
# import audonnx
import seaborn as sns
import matplotlib.pyplot as plt
import audresample
import soundfile
import numpy as np
import re
import msinference

with open(GLOBAL_FILE, 'r') as f:
    VOICES = json.load(f)['voices']

# CREATE TTS wav & README TABLE of VOICES


# if not os.path.is_dir('./wavs'):

# -- https://github.com/audeering/w2v2-how-to
# url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
# cache_root = audeer.mkdir('cache')
# model_root = audeer.mkdir('model')
# archive_path = audeer.download_url(url, cache_root, verbose=True)
# audeer.extract_archive(archive_path, model_root)
# --

# -- http://models.pp.audeering.com/speech-emotion-recognition/valence/90398682-2.0.0/
# import audmodel
# model_root = audmodel.load('90398682-2.0.0')
# --

# emotion_predictor = audonnx.load(model_root, device='cuda')
#=================================================== DAWN
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits



# load model from hub

model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to('cuda').eval()

# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


def emotion_predictor(
    x,
    sampling_rate,
    embeddings= False):
    

    
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to('cuda')

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y















# ===================================================DAWN


out_dir = 'assets/'
Path(out_dir).mkdir(parents=True, exist_ok=True)

sentences = [
    'Sweet dreams are made of this, .. !!! # I travel the world and the seven seas.',
    'The birch canoe slid on the smooth planks.',
    'Glue the sheet to the dark blue background.',
    'It\'s easy to tell the depth of a well.',
    'These days a chicken leg is a rare dish.',
    'Rice is often served in round bowls.',
    'The juice of lemons makes fine punch.',
    'The box was thrown beside the parked truck.',
    'The hogs were fed chopped corn and garbage.',
    'Four hours of steady work faced us.',
    'A large size in stockings is hard to sell.',
    # 'List 2',
    'The boy was there when the sun rose.',
    'A rod is used to catch pink salmon.',
    'The source of the huge river is the clear spring.',
    'Kick the ball straight and follow through.',
    'Help the woman get back to her feet.',
    'A pot of tea helps to pass the evening.',
    'Smoky fires lack flame and heat.',
    'The soft cushion broke the man\'s fall.',
    'The salt breeze came across from the sea.',
    'The girl at the booth sold fifty bonds.',
    # 'List 3',
    'The small pup gnawed a hole in the sock.',
    'The fish twisted and turned on the bent hook.',
    'Press the pants and sew a button on the vest.',
    'The swan dive was far short of perfect.',
    'The beauty of the view stunned the young boy.',
    'Two blue fish swam in the tank.',
    'Her purse was full of useless trash.',
    'The colt reared and threw the tall rider.',
    'It snowed, rained, and hailed the same morning.',
    'Read verse out loud for pleasure.',
    # 'List 4',
    'Hoist the load to your left shoulder.',
    'Take the winding path to reach the lake.',
    'Note closely the size of the gas tank.',
    'Wipe the grease off his dirty face.',
    'Mend the coat before you go out.',
    'The wrist was badly strained and hung limp.',
    'The stray cat gave birth to kittens.',
    'The young girl gave no clear response.',
    'The meal was cooked before the bell rang.',
    'What joy there is in living.',
    # https://www.cs.columbia.edu/~hgs/audio/harvard.html
             ]


emotion_per_voice = {}  # {'vctk_low#p246': np.zeros(len(sentences), 3)}

for _voice in list(VOICES.keys())[:2]:
    
    str_voice = _voice.replace('/', '_').replace('#', '_').replace('_low', '')
    
    fig_file = out_dir + str_voice + '.png'

    

    

    
    

    if 'cmu-arctic' in str_voice:
        wav_path = out_dir + '/wavs/' + str_voice.replace('cmu-arctic', 'cmu_arctic') + '.wav'
    else:
        wav_path = out_dir + '/wavs/' + str_voice + '.wav' # [...cmu-arctic...](....cmu_arctic....wav) # README has underscore in cmu_arctic.wav
    


    print(_voice)
    if 1: #not os.path.isfile(fig_file):

        all_emotions_voice_k = np.zeros((len(sentences), 3))

        for i_sentence, _text_ in enumerate(sentences):
            print(f'  > [{i_sentence}]   {_text_}')


            _text_ = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               [,.;@#?!&$ ]*  #  --- perhaps even more punctuation plus spaces
               """,
               ", ",          # and replace it with a single space
               _text_, flags=re.VERBOSE)
            _text_ = _text_[:-2] + '.'



            x = msinference.inference(_text_,
                                    msinference.compute_style(wav_path), #'tgt_spk.wav'),
                                    alpha=0.3,
                                    beta=0.7,
                                    diffusion_steps=7,
                                    embedding_scale=1)


            # # predict emotion of synthesized speech

            # x, fs = soundfile.read('_tmp.wav')
            x = audresample.resample(x.astype(np.float32),
                                     16000,
                                     24000)  # state.sample_rate_hz
            all_emotions_voice_k[i_sentence, :] = emotion_predictor(x, 16000)[0]  #['logits'][0]

        # --
        # after tts all sentences - plot
        #
        # For German Fr etc voices we will not have StyleTTS
        # 
        #
        #
        
        centroid_emotion = all_emotions_voice_k.mean(0) # Add all emotions to the cube?

        VOICES[_voice]['emotion'] = centroid_emotion.tolist()

        # print(all_emotions_voice_k.shape)
        # ==
        # fill neightbor pixels
        cube_fill = np.zeros((N_PIX, N_PIX, N_PIX))
        for adv in all_emotions_voice_k:
            # cube_fill[int(a*N_PIX), int(d*N_PIX), int(v*N_PIX)] += 1
            print(adv)
            cube_fill += pixel_to_cube(adv=adv)
        # cube_fill /= cube_fill.sum()
        # -- PLOT

        # CUBE
        print('Making cube\n')
        visual_cube(
                   (24 * cube_fill), #clip(0, 1),
                   png_file=fig_file, fig_title=_voice)
            

    # -- END FIG EXISTS - THUS VOICE IS ALREADY WITH EMOTION INSIDE voices.json


# save dict of voice; all_emotions_voice_k


with open(GLOBAL_FILE, 'w') as f:
    json.dump({'voices': VOICES}, f, indent=4)




# -- END 2.


# 3. RUN voice_table to update README.md (./assets/*.png) SHOULD EXIST