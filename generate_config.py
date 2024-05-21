# 1. write available mimic33 voices.json (without emotion ) - very fast
# 2. synthesize .wavs .png & emotion_prediction -> append to voices.json
# 3. write REadme.md

import json
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import csv
import json
import io
import os
import typing
import wave
import time
import seaborn as sns
import matplotlib.pyplot as plt
import audresample
import soundfile
import numpy as np
import re


out_dir = 'assets/'
Path(out_dir).mkdir(parents=True, exist_ok=True)
wav_dir = 'assets/wavs/'
Path(wav_dir).mkdir(parents=True, exist_ok=True)
N_PIX = 11
GLOBAL_FILE = 'voices.json'

def pixel_to_cube(adv=None):
    # x = .0000001 * np.ones((N_PIX, N_PIX, N_PIX))
    # ar, do, va = np.array([.5, .5, .5]) -.5
    ar, do, va = np.array(adv)-.5


    _grid = np.linspace(-1, 1, N_PIX)
    _grid = (_grid[:, None, None] - ar) ** 2 + (_grid[None, None, :] - do) ** 2 + (_grid[None, :, None] - va) ** 2
    # this has mean .5 for the gaussian so has to subtract the mean
    # [x, y, z]*1*[x-xm, y-ym, z-zm] = (x-xm)**2 + (y-ym)**2 + (z..)
    gauss = np.exp(-_grid*244)  # is a pmf
    # gauss /= gauss.sum()
    # x = (x + gauss).clip(0, 1)

    # anything above the gauss max point 
    return gauss




def explode(data):
    '''replicate 16 x 16 x 16 cube to edges array 31 x 31 x 31'''
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def visual_cube(_h, png_file='cube3D.png', fig_title=''):
    '''_h = cuda tensor (N_PIX, N_PIX, N_PIX)


    if I make the x 1-x then I can revert the label Hi -> Lo
     '''
    # _h[N_PIX-1, N_PIX-1, N_PIX-1] = 1
    # half = int((N_PIX)/2)
    # _h[half, :half, :] = .9
    # _h[half, half:, :] = .01


    # neg = np.zeros_like(_h) #.copy()
    # neg[half, half:, :] = -5
    # neg[half:, half, :] = -5
    # neg[half, :half, :] = -3    










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
    # Plan vertical
    half = N_PIX / 2 -.05
    _DO = np.array([half, N_PIX+.74])
    _VA = np.array([-.24, half])
    _VA, _DO = np.meshgrid(_DO, _VA)
    _AR = half * np.ones_like(_VA)
    # plan
    GR = [104/255.,104/255., 104/255.]
    ax.plot_surface(_AR, _VA, _DO, color=GR, #cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=.16)
    # plan horz
    _DO = np.array([half+.1, N_PIX+.7])
    _AR = np.array([half+.1, N_PIX+.7])
    _AR, _DO = np.meshgrid(_AR, _DO)
    _VA = half * np.ones_like(_DO)
    ax.plot_surface(_AR, _DO, _VA, color=GR, #cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=.47)
# Plan left
    half = N_PIX / 2 -.05
    _AR = _VA = np.array([-.24, half])  # plane vertical on half low LHS
    _VA, _AR = np.meshgrid(_AR, _VA)
    _DO = half * np.ones_like(_VA)
    # plan
    ax.plot_surface(_VA, _DO, _AR, color=GR, #cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=.14)


    f_2 = np.ones([2 * N_PIX - 1,
                   2 * N_PIX - 1,
                   2 * N_PIX - 1, 4], dtype=np.float64)
    f_2[:, :, :, 3] = explode(_h)
    cm = plt.get_cmap('cool') #gist_rainbow' coolwarm') #
    f_2[:, :, :, :3] = cm(f_2[:, :, :, 3])[..., :3]
    # so now we want the color to be the attribute not the alpha

    f_2[:, :, :, 3] = f_2[:, :, :, 3].clip(.0009, .97)
    # f_2[:, :, :, 3] *= (f_2[:, :, :, 3] > .24)  # deflate transparent almost zero pixels unclutter

    ecolors_2 = f_2
    # ecolors_2[:, :, :, :3] = 0
    
    ax.voxels(x, y, z, filled_2, facecolors=f_2, edgecolors=.006 * ecolors_2)
    ax.set_aspect('equal')
    #
    ax.set_zticklabels(['Low', '', 'High'])
    ax.set_zticks([0, N_PIX/2-.05, N_PIX])
    # ax.set_zticklabels([f'{n/N_TICK:.2f}' for n in ax.get_zticks()])
    ax.set_zlabel('Valence', fontsize=11, labelpad=-11, color=[.4,.4,.4])
    # ax.zaxis.set_label_coords(-10.1,-1.02, -5)
    # ax.set_xticklabels([f'{n/N_TICK:.2f}' for n in ax.get_xticks()])
    ax.set_xticklabels(['Low', 'Neutral', 'High'])
    ax.set_xticks([-.1, N_PIX/2-.05, N_PIX])
    ax.set_xlabel('Arousal', fontsize=11, labelpad=-2, color=[.4,.4,.4])
    # ax.set_yticklabels([f'{n/N_TICK:.2f}' for n in ax.get_yticks()], rotation=275)
    ax.set_yticklabels(['Low', '', 'High'])
    ax.set_yticks([-.1, N_PIX/2-.05, N_PIX])
    ax.set_ylabel('Dominance', fontsize=11, labelpad=-14, color=[.4,.4,.4])
    ax.set_title(fig_title, color='b', fontsize=10)
    #plt.savefig(png_file, dpi=300, format=png_file.split('.')[-1], bbox_inches='tight')
    #plt.close()
    
    ax.view_init(elev=45, azim=-125)  # nic
    # ax.view_init(elev=14, azim=-104)  # nic

    ax.tick_params(axis='x', which='major', pad=-4, labelcolor=[.5,.5,.5])
    ax.tick_params(axis='y', which='major', pad=0, labelcolor=[.5,.5,.5])
    ax.tick_params(axis='z', which='major', pad=4, labelcolor=[.5,.5,.4])
    # plt.show()
    plt.savefig(png_file, format=png_file.split('.')[-1], bbox_inches='tight') # , dpi=300
    plt.close()











# =======================================================================
# S T A R T                 G E N E R A T E   png/wav
# =======================================================================
if not os.path.isfile(GLOBAL_FILE):
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

    b = []

    for row in a:
        b.append(f'en_US/vctk_low#{row}')

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
        b.append(f'en_US/cmu-arctic_low#{row}')

    # HIFItts

    a = ['9017',
        '6097',
        '92']

    for row in a:
        b.append(f'en_US/hifi-tts_low#{row}')

    a = [
        'elliot_miller',
        'judy_bieber',
        'mary_ann']

    for row in a:
        b.append(f'en_US/m-ailabs_low#{row}')

    # LJspeech - single speaker

    b.append(f'en_US/ljspeech_low')

    # en_UK apope - only speaker

    b.append(f'en_UK/apope_low')

    all_names = b

    # ================


    



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
    import msinference
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
        y = y.detach().cpu().numpy() #.clip(0, 1)

        return y








    sentences = [
        # 'List 2',
        'The boy was there when the sun rose.',
        'A rod is used to catch pink salmon.',
        # https://www.cs.columbia.edu/~hgs/audio/harvard.html
        "Metamorphosis of cultural heritage to augmented hypermedia for accessibility and inclusion.",
        'Sweet dreams are made of this, .. !!! I travel the world and the seven seas.',
    ]


    
    VOICES = {}
    for _id, _voice in enumerate(all_names):   # [:2]
        
        str_voice = _voice.replace('/', '_').replace('#', '_').replace('_low', '')
        
        fig_file = out_dir + str_voice + '.png'

        

        

        
        

        if 'cmu-arctic' in str_voice:
            tgt_wav = wav_dir + str_voice.replace('cmu-arctic', 'cmu_arctic') + '.wav'
            affect_wav = wav_dir + str_voice.replace('cmu-arctic', 'cmu_arctic') + '_affect.wav'
        else:
            tgt_wav = wav_dir + str_voice + '.wav' # [...cmu-arctic...](....cmu_arctic....wav) # README has underscore in cmu_arctic.wav
            affect_wav = wav_dir + str_voice.replace('cmu-arctic', 'cmu_arctic') + '_affect.wav'
        


        print(_voice)
        if 1: #not os.path.isfile(fig_file):


            # -- START Mimic3

            # print('____\n', _text_, '\n__')
            # Different rate per comma-split sentence piece - CONCAT subsentences 2 GLOBAL SSML
            volume = int(40 * np.random.rand() + 54)
            rate = 4  # speed sounds nice for speaker-reference wav

              # mimic - target reference - wav
            _ssml = (
                '<speak>'
                f'<prosody volume=\'{volume}\'>'
                    f'<prosody rate=\'{rate}\'>'
                    f'<voice name=\'{_voice}\'>'
                    f'<s>Sweet dreams are made of this, .. !!! I travel the world and the seven seas.</s>'
                    '</voice>'
                    '</prosody>'
                    '</prosody>')
                
            # SINGLE CALL TTS per DESCRIPTION    
            _ssml += '</speak>'
            with open('_tmp_ssml.txt', 'w') as f:
                f.write(_ssml)

            
            ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > {tgt_wav}', shell=True)
            ps.wait()  # using ps to call mimic3 because samples dont have time to be written in stdout buffer

            # -- END Mimic3

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


                # resynthesize wav_pth via mimic3 & random speed

                



                x = msinference.inference(_text_,
                                        msinference.compute_style(tgt_wav), #'tgt_spk.wav'),
                                        alpha=0.3,
                                        beta=0.7,
                                        diffusion_steps=7,
                                        embedding_scale=1)


                # StyleTTS2
                soundfile.write(affect_wav, x, 24000)

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
            
            VOICES[_id] = {'voice': _voice,
                           'emotion': centroid_emotion.tolist(),
                           'tgt_wav': tgt_wav,
                           'affect_wav': affect_wav,
                           'fig_file': fig_file,
                           'str_voice': str_voice
            }
            
            if not os.path.isfile(fig_file):
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
            else:
                print('EXISTS')
                

        # -- END FIG EXISTS - THUS VOICE IS ALREADY WITH EMOTION INSIDE voices.json


    # save dict of voice; all_emotions_voice_k


    with open(GLOBAL_FILE, 'w') as f:
        json.dump({'voices': VOICES}, f, indent=4)
































# =======================================================================
# START             G E N E R A T E   README.md
# =======================================================================

import pandas as pd
y = pd.read_json(GLOBAL_FILE)['voices']



# == markdown table

y = sorted(y, key=lambda d: sum(d['emotion']))  # sort wav_files by emotion

# SORTING OUTPUT IS LIST - 0-th ELEMENT = LOWEST VALENCE
#_________________________________________________
# [{'voice': 'en_US/vctk_low#p236',
#   'emotion': [0.017387679778039, 0.20520514249801602, 0.323482856154441],
#   'tgt_wav': 'assets/wavs/en_US_vctk_p236.wav',
#   'affect_wav': 'assets/wavs/en_US_vctk_p236_emo.wav',
#   'fig_file': 'assets/en_US_vctk_p236.png',
#   'str_voice': 'en_US_vctk_p236'},
#  {'voice': 'en_US/vctk_low#p239',
#   'emotion': [0.008738230913877001, 0.18520271033048602, 0.34983529150485904],
#   'tgt_wav': 'assets/wavs/en_US_vctk_p239.wav',
#   'affect_wav': 'assets/wavs/en_US_vctk_p239_emo.wav',
#   'fig_file': 'assets/en_US_vctk_p239.png',
#   'str_voice': 'en_US_vctk_p239'}]


# table = (
#     f'<table><tr>'
#     f'<td>\n\n \n\n</td>'
#     f'<td>\n\n voice \n\n</td>'
#     f'<td>\n\n StyleTTS2 \n\n</td>'
#     f'<td>\n\n [Voice emotion](https://www.cs.columbia.edu/~hgs/audio/harvard.html)\n\n</td>'
#     f'<td>\n\n `arousal` \n\n</td>'
#     f'<td>\n\n `valence` \n\n</td>'
#     f'<td>\n\n `dominance` \n\n</td>')

table = (
   f'<html lang="en">\n<body>\n<h1>Available Voices</h1>'
   f'\nYou can use all Affective or Non Affective voices for TTS: \n'
   f'<a href="https://github.com/audeering/shift/blob/main/demo.py">demo.py</a> .'
   f'<hr>'
   f'<table><tr><td>'  # count
   f'</td><td>\n\n Voice \n\n</td>'
   f'<td>\n\n Non-Affective \n\n</td>'
   f'<td>\n\n Emotion Volatility \n\n</td>'
   f'<td>\n\n Affective \n\n</td>'
)

for i, tup in enumerate(reversed(y)):

    _voice, emotion, tgt_wav, affect_wav, fig_file, str_voice = tup.values()
    print('\n\n', _voice, '\n\n')
    # append row in MarkDown table

    # row = (
    #     f'<tr>\n <td>\n {i} \n</td>\n<td>\n\n ```\n{_voice}\n``` \n\n</td>'
    #     f'<td>\n\n ![{wav_path}]({wav_path}) \n\n</td>'
    #     f'<td>\n\n <img src=\"{fig_path}\" alt="sounds" width="101" height="101"> \n\n</td>'
    #     f'<td>\n\n ```\n{arousal:.3f}\n``` \n\n</td>'
    #     f'<td>\n\n ```\n{valence:.3f}\n``` \n\n</td>'
    #     f'<td>\n\n ```\n{dominance:.3f}\n``` \n\n</td>'
    #     f'</tr>\n\n\n\n')


    row = (
        f'<tr>\n <td>\n {i} \n</td>\n<td>\n\n \n{_voice}\n \n\n</td>'
        f'<td>\n\n<audio preload="none" controls src="{tgt_wav}"></audio>\n\n'
        f'</td><td>\n\n <img src=\"{fig_file}\" alt="sounds" width="420" height="420"> \n\n</td><td>'
        f'<audio preload="none" controls src="{affect_wav}"></audio></td></tr>\n'
    )    

    # print(row)

    table += row

table += '</table>\n</body>\n</html>'

with open('index.html', 'w') as f:
    f.write(table)
