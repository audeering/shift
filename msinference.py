import torch
from cached_path import cached_path
import nltk
import audresample
# nltk.download('punkt')
import numpy as np
np.random.seed(0)
import time
import yaml
import torch.nn.functional as F
import copy
import torchaudio
import librosa
from models import *
from munch import Munch
from torch import nn
from nltk.tokenize import word_tokenize

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print('CLEAN', text)
        return indexes



textclenaer = TextCleaner()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

# START UTIL



def alpha_num(f):
    f = re.sub(' +', ' ', f)              # delete spaces
    f = re.sub(r'[^A-Z a-z0-9 ]+', '', f)  # del non alpha num
    return f



def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    

    
# ======== UTILS ABOVE    

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    # print("MPS would be available but cannot be used rn")
    pass
    # device = 'mps'

import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
# phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))


config = yaml.safe_load(open(str('Utils/config.yml')))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params_whole = torch.load(str(cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth")), map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def inference(text, 
              ref_s, 
              alpha = 0.3, 
              beta = 0.7, 
              diffusion_steps=5, 
              embedding_scale=1, 
              use_gruut=False):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    # print(f'PHONEMIZER: {ps=}\n\n') #PHONEMIZER: ps=['ɐbˈɛbæbləm ']
    ps = word_tokenize(ps[0])
    # print(f'TOKENIZER: {ps=}\n\n') #OKENIZER: ps=['ɐbˈɛbæbləm']
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    # print(f'TEXTCLEAN: {ps=}\n\n') #TEXTCLEAN: ps='ɐbˈɛbæbləm'
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    # print(f'TOKENSFINAL: {ps=}\n\n')

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        # -----------------------
        # WHO TRANSLATES these tokens to sylla
        # print(text_mask.shape, '\n__\n', tokens, '\n__\n',  text_mask.min(), text_mask.max())
        # text_mask=is binary
        # tokes =  tensor([[  0,  55, 157,  86, 125,  83,  55, 156,  57, 158, 123,  48,  83,  61,
                        #  157, 102,  61,  16, 138,  64,  16,  53, 156, 138,  54,  62, 131,  85,
                        #  123,  83,  54,  16,  50, 156,  86, 123, 102, 125, 102,  46, 147,  16,
                        #   62, 135,  16,  76, 158,  92,  55, 156,  86,  56,  62, 177,  46,  16,
                        #   50, 157,  43, 102,  58,  85,  55, 156,  51, 158,  46,  51, 158,  83,
                        #   16,  48,  76, 158, 123,  16,  72,  53,  61, 157,  86,  61,  83,  44,
                        #  156, 102,  54, 177, 125,  51,  16,  72,  56,  46,  16, 102, 112,  53,
                        #   54, 156,  63, 158, 147,  83,  56,  16,   4]], device='cuda:0') 


        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        # print('BERTdu', bert_dur.shape, tokens.shape, '\n') # bert what is the 768 per token -> IS USED in sampler
        # BERTdu torch.Size([1, 11, 768]) torch.Size([1, 11])

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)
     

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        x = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    x = x.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model
    
    x /= np.abs(x).max() + 1e-7
    
    return x




# ___________________________________________________________

# https://huggingface.co/spaces/mms-meta/MMS/blob/main/tts.py
# ___________________________________________________________

# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import tempfile
import torch
import sys
import numpy as np
import audiofile
from huggingface_hub import hf_hub_download

# Setup TTS env
if "vits" not in sys.path:
    sys.path.append("Modules/vits")

from Modules.vits import commons, utils
from Modules.vits.models import SynthesizerTrn

TTS_LANGUAGES = {}
# with open('_d.csv', 'w') as f2:
with open(f"Utils/all_langs.csv") as f:
    for line in f:
        iso, name = line.split(",", 1)
        TTS_LANGUAGES[iso.strip()] = name.strip()
        # f2.write(iso + ',' + name.replace("a S","")+'\n')
        
        
        
# LOAD hun / ron / serbian - rmc-script_latin / cyrillic-Carpathian (not Vlax)




def has_cyrillic(text):
    # https://stackoverflow.com/questions/48255244/python-check-if-a-string-contains-cyrillic-characters
    return bool(re.search('[\u0400-\u04FF]', text))

class TextForeign(object):
    def __init__(self, vocab_file):
        self.symbols = [
            x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()
        ]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text, lang=None):
        text = self.preprocess_char(text, lang=lang)
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        return txt_filt

    def preprocess_char(self, text, lang=None):
        """
        Special treatement of characters in certain languages
        """
        if lang == "ron":
            text = text.replace("ț", "ţ")
            print(f"{lang} (ț -> ţ): {text}")
        return text


def foreign(text=None, lang='romanian', speed=None):
    # TTS for non english languages supported by 
    # https://huggingface.co/spaces/mms-meta/MMS
    
    if 'hun' in lang.lower():
        
        lang_code = 'hun'
        
    elif 'ser' in lang.lower():
        
        if has_cyrillic(text):
            
            lang_code = 'rmc-script_cyrillic'   # romani carpathian (has also Vlax)
        
        else:
            
            lang_code = 'rmc-script_latin'   # romani carpathian (has also Vlax)
        
    elif 'rom' in lang.lower():
        
        lang_code = 'ron'
        speed = 1.24 if speed is None else speed
        
    else:
        lang_code = lang.split()[0].strip()
    # Decoded Language
    print(f'\n\nLANG {lang_code=}\n_____________________\n')
    vocab_file = hf_hub_download(
        repo_id="facebook/mms-tts",
        filename="vocab.txt",
        subfolder=f"models/{lang_code}",
    )
    config_file = hf_hub_download(
        repo_id="facebook/mms-tts",
        filename="config.json",
        subfolder=f"models/{lang_code}",
    )
    g_pth = hf_hub_download(
        repo_id="facebook/mms-tts",
        filename="G_100000.pth",
        subfolder=f"models/{lang_code}",
    )
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextForeign(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(g_pth, net_g, None)
    
    # TTS via MMS

    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"

    if is_uroman:
        uroman_dir = "Utils/uroman"
        assert os.path.exists(uroman_dir)
        uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
        text = text_mapper.uromanize(text, uroman_pl)

    text = text.lower()
    text = text_mapper.filter_oov(text, lang=lang)
    stn_tst = text_mapper.get_text(text, hps)
    with torch.no_grad():
        print(f'{speed=}\n\n\n\n_______________________________')
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        x = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0 / speed)[0][0, 0].cpu().float().numpy()
            )
    x /= np.abs(x).max() + 1e-7

    # hyp = (hyp * 32768).astype(np.int16)
    # x =  hyp  #, text
    print(x.shape, x.min(), x.max(), hps.data.sampling_rate)  # (hps.data.sampling_rate, 
    
    x = audresample.resample(signal=x.astype(np.float32),
                             original_rate=16000,
                             target_rate=24000)[0, :]  # reshapes (64,) -> (1,64)
    return x




# LANG = 'eng'
# _t = 'Converts a string of text to a sequence of IDs corresponding to the symbols in the text. Args: text: string to convert to a sequence'

# x = synthesize(text=_t, lang=LANG, speed=1.14)
# audiofile.write('_r.wav', x, 16000)  # mms-tts = 16,000
