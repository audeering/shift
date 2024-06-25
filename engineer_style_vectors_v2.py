
from pathlib import Path
import shutil
import csv
import io
import os
import typing
import wave
import sys
from mimic3_tts.__main__ import (CommandLineInterfaceState,
                                 get_args,
                                 initialize_args,
                                 initialize_tts,
                                 # print_voices,
                                 # process_lines,
                                 shutdown_tts,
                                 OutputNaming,
                                 process_line)


def process_lines(state: CommandLineInterfaceState, wav_path=None):
    '''MIMIC3 INTERNAL CALL that yields the sigh sound'''

    args = state.args

    result_idx = 0
    print(f'why waitings in the for loop LIN {state.texts=}\n')
    for line in state.texts:
        print(f'LIN {line=}\n')  # prints \n so is empty not getting the predifne text of state.texts
        line_voice: typing.Optional[str] = None
        line_id = ""
        line = line.strip()
        # if not line:
        #     continue

        if args.output_naming == OutputNaming.ID:
            # Line has the format id|text instead of just text
            with io.StringIO(line) as line_io:
                reader = csv.reader(line_io, delimiter=args.csv_delimiter)
                row = next(reader)
                line_id, line = row[0], row[-1]
                if args.csv_voice:
                    line_voice = row[1]

        process_line(line, state, line_id=line_id, line_voice=line_voice)
        result_idx += 1

    print('\nARRive at All Audio writing\n\n\n\n')
    # -------------------------------------------------------------------------

    # Write combined audio to stdout
    if state.all_audio:
        # _LOGGER.debug("Writing WAV audio to stdout")

        if sys.stdout.isatty() and (not state.args.stdout):
            with io.BytesIO() as wav_io:
                wav_file_play: wave.Wave_write = wave.open(wav_io, "wb")
                with wav_file_play:
                    wav_file_play.setframerate(state.sample_rate_hz)
                    wav_file_play.setsampwidth(state.sample_width_bytes)
                    wav_file_play.setnchannels(state.num_channels)
                    wav_file_play.writeframes(state.all_audio)

                    # play_wav_bytes(state.args, wav_io.getvalue())
                # wav_path = '_direct_call_2.wav'
                with open(wav_path, 'wb') as wav_file:
                    wav_file.write(wav_io.getvalue())
                    wav_file.seek(0)

# -----------------------------------------------------------------------------
# cat _tmp_ssml.txt | mimic3 --cuda --ssml --noise-w 0.90001 --length-scale 0.91 --noise-scale 0.04 > noise_w=0.90_en_happy_2.wav
# ======================================================================
out_dir = 'assets/'
reference_wav_directory = 'assets/wavs/style_vectors_v2/'
Path(reference_wav_directory).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)

wav_dir = 'assets/wavs/'
Path(wav_dir).mkdir(parents=True, exist_ok=True)
N_PIX = 11


# =======================================================================
# S T A R T                 G E N E R A T E   png/wav
# =======================================================================

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


a = [
    'awb'  # comma
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
    'gka',
    ]


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


VOICES = {}
for _id, _voice in enumerate(all_names):

    # If GitHub Quota exceded copy mimic-voices from local copies
    #
    # https://github.com/MycroftAI/mimic3-voices
    #
    home_voice_dir = f'/home/audeering.local/dkounadis/.local/share/mycroft/mimic3/voices/{_voice.split("#")[0]}/'
    Path(home_voice_dir).mkdir(parents=True, exist_ok=True)
    speaker_free_voice_name = _voice.split("#")[0] if '#' in _voice else _voice
    if not os.path.isfile(home_voice_dir + 'generator.onnx'):
        shutil.copyfile(
            f'/data/dkounadis/cache/mimic3-voices/voices/{speaker_free_voice_name}/generator.onnx',
            home_voice_dir + 'generator.onnx')  # 'en_US incl. voice

    prepare_file = _voice.replace('/', '_').replace('#', '_').replace('_low', '')
    if 'cmu-arctic' in prepare_file:
        prepare_file = prepare_file.replace('cmu-arctic', 'cmu_arctic') + '.wav'
    else:
        prepare_file = prepare_file + '.wav' # [...cmu-arctic...](....cmu_arctic....wav) 

    file_true = prepare_file.split('.wav')[0] + '_true_.wav'
    file_false = prepare_file.split('.wav')[0] + '_false_.wav'
    print(prepare_file, file_false, file_true)


    reference_wav = reference_wav_directory + prepare_file
    rate = 4  # high speed sounds nice if used as speaker-reference audio for StyleTTS2
    _ssml = (
        '<speak>'
        '<prosody volume=\'74\'>'
        f'<prosody rate=\'{rate}\'>'
        f'<voice name=\'{_voice}\'>'
        '<s>'
        'Sweet dreams are made of this, .. !!! # I travel the world and the seven seas.'
        '</s>'
        '</voice>'
        '</prosody>'
        '</prosody>'
        '</speak>'
    )
    with open('_tmp_ssml.txt', 'w') as f:
        f.write(_ssml)


    # ps = subprocess.Popen(f'cat _tmp_ssml.txt | mimic3 --ssml > {reference_wav}', shell=True)
    # ps.wait()  # using ps to call mimic3 because samples dont have time to be written in stdout buffer
    args = get_args()
    args.ssml = True
    args.text = [_ssml]  #['aa', 'bb'] #txt
    args.interactive = False
    # args.output_naming = OutputNaming.TIME

    state = CommandLineInterfaceState(args=args)
    initialize_args(state)
    initialize_tts(state)
    # args.texts = [txt] #['aa', 'bb'] #txt
    # state.stdout = '.' #None #'makeme.wav'
    # state.output_dir = '.noopy'
    # state.interactive = False
    # state.output_naming = OutputNaming.TIME
    # # state.ssml = 1234546575
    # state.stdout = True
    # state.tts = True
    process_lines(state, wav_path=reference_wav)
    shutdown_tts(state)