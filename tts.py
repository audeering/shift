# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import re
import requests
from pathlib import Path
Path('out/').mkdir(parents=True, exist_ok=True)

# SSH AGENT
#   eval $(ssh-agent -s)
#   ssh-add ~/.ssh/id_ed25519_github2024
#
#   git remote set-url origin git@github.com:audeering/shift
# ==


def alpha_num(f):
    f = re.sub(' +', ' ', f)              # delete spaces
    f = re.sub(r'[^A-Za-z0-9 ]+', '', f)  # del non alpha num
    return f


def command_line_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--affective',
        help="Select Emotional or non-emotional variant of Available voices: https://audeering.github.io/shift/",
        action='store_false',
    )
    parser.add_argument(
        '--device',
        help="Device ID",
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--text',
        help="Text to be synthesized.",
        default='sample.txt',
        type=str,
    )
    parser.add_argument(
        '--native',
        help="""
        --native: (without argument) a flag to do voice cloning using the speech from --video,
        --native my_voice.wav:  Voice cloning from user provided audio""",
        # nargs='?',
        # const=None,
        # default=False   # default has to be none
        )
    parser.add_argument(
        '--voice',
        help="TTS voice - Available voices: https://audeering.github.io/shift/",
        default="en_US/m-ailabs_low#judy_bieber", #'en_US/cmu-arctic_low#lnh',
        type=str,
    )
    parser.add_argument(
        '--image',
        help="If provided is set as background for output video, see --text",
        type=str,
    )
    parser.add_argument(
        '--video',
        help="Video file for video translation. Voice cloned from the video",
        type=str,
    )
    parser.add_argument(
        '--out_file',
        help="Output file name.",
        type=str,
        default=None
    )
    parser.add_argument(
        '--speed',
        help='speec of TTS (only used in Non English voices).',
        type=str,
        default=1.24,
    )
    return parser

def send_to_server(args):
    url = "http://192.168.88.209:5000"

    payload = {
        'affective': args.affective,
        'voice': args.voice,
        'native': args.native,
        'text': args.text,
        'image': args.image,
        'video': args.video,
        'speed': args.speed,
        # 'out_file': args.out_file   # let serve save as temp
    }

    # In data= we can write args

    # In files=  sent actual files if provided
    text_file = open(args.text, 'rb')

    image_file, video_file, native_file = None, None, None
    if args.image is not None:
        print('\nLOADING IMAGE\n')
        try:
            image_file = open(args.image, 'rb')
        except FileNotFoundError:
            pass
            

    if args.video is not None:
        print('\nLOADING vid\n')
        try:
            video_file = open(args.video, 'rb')
        except FileNotFoundError:
            pass

    if args.native is not None:
        print('\nLOADING natv\n')
        try:
            native_file = open(args.native, 'rb')
        except FileNotFoundError:
            pass
        
        
        
        # --------------------- send this extra 
            
    print('Sending...\n') 

    response = requests.post(url, data=payload, 
                             files=[(args.text, text_file),
                                    (args.image, image_file),
                                    (args.video, video_file),
                                    (args.native, native_file)])  # NONEs do not arrive to servers dict

    # Check the response from the server
    if response.status_code == 200:
        print("\nRequest was successful!")
        # print("Response:", respdonse.__dict__.keys(), '\n=====\n')

    else:
        print("Failed to send the request")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    return response


def cli():
    parser = command_line_args()
    args = parser.parse_args()
    
    if args.out_file is None:
        vid = alpha_num(args.video) if args.video else f'{np.random.rand()*1e7}'[:6]
        args.out_file = alpha_num(args.text) + '_' + alpha_num(args.voice) + '_' + vid
    response = send_to_server(args)
    
    with open(
        # args.out_file is not send to server - server writes tmp - copied by client
        './out/' + args.out_file + '.' + response.headers['suffix-file-type'].split('.')[-1],
        'wb'
        ) as f:
        f.write(response.content)
    print('REsponse AT client []\n----------------------------', response.headers)


if __name__ == '__main__':
    cli()

# assume also video and text for video we have to write some classes for video for audiocraft
# then call tts.py on this video with nonempty labels - thus calls audiocraft