# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import requests
# SSH AGENT
#   eval $(ssh-agent -s)
#   ssh-add ~/.ssh/id_ed25519_github2024
#
#   git remote set-url origin git@github.com:audeering/shift
# ==





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
        default='out'
    )
    return parser

def send_to_server(args):
    url = "http://127.0.0.1:5000/api/v1/static"

    payload = {
        'affective': args.affective,
        'voice': args.voice,
        'native': args.native
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

            
    print('Sending...\n') 

    response = requests.post(url, data=payload, files=[(args.text, text_file),
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
    # main(args)  # --> to be run on server  -> server sends back video or audio -> saved as file by cli()
    send_to_server(args)


if __name__ == '__main__':
    cli()
