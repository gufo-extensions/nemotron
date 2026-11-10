#!/bin/sh
# Writes the images of the image benchmarks and parity checks into DIR:
# a 512x512 photo and a 1024x1024 logo from a Gufo checkout, and rendered
# screens at 1024x1024, 1920x1080 and 3840x2160 (ffmpeg with drawtext).
#
#   tools/bench/vision_images.sh GUFO_CHECKOUT DIR
set -eu
gufo=$1
out=$2
mkdir -p "$out"
cp "$gufo/docs/internal/models/qwen-image-2.1/artifacts/turbo-editing-input.png" \
  "$out/photo-512.png"
ffmpeg -v error -y -i "$gufo/assets/gufo-logo.jpg" -vf scale=1024:1024 \
  "$out/logo-1024.png"
ffmpeg -v error -y -f lavfi -i "color=c=white:s=1024x1024" -vf "\
drawbox=x=0:y=0:w=1024:h=56:color=0x2b2b2b:t=fill,\
drawtext=text='Settings':fontsize=30:fontcolor=white:x=20:y=12,\
drawtext=text='Username\: alice_42':fontsize=34:fontcolor=black:x=60:y=160,\
drawtext=text='Email\: alice@example.org':fontsize=34:fontcolor=black:x=60:y=230,\
drawbox=x=60:y=330:w=300:h=70:color=0xd93025:t=fill,\
drawtext=text='Delete account':fontsize=28:fontcolor=white:x=85:y=350" \
  -frames:v 1 "$out/settings-1024.png"
ffmpeg -v error -y -f lavfi -i "color=c=white:s=1920x1080" -vf "\
drawbox=x=0:y=0:w=1920:h=60:color=0x2b2b2b:t=fill,\
drawtext=text='File  Edit  View  Help':fontsize=28:fontcolor=white:x=20:y=15,\
drawtext=text='Nemotron vision test - the quick brown fox jumps over the lazy dog 0123456789':fontsize=36:fontcolor=black:x=60:y=200,\
drawbox=x=60:y=300:w=400:h=80:color=0x1f6feb:t=fill,\
drawtext=text='Submit':fontsize=32:fontcolor=white:x=200:y=322" \
  -frames:v 1 "$out/screenshot.png"
ffmpeg -v error -y -f lavfi -i "color=c=white:s=3840x2160" -vf "\
drawbox=x=0:y=0:w=3840:h=110:color=0x2b2b2b:t=fill,\
drawtext=text='File  Edit  View  Window  Help':fontsize=56:fontcolor=white:x=40:y=28,\
drawtext=text='Quarterly report - revenue 4.2M, costs 3.1M, margin 26 percent':fontsize=72:fontcolor=black:x=120:y=400,\
drawbox=x=120:y=600:w=800:h=160:color=0x1f6feb:t=fill,\
drawtext=text='Export PDF':fontsize=64:fontcolor=white:x=330:y=645" \
  -frames:v 1 "$out/screenshot-4k.png"
