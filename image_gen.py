# -*- coding: utf-8 -*-

import subprocess
import os

KANJI = "kanji_only.txt"
FONT_FULL="'/Users/Sloane/Downloads/mplus-TESTFLIGHT-061/mplus-2p-regular.ttf'"
FONT_SHORT = "MPlusRegular"

directory = "data/kanji-%s" % FONT_SHORT

if not os.path.exists(directory):
  os.makedirs(directory)

with open(KANJI, 'r') as f:
  idx = 0
  for l in f:
    text = l.strip()
    if not text or len(text) == 0: continue
    idx += 1
    cmd ="""
      convert -font %s -size 256x -background white -fill black label:"%s" data/kanji-%s/kanji_%s.png
    """ % (FONT_FULL, text, FONT_SHORT, idx)
    print cmd 
    subprocess.call(cmd, shell=True)
