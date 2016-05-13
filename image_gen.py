import subprocess

KANJI = "kanji_only.txt"
FONT = "Gulim"

with open(KANJI, 'r') as f:
  idx = 0
  for l in f:
    text = l.strip()
    if not text or len(text) == 0: continue
    idx += 1
    cmd ="""
      convert -font %s -size 256x -background white -fill black label:"%s" data/kanji-%s/kanji_%s.png
    """ % (FONT, text, FONT, idx)
    print cmd 
    subprocess.call(cmd, shell=True)
