#!/usr/bin/env python
# -*- coding: utf-8 -*-

from features import PDC_features
from sklearn.naive_bayes import GaussianNB
import cv2
import unicodecsv as csv
import time
import numpy as np
import codecs

NUM_KANJI = 2289


class Classifier:
	fonts = ['Gothic']#, 'Lantinghei', 'Meiryo', 'Mincho', 'Osaka', 'STFangSong', 'GenEiExtraLight', 'GenEiHeavy', 'GenEiSemiBold', 
	#'HonyaJi', 'Mamelon', 'MPlusBold', 'MPlusRegular', 'MPlusThin', 'WawaSC', 'WeibeiSC']

	def __init__(self):
		self.kanjiFile = "kanji_list.txt"

		self.training_data = []
		self.targets = []
		self.gnb = GaussianNB()

		with open(self.kanjiFile, 'rb') as f:
		    reader = csv.reader(f, encoding='utf-8')
		    self.kanjiList = list(reader)

	def train(self):
		for font in self.fonts:
			for i in range(NUM_KANJI):
				im = cv2.imread('data/kanji-%s/kanji_%d.png' % (font, i + 1), cv2.IMREAD_GRAYSCALE)
				feats = PDC_features(im)
				self.training_data.append(feats)
				self.targets.append(self.kanjiList[i])

		self.training_data = np.array(self.training_data)
		self.targets = np.array(self.targets).ravel()

		self.gnb.fit(self.training_data, self.targets)

	def classify(self, im):
		feats = PDC_features(im, True)
		results =  self.gnb.predict(feats)
		return results

	def getKanji(self, i):
		return self.kanjiList[i]

def main():
	classifier = Classifier()
	classifier.train()
	print "done training"

	numRight = 0
	for i in range(NUM_KANJI):
		im = cv2.imread('data/kanji-Gothic/kanji_%d.png' % (i + 1), cv2.IMREAD_GRAYSCALE)
		result = classifier.classify(im)
		if result == classifier.getKanji(i):
			numRight += 1

	print "Got %d/%d right" % (numRight, NUM_KANJI)

if __name__ == "__main__":
	main()

