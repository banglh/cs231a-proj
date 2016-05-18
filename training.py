from features import PDC_features
from sklearn.naive_bayes import GaussianNB
import cv2
import csv
import time
import numpy as np


class Classifier:
	fonts = ['Gothic']
	#, 'Lantinghei', 'Meiryo', 'Mincho', 'Osaka', 'STFangSong', 'GenEiExtraLight', 'GenEiHeavy', 'GenEiSemiBold',
	#		'HonyaJi', 'Mamelon', 'MPlusBold', 'MPlusRegular', 'MPlusThin', 'WawaSC', 'WeibeiSC', 'Yuanti']

	def __init__(self):
		self.kanjiFile = "data/kanjiOnly.csv"

		self.training_data = []
		self.targets = []
		self.gnb = GaussianNB()

		with open(self.kanjiFile, 'rb') as f:
		    reader = csv.reader(f)
		    self.kanjiList = list(reader)

	def train(self):
		for font in self.fonts:
			for i in range(2136):
				im = cv2.imread('data/kanji-%s/kanji_%d.png' % (font, i + 1), cv2.IMREAD_GRAYSCALE)
				feats = PDC_features(im)
				self.training_data.append(feats)
				self.targets.append(i)

		self.training_data = np.array(self.training_data)
		self.targets = np.array(self.targets).ravel()

		self.gnb.fit(self.training_data, self.targets)

	def classify(self, im):
		feats = PDC_features(im, True).reshape(-1, 1)
		results =  self.gnb.predict(feats)
		return results

