from features import PDC_features
from sklearn.naive_bayes import GaussianNB
import cv2
import csv



class Classifier:
	fonts = ['Gothic', 'Lantinghei', 'Meiryo', 'Mincho', 'Osaka', 'STFangSong', 'GenEiExtraLight', 'GenEiHeavy', 'GenEiSemiBold',
			'HonyaJi', 'Mamelon', 'MPlusBold', 'MPlusRegular', 'MPlusThin', 'WawaSC', 'WeibeiSC', 'Yuanti']

	kanjiFile = "data/kanji_list.csv"

	def __init__(self):
		self.training_data = []
		self.targets = []
		self.gnb = GaussianNB()

		with open(kanjiFile, 'rb') as f:
		    reader = csv.reader(f)
		    self.kanjiList = list(reader)

	def train(self):
		for font in self.fonts:
			for i in range(2136):
				im = cv2.imread('data/kanji-%s/kanji_%d' % (font, i + 1) )
				feats = PDC_features(im)
				self.training_data.append(feats)
				self.targets.append(self.kanjiList[i])

		self.gnb.fit(training_data, targets)

	def classify(self, im):
		feats = PDC_features(im)
		return self.gnb.predict(feats)

