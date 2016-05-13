from features import PDC_features
from sklearn.naive_bayes import GaussianNB
import cv2


training_data = []
targets = []
fonts = ['Gothic', 'Lantinghei', 'Meiryo', 'Mincho', 'Osaka']
for font in fonts:
	for i in range(2136):
		im = cv2.imread('data/kanji-%s/kanji_%d' % (font, i + 1) )
		feats = PDC_features(im)
		training_data.append(feats)
		targets.append(i+1)

gnb = GaussianNB()
gnb.fit(training_data, targets)

for i in range(2136):
	im = cv2.imread('data/kanji-STFangSong/kanji_%d' % (i+1))
	feats = PDC_features(im)
	pred = gnb.predict(feats)
	if pred != i + 1:
		print "%d predicted instead of %d" % (pred, i)

