import os
from training import Classifier
from splitter import split
import time
import cv2


def main():
	start = time.time()

	classifier = Classifier()
	classifier.train()

	print "Training finished at %ds\n" % (time.time() - start)

	i = 1
	for fn in os.listdir('./data/matthew/'):
		with open('./data/matthew/out/%d.txt' % i, 'w') as fw:
			i += 1
			print './data/matthew/%s' % fn
			chars = split('./data/matthew/%s' % fn)
			for _, line in chars.iteritems():
				for char in line:
					fw.write(classifier.classify(char))
					fw.write(',')
				fw.write('\n')
		print "Finished Matthew %d at %ds\n" % (i, (time.time() - start))


if __name__ == "__main__":
  main()
