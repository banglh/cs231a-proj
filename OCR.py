import os
from training import Classifier
from splitter import split



def main():
	classifier = Classifier()
	classifier.train()

	i = 1
	for fn in os.listdir('./data/matthew/'):
		with open('./data/matthew/out/%d.txt' % i, 'w') as fw:
			i += 1
			chars = split('./data/matthew/%s' % fn)
			for line in chars:
				for char in line:
					fw.write(classifier.classify(char))
				fw.write('\n')


if __name__ == "__main__":
  main()
