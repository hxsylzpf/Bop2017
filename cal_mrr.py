import io

def cal_mrr(fname):
	labels = []
	label_example = []
	with open(fname) as f:
		while 1:
			line = f.readline()
			if not line:
				break
			if line == '\n':
				labels.append(label_example)
				label_example = []
				continue
			label = line.split('\t')[0]
			
			label_example.append(label)
	print(len(labels))
	print(labels[0])
	print(labels[2])
	print(labels[len(labels) - 1])
	return labels

def cal_mrr_2(labels):
	mrr = 0
	sum = 0
	for example in labels:
		sum += len(example)
		if '1' not in example:
			idx = 0
		else:
			idx = example.index('1')
			idx += 1
			idx = 1/idx
		mrr += idx
	mrr = mrr/len(labels)
	print(mrr)
	print(sum)
if __name__ == '__main__':
	labels = cal_mrr('./data/result_score_3')
	cal_mrr_2(labels)