import os
import io
import pickle
import numpy as np

def main(fname_r, fname_w):
	vocab_embed = {}
	embedding_dict = open(fname_r, 'rb')
		# for i in range(len(embedding_dict)):
		# vec = ' '.join([str(value) for value in embedding_dict[i]])
		# embedding_f.write(vec + '\n')
	for line in	embedding_dict:
		line = line.strip().decode('utf-8').split(' ')
		word = line[0]
		embedding = [float(value) for value in line[1:]]
		vocab_embed[word] = embedding
	vocab_embed_f = open(fname_w, 'wb')
	print(len(vocab_embed))
	pickle.dump(vocab_embed, vocab_embed_f)
	return vocab_embed
def get_embed_for_train_data(f_name, vocab_embed):
	num = 0
	not_in_words = []
	all_words = []
	with open(f_name, 'r') as f_r:
		for line in f_r:
			words = line.strip().split(' ')
			for word in words:
				if word not in all_words:
					all_words.append(word)
				if word not in vocab_embed:
					# print(word)
					if word not in not_in_words:
						not_in_words.append(word)
					
		print(len(all_words))
		print(len(not_in_words))


if __name__ == '__main__':
	vocab_embed = main('./data/newsblogbbs.vec', './data/vocab_embed.pickle')
	# get_embed_for_train_data('./data/a_cut', vocab_embed)