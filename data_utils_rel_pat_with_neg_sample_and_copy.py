#!/usr/bin/python
# -*- coding:utf-8 -*-
import nltk
import os
import re
import string
# import cPickle
import pickle
import codecs
import time
from nltk import word_tokenize

# import random
# import jieba
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
#引入特殊字符
_pad = "_pad"
_unk = "_unk"
_go = "_go"
_end = "_end"

_start_vocab = [_pad, _unk, _go, _end]

pad_id = 0
unk_id = 1
go_id = 2
end_id = 3

global all_rel_list
global train_ques_mention_dict


def basic_tokenizer(sentence):
	return sentence.strip().split()
def tab_tokenizer(sentence):
	return sentence.strip().split("\t")
def standard_tokenizer(sentence):
	return filter(lambda x: not re.match('\s',x) and x != '', re.split('([\s|%s])' % re.escape(string.punctuation), sentence))

def create_vocabulary_and_embed_dict(vocabulary_path, data_path, embedding_dict_path, pre_embed):
	print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
	vocab = {}
	num = 0
	embedding_dict = {}
	with open(data_path, "r") as f:
		counter = 0
		for line in f:
			tokens = basic_tokenizer(line)
			for word in tokens:
				word = word.decode('utf-8')
				if word not in _start_vocab:
					if word in vocab:
						vocab[word] += 1
					else:
						vocab[word] = 1
		vocab_list = _start_vocab + sorted(vocab, key=vocab.get, reverse=True)
		vocab_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
		for word in vocab_dict:
			embedding = [random.uniform(-0.25, 0.25) for i in range(200)]
			
			if word not in pre_embed:
				embedding_dict[vocab_dict[word]] = embedding
				num += 1
			else:
				embedding_dict[vocab_dict[word]] = pre_embed[word]
		# if len(vocab_list) > max_vocabulary_size:
		# 	vocab_list = vocab_list[:max_vocabulary_size]
		with open(vocabulary_path, "w") as vocab_file:
			for w in vocab_list:
				vocab_file.write(w + "\n")
		print(len(vocab_list))
		print(num)
		pickle.dump(embedding_dict, open(embedding_dict_path, 'wb'))
		# return vocab_dict, embedding_dict

def create_rel_vocab(rel_list, rel_vocab_path, max_vocabulary_size):
	print("creating relation vocabulary %s" % (rel_vocab_path))
	rel_vocab_file = open(rel_vocab_path, 'w')
	rel_vocab = {}
	for rel in rel_list:
		rel_vocab_file.write(rel + "\n")

def initialize_vocabulary(vocabulary_path):
	if os.path.exists(vocabulary_path):
		rev_vocab = []
		with open(vocabulary_path, "r") as f:
			rev_vocab.extend(f.readlines())
		rev_vocab = [line.strip() for line in rev_vocab]
		vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_tok_ids(sentence, vocabulary):
  words = basic_tokenizer(sentence)
  return [vocabulary.get(w, unk_id) for w in words]

def data_to_tok_ids(data_path, target_path, vocabulary_path):
	print("Tokenizing data in %s" % data_path)
	vocab, _ = initialize_vocabulary(vocabulary_path)
	# vocab = pickle.load(open(vocab_dict, 'rb'))
	# for word in vocab:
	# 	f_vocab.write(word + '\t' + str(vocab[word]) + '\n')
	# 	rev_vocab[vocab[word]] = word
	# print(rev_vocab[1])
	with open(data_path, "r") as data_file:
		with open(target_path, "w") as tokens_file:
			counter = 0
			for line in data_file:
				counter += 1
				if counter % 100000 == 0:
					print("  tokenizing line %d" % counter)
				tokdst_ids = sentence_to_tok_ids(line, vocab)
				tokens_file.write(" ".join([str(tok) for tok in tokdst_ids]) + "\n")

def relation_to_tok_ids(data_path, target_path, rel_vocab_path):
	with open(data_path, 'r') as f_a:
		rel_vocab, _ = initialize_vocabulary(rel_vocab_path)
		with open(target_path, 'w') as f_rel:
			for rel in f_a:
				if rel == '\n':
					f_rel.write('\n')
				else:
					f_rel.write(str(rel_vocab.get(rel.strip(), unk_rel_id)) + "\n")

def format_qa(qa_path, mid2name_path, q_path, a_path, train_ques_mention_dict):
	# global train_ques_mention_dict
	with open(mid2name_path, "rb") as dic_f:
		dic = pickle.load(dic_f)
	# with codecs.open(qa_path, 'r', encoding='utf-8', errors='ignore') as qa_f:
	with open(qa_path, 'r') as qa_f:
		with open(q_path, "w") as q_f:
			with open(a_path, "w") as a_f:
				for line in qa_f:
					tokens = tab_tokenizer(line)
					if tokens[0] in train_ques_mention_dict:
						tokens[0] = tokens[0].replace(train_ques_mention_dict[tokens[0]], '<E>')
						q = ' '.join(standard_tokenizer(tokens[0].lower()))
						# q = q.replace('< e >', '<e>')
						smid = tokens[1].replace("www.freebase.com","").lstrip("/").replace("/",".")
						sub = "_bos " + (smid in dic and dic[smid].strip("\"").lower() or "") + " _eos"
						rel = "_bor " + tokens[2].lower().replace("www.freebase.com","").strip("/").replace(".", " ").replace("_", " ") + " _eor"
						omid = tokens[3].replace("www.freebase.com","").lstrip("/").replace("/",".")
						obj = "_boo " + (omid in dic and dic[omid].lower().strip("\"") or "") + " _eoo"
						a = rel
						q_f.write(q + "\n")
						a_f.write(a + "\n")

def get_train_ques_mention_dict(filename):
	train_ques_mention_dict = {}
	with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
		for line in f:
			mention = ''
			ques = line.strip().split('\t')[0]
			ques_tokens = ques.split(' ')
			ment_indice = line.strip().split('\t')[1].split(' ')
			for id in ment_indice:
				id = int(id)
				mention += ques_tokens[id] + ' '
			train_ques_mention_dict[ques] = mention.strip()
	return train_ques_mention_dict

def find_target_in_source(src_path, tar_path, tar_in_src_path):
	with open(src_path, "rb") as src_f:
		with open(tar_path, "rb") as tar_f:
			tar_in_srcs = []
			for tar_line in tar_f:
				src_line = src_f.readline().decode("utf-8")
				tar_line = tar_line.decode("utf-8")
				tars = tar_line.split()
				

				srcs = src_line.split()
				
				tar_in_src = []
				for tar in tars:
					tar_in_src_one = []
					for src in srcs:
						if tar == src:
							tar_in_src_one.append(1)
						else:
							tar_in_src_one.append(0)
					
					tar_in_src.append(tar_in_src_one)
				tar_in_srcs.append(tar_in_src)
	with open(tar_in_src_path, "wb") as tar_in_src_f:
		pickle.dump(tar_in_srcs, tar_in_src_f)

def reload_old(q_path, a_path, q_in_a_path, q_s_prob_path, q_r_prob_path):
	with open(q_path, "r") as q_f:
		with open(a_path, "r") as a_f:
			with open(q_in_a_path, "rb") as q_in_a_f:
				with open(q_s_prob_path, "r") as q_s_prob_f:
					with open(q_r_prob_path, "r") as q_r_prob_f:
						data = []
						tar_in_srcs = pickle.load(q_in_a_f)
						i = 0
						for q_line in q_f:
							a_line = a_f.readline()
							q_s_prob_line = q_s_prob_f.readline()
							q_r_prob_line = q_r_prob_f.readline()
							q = [int(x) for x in q_line.split()]
							a = [int(x) for x in a_line.split()]
							tar_in_src = tar_in_srcs[i]
							i += 1
							q_s_prob = [float(x) for x in q_s_prob_line.split()]
							q_r_prob = [float(x) for x in q_r_prob_line.split()]
							#注意最后输送训练数据时，把a放在前面而q在后面，分别对应encoder和decoder。
							data.append([a, q, tar_in_src, q_s_prob, q_r_prob])
	return data

def reload(q_path, a_path, q_in_a_path, q_s_prob_path, q_r_prob_path):
	with open(q_path, "r") as q_f:
		with open(a_path, "r") as a_f:
			# with open(q_in_a_path, "rb") as q_in_a_f:
				with open(q_s_prob_path, "r") as q_s_prob_f:
					with open(q_r_prob_path, "r") as q_r_prob_f:
						data = []
						#tar_in_srcs = pickle.load(q_in_a_f)
						i = 0
						for q_line in q_f:
							a_line = a_f.readline()
							q_s_prob_line = q_s_prob_f.readline()
							q_r_prob_line = q_r_prob_f.readline()
							q = [int(x) for x in q_line.split()]
							a = [int(x) for x in a_line.split()]
							#tar_in_src = tar_in_srcs[i]
							i += 1
							q_s_prob = [float(x) for x in q_s_prob_line.split()]
							q_r_prob = [float(x) for x in q_r_prob_line.split()]
							#注意最后输送训练数据时，把a放在前面而q在后面，分别对应encoder和decoder。
							data.append([a, q, q_s_prob, q_r_prob])
	return data

def fake_probs(q_path, q_s_prob_path, q_r_prob_path):
	with open(q_path, "r") as q_f:
		with open(q_s_prob_path, "w") as q_s_prob_path:
			with open(q_r_prob_path, "w") as q_r_prob_path:
				for line in q_f:
					num = len(line.split())
					fake_s = [0.0]*num
					fake_r = [0.0]*num
					q_s_prob_path.write(" ".join(str(s) for s in fake_s) + "\n")
					q_r_prob_path.write(" ".join(str(r) for r in fake_r) + "\n")
'''
def cPickle2pickle(entity_mid_name_path, train_q_in_a_path, entity_mid_name_path2, train_q_in_a_path2):
	with open(entity_mid_name_path, "r") as f1:
		with open(train_q_in_a_path, "r") as f2:
			with open(entity_mid_name_path2, "w") as f3:
				with open(train_q_in_a_path2, "w") as f4:
					data1 = cPickle.load(f1)
					data2 = cPickle.load(f2)
					data3 = pickle.dump(data1, f3)
					data4 = pickle.dump(data2, f4)
'''

def format_qa_test(mid2name_path, raw_q_path, raw_a_path, q_path, a_path):
	with open(raw_a_path, "r") as r_a_f:
		with open(a_path, "w") as a_f:
			with codecs.open(raw_q_path, "r",encoding='utf-8', errors='ignore') as r_q_f:
				with open (q_path, "w") as q_f:
					r_q_line = r_q_f.readline()
					q = ' '.join(standard_tokenizer(r_q_line.lower()))
					for r_a_line in r_a_f:
						if r_a_line == "\n":
							q_f.write("\n")
							a_f.write("\n")
							r_q_f.readline()
							r_q_line = r_q_f.readline()
							q = ' '.join(standard_tokenizer(r_q_line.lower()))
						else:
							q_f.write(q + "\n")
							tokens = tab_tokenizer(r_a_line)
							sub = "_bos " + ' '.join(standard_tokenizer(tokens[2].strip("\"").lower())) + " _eos"
							rel = "_bor " + tokens[1].lower().replace(".", " ").replace("_", " ") + " _eor"
							a = sub + " " + rel
							a_f.write(a + "\n")

def format_qa_test_new(test_raw_path, test_q_path, test_a_path, isanswer_path, correct_subId_path):
	correct_subId_f = open(correct_subId_path, "w")
	with open(test_raw_path, "r") as r_f:
		with open(test_q_path, "w") as q_f:
			with open(test_a_path, "w") as a_f:
				with open(isanswer_path, "w") as ia_f:		
					for line in r_f:
						if line == "\n":
							q_f.write("\n")
							a_f.write("\n")
							ia_f.write("\n")
							correct_subId_f.write("\n")
						else:
							tokens = tab_tokenizer(line)
							tokens[4] = tokens[4].strip()
							q = ' '.join(standard_tokenizer(tokens[4].lower()))
							# q = q.replace('< e >', '<e>')
							sub = "_bos " + ' '.join(standard_tokenizer(tokens[3].strip("\"").lower())) + " _eos"
							rel = "_bor " + tokens[2].replace(".", " ").replace("_", " ").lower() + " _eor"
							a = sub + ' ' + rel

							cand_subId = tokens[1]
							cand_rel = tokens[2]
							cand_sub_alias = tokens[3]

							ia = tokens[0].strip()
							q_f.write(q + "\n")
							a_f.write(a + "\n")
							ia_f.write(ia + "\n")
							correct_subId_f.write(cand_subId + ' ' + cand_rel + '\t' + cand_sub_alias + '\n')
def format_qa_train_new(train_raw_path, train_q_path, train_a_path, isanswer_path):
	limited = 30
	a_q_label_list = []
	example_a_q_label_list = []
	with open(train_raw_path, "r") as r_f:	
		for line in r_f:
			if line == "\n":
				example_a_q_label_list = sorted(example_a_q_label_list, key = lambda x: x[2], reverse = True)
				a_q_label_list.append(example_a_q_label_list[0:limited])
				example_a_q_label_list = []
				# q_f.write("\n")
				# a_f.write("\n")
				# ia_f.write("\n")
			else:
				tokens = tab_tokenizer(line)
				# tokens[4] = tokens[4].replace(tokens[8], _ent).strip()
				tokens[4] = tokens[4].strip()
				q = ' '.join(standard_tokenizer(tokens[4].lower()))
				# q = q.replace('< e >', '<e>')
				sub = "_bos " + ' '.join(standard_tokenizer(tokens[3].strip("\"").lower())) + " _eos"
				rel = "_bor " + tokens[2].replace(".", " ").replace("_", " ").lower() + " _eor"
				a = sub + " " + rel
				ia = tokens[0].strip()
				a_q_label = [a, q, ia]
				example_a_q_label_list.append(a_q_label)
				# q_f.write(q + "\n")
				# a_f.write(a + "\n")
				# ia_f.write(ia + "\n")
	with open(train_q_path, "w") as q_f:
			with open(train_a_path, "w") as a_f:
				with open(isanswer_path, "w") as ia_f:
					for example in a_q_label_list:
						for element in example:
							a_f.write(element[0] + '\n')
							q_f.write(element[1] + '\n')
							ia_f.write(element[2] + '\n')
						q_f.write("\n")
						a_f.write("\n")
						ia_f.write("\n")

def format_qa_train_bczm(train_raw_path, train_q_path, train_a_path, isanswer_path):
	limited = 30
	a_q_label_list = []
	example_a_q_label_list = []
	flag = 0
	with codecs.open(train_raw_path, "r", "utf-8-sig") as r_f:
		for line in r_f:
			if(line == '\n'):
				example_a_q_label_list = sorted(example_a_q_label_list, key = lambda x: x[2], reverse = True)
				
				example_a_q_label_list_one_true = [example_a_q_label_list[0]]
				for i in xrange(1,len(example_a_q_label_list)):
					if(example_a_q_label_list[i][2] == 1):
						continue
					else:
						example_a_q_label_list_one_true.append(example_a_q_label_list[i])
				a_q_label_list.append(example_a_q_label_list_one_true[0:limited])
				break
			tokens = tab_tokenizer(line)
			if(flag == 0):
				prev = tokens[1].strip()
				flag = 1
			else:
				flag = 1
			if prev != tokens[1].strip():
				# print(len(example_a_q_label_list))
				example_a_q_label_list = sorted(example_a_q_label_list, key = lambda x: x[2], reverse = True)
				
				example_a_q_label_list_one_true = [example_a_q_label_list[0]]
				for i in xrange(1,len(example_a_q_label_list)):
					if(example_a_q_label_list[i][2] == 1):
						continue
					else:
						example_a_q_label_list_one_true.append(example_a_q_label_list[i])
				a_q_label_list.append(example_a_q_label_list_one_true[0:limited])
				example_a_q_label_list = []
				# tokens[4] = tokens[4].replace(tokens[8], _ent).strip()
				tokens[1] = tokens[1].strip()
				q = ' '.join(standard_tokenizer(tokens[1]))
				# q = q.replace('< e >', '<e>')
				a = tokens[2].strip()
				ia = int(tokens[0].strip())
				# print(ia)
				a_q_label = [a, q, ia]
				example_a_q_label_list.append(a_q_label)
				prev = tokens[1].strip()
				# q_f.write("\n")
				# a_f.write("\n")
				# ia_f.write("\n")
				# break
			else:
				
				# tokens[4] = tokens[4].replace(tokens[8], _ent).strip()
				tokens[1] = tokens[1].strip()
				q = ' '.join(standard_tokenizer(tokens[1]))
				# q = q.replace('< e >', '<e>')
				a = tokens[2].strip()
				ia = int(tokens[0].strip())
				# print(ia)
				a_q_label = [a, q, ia]
				example_a_q_label_list.append(a_q_label)
				# q_f.write(q + "\n")
				# a_f.write(a + "\n")
				# ia_f.write(ia + "\n")
				prev = tokens[1]
	with open(train_q_path, "w") as q_f:
			with open(train_a_path, "w") as a_f:
				with open(isanswer_path, "w") as ia_f:
					for example in a_q_label_list:
						for element in example:
							a_f.write(element[0] + '\n')
							q_f.write(element[1] + '\n')
							ia_f.write(str(element[2]) + '\n')
						q_f.write("\n")
						a_f.write("\n")
						ia_f.write("\n")

def format_qa_test_bczm(test_raw_path, test_q_path, test_a_path, isanswer_path):
	# limited = 30
	a_q_label_list = []
	example_a_q_label_list = []
	flag = 0
	with codecs.open(test_raw_path, "r", "utf-8-sig") as r_f:
		for line in r_f:
			if(line == '\n'):
				example_a_q_label_list = sorted(example_a_q_label_list, key = lambda x: x[2], reverse = True)
				
				example_a_q_label_list_one_true = [example_a_q_label_list[0]]
				for i in xrange(1,len(example_a_q_label_list)):
					if(example_a_q_label_list[i][2] == 1):
						continue
					else:
						example_a_q_label_list_one_true.append(example_a_q_label_list[i])
				a_q_label_list.append(example_a_q_label_list_one_true)
				break
			tokens = tab_tokenizer(line.strip())
			if(flag == 0):
				prev = tokens[1].strip()
				flag = 1
			else:
				flag = 1
			if prev != tokens[1].strip():
				# print(len(example_a_q_label_list))
				# example_a_q_label_list = sorted(example_a_q_label_list, key = lambda x: x[2], reverse = True)
				a_q_label_list.append(example_a_q_label_list)
				example_a_q_label_list = []
				tokens[1] = tokens[1].strip()
				q = ' '.join(standard_tokenizer(tokens[1]))
				# q = q.replace('< e >', '<e>')
				a = tokens[2].strip()
				ia = int(tokens[0].strip())
				# print(ia)
				a_q_label = [a, q, ia]
				example_a_q_label_list.append(a_q_label)
				prev = tokens[1].strip()
				
				# q_f.write("\n")
				# a_f.write("\n")
				# ia_f.write("\n")
				# break
			else:
				
				# tokens[4] = tokens[4].replace(tokens[8], _ent).strip()
				tokens[1] = tokens[1].strip()
				q = ' '.join(standard_tokenizer(tokens[1]))
				# q = q.replace('< e >', '<e>')
				a = tokens[2].strip()
				ia = int(tokens[0].strip())
				# print(ia)
				
				a_q_label = [a, q, ia]
				example_a_q_label_list.append(a_q_label)
				# q_f.write(q + "\n")
				# a_f.write(a + "\n")
				# ia_f.write(ia + "\n")
				prev = tokens[1]
	with open(test_q_path, "w") as q_f:
			with open(test_a_path, "w") as a_f:
				with open(isanswer_path, "w") as ia_f:
					for example in a_q_label_list:
						for element in example:
							a_f.write(element[0] + '\n')
							q_f.write(element[1] + '\n')
							ia_f.write(str(element[2]) + '\n')
						q_f.write("\n")
						a_f.write("\n")
						ia_f.write("\n")

def find_answer_for_question_test(raw_q_path, raw_a_path, qa_path, isanswer_path):
	qa_dict = {}
	with open(qa_path, "r") as qa_f:
		for line in qa_f:
			tokens = tab_tokenizer(line)
			smid = tokens[0].replace("www.freebase.com","").lstrip("/").replace("/",".")
			rel = tokens[1].lower().replace("www.freebase.com","").lstrip("/").replace("/",".")
			q = tuple(word_tokenize(tokens[3].lower()))
			qa_dict[q] = (smid, rel)
	with open(raw_a_path, "r") as r_a_f:
		with codecs.open(raw_q_path, "r",encoding='utf-8', errors='ignore') as r_q_f:
			with open(isanswer_path, "w") as ia_f:
				r_q_line = r_q_f.readline()
				q = tuple(word_tokenize(r_q_line.lower()))
				if q in qa_dict:
					a = qa_dict[q]
				else:
					a = ""
				for r_a_line in r_a_f:
					if r_a_line == "\n":
						ia_f.write("\n")
						r_q_f.readline()
						r_q_line = r_q_f.readline()
						q = tuple(word_tokenize(r_q_line.lower()))
						if q in qa_dict:
							a = qa_dict[q]
						else:
							a = ""
					else:
						tokens = tab_tokenizer(r_a_line)
						if a == (tokens[0], tokens[1]):
							isanswer = 1
						else:
							isanswer = 0
						ia_f.write(str(isanswer) + "\n")

def reload_test_old(q_path, a_path, q_in_a_path, q_s_prob_path, q_r_prob_path, isanswer_path):
	with open(q_path, "r") as q_f:
		with open(a_path, "r") as a_f:
			with open(q_in_a_path, "rb") as q_in_a_f:
				with open(q_s_prob_path, "r") as q_s_prob_f:
					with open(q_r_prob_path, "r") as q_r_prob_f:
						with open(isanswer_path, "r") as ia_f:
							data = []
							data_sample = []
							tar_in_srcs = pickle.load(q_in_a_f)
							i = 0
							for q_line in q_f:
								a_line = a_f.readline()
								ia_line = ia_f.readline()
								q_s_prob_line = q_s_prob_f.readline()
								q_r_prob_line = q_r_prob_f.readline()
								i += 1
								if a_line == "\n":
									if len(data_sample) > 0:
										data.append(data_sample)
										data_sample = []
								else:
									q = [int(x) for x in q_line.split()]
									a = [int(x) for x in a_line.split()]
									q_s_prob = [float(x) for x in q_s_prob_line.split()]
									q_r_prob = [float(x) for x in q_r_prob_line.split()]
									ia = int(ia_line)
									tar_in_src = tar_in_srcs[i-1]
									#注意最后输送训练数据时，把a放在前面而q在后面，分别对应encoder和decoder。
									data_sample.append([a, q, tar_in_src, q_s_prob, q_r_prob, ia])
	return data


def reload_test(q_path, a_path, q_in_a_path, q_s_prob_path, q_r_prob_path, isanswer_path):
	max_len = 0
	with open(q_path, "r") as q_f:
		with open(a_path, "r") as a_f:
			with open(q_in_a_path, "rb") as q_in_a_f:
				with open(q_s_prob_path, "r") as q_s_prob_f:
					with open(q_r_prob_path, "r") as q_r_prob_f:
						with open(isanswer_path, "r") as ia_f:
							data = []
							data_sample = []
							tar_in_srcs = pickle.load(q_in_a_f)
							i = 0
							for q_line in q_f:
								a_line = a_f.readline()
								ia_line = ia_f.readline()
								q_s_prob_line = q_s_prob_f.readline()
								q_r_prob_line = q_r_prob_f.readline()
								i += 1
								# print(a_line)
								if a_line == "\n":
									if len(data_sample) > 0:

										# data_sample = sorted(data_sample, key = lambda x: x[4] ,reverse = True)
										# print(data_sample[0])
										data.append(data_sample)
										data_sample = []
								else:
									# print(a_line)
									q = [int(x) for x in q_line.split()]
									a = [int(x) for x in a_line.split()]
									q_s_prob = [float(x) for x in q_s_prob_line.split()]
									q_r_prob = [float(x) for x in q_r_prob_line.split()]
									# print(ia_line)
									ia = int(ia_line.strip().strip())
									tar_in_src = tar_in_srcs[i-1]
									
									#注意最后输送训练数据时，把a放在前面而q在后面，分别对应encoder和decoder。
									data_sample.append([a, q, tar_in_src, q_s_prob, q_r_prob, ia])
									if len(a) > max_len:
										max_len = len(a)
										print(a)
										print(max_len)
	print(len(data))
	return data

def reload_test_content(q_path, a_path):
	with open(q_path, "r") as q_f:
		with open(a_path, "r") as a_f:
			content = []
			content_sample = []
			for q_line in q_f:
				a_line = a_f.readline()
				if a_line == "\n":
					if len(content_sample) > 0:
						content.append(content_sample)
						content_sample = []
				else:
					content_sample.append([a_line, q_line])
	return content


def reload_test_content_new(q_path, a_path, correct_subId_path):
	with open(q_path, "r") as q_f:
		with open(correct_subId_path, "r") as c_f:
			content = []
			content_sample = []
			for q_line in q_f:
				c_line = c_f.readline().strip().split('\t')
				if q_line == "\n":
					if len(content_sample) > 0:
						content.append(content_sample)
						content_sample = []
				else:
					content_sample.append([c_line[0], q_line, c_line[1]])
	return content

def cut_raw_data(before, after):
	with open(before, 'r') as f_r:
		with open(after, 'w') as f_w:
			for line in f_r:
				tokens = jieba.cut(line.strip())
				line = ' '.join(tokens)
				f_w.write(line + '\n')


def prepare_data(data_path, src_vocabulary_size, tar_vocabulary_size, mode):

	# all_rel_list = pickle.load(open('./data_rel_pat/all_rel_list', 'rb'))
	# train_ques_mention_dict = get_train_ques_mention_dict('./data_rel_pat/data.train.focused_labeling')

	if mode == "train":
		train_qa_raw_path = os.path.join(data_path, "train.txt")
		entity_mid_name_path = os.path.join(data_path, "all_merge_dict")
		train_q_path = os.path.join(data_path, "q")
		train_a_path = os.path.join(data_path, "a")
		train_q_in_a_path = os.path.join(data_path, "q_in_a")
		src_vocab_path = os.path.join(data_path, "src_vocab")
		tar_vocab_path = os.path.join(data_path, "tar_vocab")
		train_q_ids_path = os.path.join(data_path, "q_ids")
		train_a_ids_path = os.path.join(data_path, "a_ids")
		train_q_s_probs_path = os.path.join(data_path, "q_s_probs")
		train_q_r_probs_path = os.path.join(data_path, "q_r_probs")
		train_isanswer_path = os.path.join(data_path, "isanswer_train")
		train_q_cut = os.path.join(data_path, "q_cut")
		train_a_cut = os.path.join(data_path, "a_cut")
		vocab_path = os.path.join(data_path, "vocab.pickle")
		vocab_word_path = os.path.join(data_path, 'vocab_word_index')
		src_embedding_dict_path = os.path.join(data_path, 'src_embedding_dict')
		tar_embedding_dict_path = os.path.join(data_path, 'tar_embedding_dict')
		pre_embed_path = os.path.join(data_path, 'vocab_embed.pickle')
		
		# format_qa_train_bczm(train_qa_raw_path, train_q_path, train_a_path, train_isanswer_path)
		# cut_raw_data(train_q_path, train_q_cut)
		# cut_raw_data(train_a_path, train_a_cut)
		
		# find_target_in_source(train_a_cut, train_q_cut, train_q_in_a_path)
		# pre_embed = pickle.load(open(pre_embed_path, 'rb'))
		# create_vocabulary_and_embed_dict(tar_vocab_path, train_q_cut, tar_embedding_dict_path, pre_embed)
		# create_vocabulary_and_embed_dict(src_vocab_path, train_a_cut, src_embedding_dict_path, pre_embed)
		
		# data_to_tok_ids(train_a_cut, train_a_ids_path, src_vocab_path)
		# data_to_tok_ids(train_q_cut, train_q_ids_path, tar_vocab_path)
		# # # # # 补充假的概率数据
		# fake_probs(train_q_ids_path, train_q_s_probs_path, train_q_r_probs_path)

		return reload_test(train_q_ids_path, train_a_ids_path, train_q_in_a_path, train_q_s_probs_path, train_q_r_probs_path, train_isanswer_path),\
			reload_test_content(train_q_path, train_a_path)
	else:
		entity_mid_name_path = os.path.join(data_path, "all_merge_dict")
		test_q_raw_path = os.path.join(data_path, "q_raw_test")
		test_a_raw_path = os.path.join(data_path, "a_raw_test")
		test_raw_path = os.path.join(data_path, "dev.txt")
		test_q_path = os.path.join(data_path, "q_test")
		test_a_path = os.path.join(data_path, "a_test")
		test_q_ids_path = os.path.join(data_path, "q_ids_test")
		test_a_ids_path = os.path.join(data_path, "a_ids_test")
		src_vocab_path = os.path.join(data_path, "src_vocab")
		tar_vocab_path = os.path.join(data_path, "tar_vocab")
		# test_qa_path = os.path.join(data_path, "annotated_fb_data_test.txt")
		test_isanswer_path = os.path.join(data_path, "isanswer_test")
		test_q_in_a_path = os.path.join(data_path, "q_in_a_test")
		test_q_s_probs_path = os.path.join(data_path, "q_s_probs_test")
		test_q_r_probs_path = os.path.join(data_path, "q_r_probs_test")
		correct_subId_path = os.path.join(data_path, "correct_subId")
		vocab_path = os.path.join(data_path, "vocab.pickle")
		test_q_cut = os.path.join(data_path, "test_q_cut")
		test_a_cut = os.path.join(data_path, "test_a_cut")
		# format_qa_test_bczm(test_raw_path, test_q_path, test_a_path, test_isanswer_path)
		# cut_raw_data(test_q_path, test_q_cut)
		# cut_raw_data(test_a_path, test_a_cut)
		# find_target_in_source(test_a_cut, test_q_cut, test_q_in_a_path)
		# data_to_tok_ids(test_a_cut, test_a_ids_path, src_vocab_path)
		# data_to_tok_ids(test_q_cut, test_q_ids_path, tar_vocab_path)
		# fake_probs(test_q_ids_path, test_q_s_probs_path, test_q_r_probs_path)
		
		return reload_test(test_q_ids_path, test_a_ids_path, test_q_in_a_path, test_q_s_probs_path, test_q_r_probs_path, test_isanswer_path), \
			reload_test_content(test_q_path, test_a_path)

def get_max_len(filename):
	max_len = 0
	with open(filename, 'r') as f:
		for line in f:
			ids = line.split(' ')
			length = len(ids)
			if(max_len < length):
				max_len = length
				print(line)
				print(length)

if __name__ == '__main__':
	# ent_rel_dict = pickle.load(open('./data_rel_pat/ls_sub_rels_dict', 'r'))
	# all_rel_list = get_all_rel_list(ent_rel_dict, './data_rel_pat/all_rel_list')
	

	# print(len(train_ques_mention_dict))
	prepare_data("./data", 10000, 10000, "test")
	# get_max_len('./data_rel_pat_with_neg_sample_and_copy/a_ids')