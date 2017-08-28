# -*- coding: utf-8 -*-
import os
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from six.moves import xrange
import data_utils_rel_pat_with_neg_sample_and_copy
from qamodel_linear_init_mlp_and_complex_attention import QAModel
import math
import numpy as np
import pickle

ISOTIMEFORMAT="%Y-%m-%d %X"

def get_batch(data, encoder_size, decoder_size, batch_size):
	encoder_inputs = []
	decoder_inputs = []
	decoder_targets_tran = []
	tar_in_srcs = []
	tar_src_entity_probs = []
	tar_src_relation_probs = []
	for _ in xrange(batch_size):
		encoder_input, decoder_input, tar_in_src, tar_src_entity_prob, tar_src_relation_prob = random.choice(data)
		if len(encoder_input) > encoder_size:
			encoder_input = encoder_input[0:encoder_size-1]
		if len(decoder_input) > decoder_size -1:
			decoder_input = decoder_input[0:decoder_size-2]
		if len(tar_in_src) > decoder_size:
			tar_in_src = tar_in_src[0:decoder_size-1]
		if len(tar_src_entity_prob) > decoder_size:
			tar_src_entity_prob = tar_src_entity_prob[0:decoder_size-1]
		if len(tar_src_relation_prob) > decoder_size:
			tar_src_relation_prob = tar_src_relation_prob[0:decoder_size-1]
		for i in xrange(len(tar_in_src)):
			if len(tar_in_src[i]) > encoder_size:
				tar_in_src[i] = tar_in_src[i][0:encoder_size-1]

		encoder_input += [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (encoder_size - len(encoder_input))
		encoder_inputs.append(encoder_input)
		decoder_target_tran = decoder_input + [data_utils_rel_pat_with_neg_sample_and_copy.end_id] + [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (decoder_size - len(decoder_input) - 1)
		decoder_targets_tran.append(decoder_target_tran)
		decoder_input = [data_utils_rel_pat_with_neg_sample_and_copy.go_id] + decoder_input + [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (decoder_size - len(decoder_input) - 1)
		decoder_inputs.append(decoder_input)
		
		for i in xrange(len(tar_in_src)):
			tar_in_src[i] += [0] * (encoder_size - len(tar_in_src[i]))
		tar_in_src += [[0] * encoder_size] * (decoder_size - len(tar_in_src))
		tar_in_srcs.append(tar_in_src)
		tar_src_entity_prob += [0]*(decoder_size - len(tar_src_entity_prob))
		tar_src_relation_prob += [0]*(decoder_size - len(tar_src_relation_prob))
		tar_src_entity_probs.append(tar_src_entity_prob)
		tar_src_relation_probs.append(tar_src_relation_prob)
	return Variable(torch.LongTensor(encoder_inputs)), Variable(torch.LongTensor(decoder_inputs)), Variable(torch.LongTensor(decoder_targets_tran)), \
		Variable(torch.FloatTensor(tar_in_srcs)), Variable(torch.FloatTensor(tar_src_entity_probs)), Variable(torch.FloatTensor(tar_src_relation_probs))

def save_config(config_path, attention, dropout_p, batch_size, encoder_ntoken, decoder_ntoken, encoder_max_len, decoder_max_len, \
	encoder_embedding_size, decoder_embedding_size, encoder_hidden_size, decoder_hidden_size, attention_hidden_size, init_range, learning_rate, decay_rate, epoch, message):
	with open(config_path, "w") as c_f:
		c_f.write("attention = " + str(attention) + "\n")
		c_f.write("dropout_p = " + str(dropout_p) + "\n")
		c_f.write("batch_size = " + str(batch_size) + "\n")
		c_f.write("encoder_ntoken = " + str(encoder_ntoken) + "\n")
		c_f.write("decoder_ntoken = " + str(decoder_ntoken) + "\n")
		c_f.write("encoder_max_len = " + str(encoder_max_len) + "\n")
		c_f.write("decoder_max_len = " + str(decoder_max_len) + "\n")
		c_f.write("encoder_embedding_size = " + str(encoder_embedding_size) + "\n")
		c_f.write("decoder_embedding_size = " + str(decoder_embedding_size) + "\n")
		c_f.write("encoder_hidden_size = " + str(encoder_hidden_size) + "\n")
		c_f.write("decoder_hidden_size = " + str(decoder_hidden_size) + "\n")
		c_f.write("attention_hidden_size = " + str(attention_hidden_size) + "\n")
		c_f.write("init_range = " + str(init_range) + "\n")
		c_f.write("learning_rate = " + str(learning_rate) + "\n")
		c_f.write("decay_rate = " + str(decay_rate) + "\n")
		c_f.write("epoch = " + str(epoch) + "\n")
		c_f.write("message = " + str(message) + "\n")

def lookupEmbed(encoder_inputs, embedding_dict, batch_size, max_len, embedding_size):
	encoder_inputs_embedding = []
	encoder_input_embedding = []
	for batch_step in range(batch_size):
		for id in encoder_inputs[batch_step]:
			encoder_input_embedding.append(embedding_dict[id])
		encoder_inputs_embedding.append(encoder_input_embedding)
		encoder_input_embedding = []
	# print(len(encoder_inputs_embedding))
	return encoder_inputs_embedding


if __name__ == '__main__':
	attention = True
	dropout_p = 0
	batch_size = 20
	encoder_ntoken = 265521
	decoder_ntoken = 21959
	encoder_max_len = 60
	decoder_max_len = 30
	encoder_embedding_size = 200
	decoder_embedding_size = 200
	encoder_hidden_size = 64
	decoder_hidden_size = 64
	attention_hidden_size = 64
	init_range = 0.08
	learning_rate = 0.003
	decay_rate = 0.95
	epoch = 100
	mode = "train"
	data_path = "./data"
	model_path = "./model"
	config_path = "./config"
	start_time = time.strftime(ISOTIMEFORMAT, time.localtime())
	file_name = os.path.basename(__file__)
	message = "改动：对模型几个简化的部分进行了补充编写"
	# exist_model_path = "./model/train_model_linear_init_mlp_and_complex_attention_with_neg_sample.py_2017-05-30 09:13:29_model11.dat"
	
	src_embedding_dict_path = "./data/src_embedding_dict"
	tar_embedding_dict_path = "./data/tar_embedding_dict"
	
	train, _ = data_utils_rel_pat_with_neg_sample_and_copy.prepare_data(data_path, encoder_ntoken, decoder_ntoken, mode)
	print("finish preparing data")
	batch_num = int(len(train)/batch_size)
	print (batch_num)
	
	src_embedding_dict = pickle.load(open(src_embedding_dict_path, "rb"))
	tar_embedding_dict = pickle.load(open(tar_embedding_dict_path, "rb"))
	print(len(src_embedding_dict))
	print(len(tar_embedding_dict))
	# print(len(embedding_dict))
	model = QAModel(encoder_ntoken, decoder_ntoken, encoder_embedding_size, decoder_embedding_size, encoder_hidden_size, decoder_hidden_size, \
					 attention_hidden_size, encoder_max_len, decoder_max_len, attention=attention, dropout_p=dropout_p)
	if torch.cuda.is_available():
		model.cuda()
	model.init_weights(init_range)

	# if os.path.exists(exist_model_path):
	# 	print('load already exist model')
	# 	saved_state = torch.load(exist_model_path)
	# 	model.load_state_dict(saved_state)

	optimizer = optim.RMSprop(model.parameters(), lr = learning_rate, alpha=decay_rate)
	
	r_margin = 10
	tolerance = 100

	for e in xrange(epoch):
		epoch_loss = 0
		for b in xrange(batch_num):
			batch_loss = None
			print("\tTraining the %d batch" % b)
			#decoder_targets_tran:batch_size*decoder_max_len, 在每个位置上是问题词在目标端词库的序号
			#tar_in_srcs:batch_size*decoder_max_len*encoder_max_len, 在每个位置上是问题词在源端的0,1位置表示
			#tar_src_entity_probs:batch_size*decoder_max_len, 在每个问题上是问题词对实体基于“实体-单词”表的概率
			#tar_src_relation_probs:batch_size*decoder_max_len, 在每个问题上是问题词对关系基于“关系-单词”表的概率

			for example_index in xrange(batch_size):
				flag = 0
				example_loss = None
				example = random.choice(train)
				predict = []
				ground_truth = []
				mini_batch_size = len(example)
				if(mini_batch_size == 1):
					continue
				# rounds = [[example[0], example[random.randint(1,mini_batch_size-1)]]]
				rounds = [example[0:30]]
				for single_round in rounds:
					encoder_inputs = []
					decoder_inputs = []
					decoder_targets_tran = []
					tar_in_srcs = []
					tar_src_entity_probs = []
					tar_src_relation_probs = []
					round_size = len(single_round)
					# print("\t\t\tThis round has %d entries." % round_size)
					for entry in single_round:
						# print(entry)
						encoder_input, decoder_input, tar_in_src, tar_src_entity_prob, tar_src_relation_prob, is_answer = entry
						

						ground_truth.append(is_answer)

						if len(encoder_input) > encoder_max_len:
							encoder_input = encoder_input[0:encoder_max_len-1]
						if len(decoder_input) > decoder_max_len -1:
							decoder_input = decoder_input[0:decoder_max_len-2]
						if len(tar_in_src) > decoder_max_len:
							tar_in_src = tar_in_src[0:decoder_max_len-1]
						# if len(tar_src_entity_prob) > decoder_max_len:
						# 	tar_src_entity_prob = tar_src_entity_prob[0:decoder_max_len-1]
						# if len(tar_src_relation_prob) > decoder_max_len:
						# 	tar_src_relation_prob = tar_src_relation_prob[0:decoder_max_len-1]
						for i in xrange(len(tar_in_src)):
							if len(tar_in_src[i]) > encoder_max_len:
								tar_in_src[i] = tar_in_src[i][0:encoder_max_len-1]
						encoder_input = [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (encoder_max_len - len(encoder_input)) + encoder_input
						decoder_target_tran = decoder_input + [data_utils_rel_pat_with_neg_sample_and_copy.end_id] + [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (decoder_max_len - len(decoder_input) - 1)
						decoder_input = [data_utils_rel_pat_with_neg_sample_and_copy.go_id] + decoder_input + [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (decoder_max_len - len(decoder_input) - 1)

						

						for i in xrange(len(tar_in_src)):
							tar_in_src[i] = [0] * (encoder_max_len - len(tar_in_src[i])) + tar_in_src[i]
						tar_in_src += [[0] * encoder_max_len] * (decoder_max_len - len(tar_in_src))
						tar_in_srcs.append(tar_in_src)

						# tar_src_entity_prob += [0]*(decoder_max_len - len(tar_src_entity_prob))
						# tar_src_relation_prob += [0]*(decoder_max_len - len(tar_src_relation_prob))
						# tar_src_entity_probs.append(tar_src_entity_prob)
						# tar_src_relation_probs.append(tar_src_relation_prob)

						encoder_inputs.append(encoder_input)
						decoder_inputs.append(decoder_input)
						decoder_targets_tran.append(decoder_target_tran)

					# print(encoder_inputs)
					# print(decoder_inputs)
					# print(tar_in_srcs)
					# print(tar_src_entity_probs)
					# print(tar_src_relation_probs)
					# print(ground_truth)



					encoder_inputs_embedding = lookupEmbed(encoder_inputs, src_embedding_dict, round_size, encoder_max_len, encoder_embedding_size)
					decoder_inputs_embedding = lookupEmbed(decoder_inputs, tar_embedding_dict, round_size, decoder_max_len, decoder_embedding_size)



					# encoder_inputs, decoder_inputs = Variable(torch.LongTensor(encoder_inputs)), Variable(torch.LongTensor(decoder_inputs))

					encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs = Variable(torch.FloatTensor(encoder_inputs_embedding)),\
																Variable(torch.FloatTensor(decoder_inputs_embedding)), Variable(torch.FloatTensor(tar_in_srcs))
					


					if torch.cuda.is_available():
						encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs = encoder_inputs_embedding.cuda(), decoder_inputs_embedding.cuda(), tar_in_srcs.cuda()
	
					pred_tar, pred_src, pred_gate = model(round_size, encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs, feed_previous = False)

					for batch_step in xrange(round_size):
						example_prob = 0
						for time_step in xrange(decoder_max_len):
							pred_tar_pos = pred_tar[time_step][batch_step]
							prob_tran = pred_tar_pos[decoder_targets_tran[batch_step][time_step]]
							
							pred_src_pos = pred_src[time_step][batch_step]
							prob_copy = torch.dot(pred_src_pos, tar_in_srcs[batch_step][time_step])


							# prob_refer = tar_src_entity_probs[batch_step][time_step]*pred_att[time_step][batch_step][0] +\
							# 				tar_src_relation_probs[batch_step][time_step]*pred_att[time_step][batch_step][1]
							prob = prob_tran * pred_gate[time_step][batch_step][0] + prob_copy * pred_gate[time_step][batch_step][1]
							example_prob += prob
							# print(prob.data[0])
						if(flag == 0):
							true_ans_prob = example_prob
							example_loss = 0
						# if example_loss is None:
						# 	example_loss = torch.max(zero_cmp, r_margin - true_ans_prob + example_prob)
						# else:
						# 	example_loss += torch.max(zero_cmp, r_margin - true_ans_prob + example_prob)
						if(flag != 0):
							loss = r_margin - true_ans_prob + example_prob
							# print(loss)
							if example_loss is None:
								if(loss.data[0] > 0):
									example_loss = loss
								else:
									example_loss += 0
							else:
								if(loss.data[0] > 0):
									example_loss += loss
								else:
									example_loss += 0
						flag = 1
						# print(example_loss)
				if batch_loss is None:
					batch_loss = example_loss
				else:
					batch_loss += example_loss
			batch_loss /= batch_size

			
			print (batch_loss.data[0])
			optimizer.zero_grad()
		# print("backward_____sleep_10")
		# time.sleep(10)

			batch_loss.backward()
			optimizer.step()
			del batch_loss

		'''
			if torch.cuda.is_available():
				encoder_inputs, decoder_inputs, decoder_targets_tran, tar_in_srcs, tar_src_entity_probs, tar_src_relation_probs = \
				encoder_inputs.cuda(), decoder_inputs.cuda(), decoder_targets_tran.cuda(), tar_in_srcs.cuda(), tar_src_entity_probs.cuda(), tar_src_relation_probs.cuda()
			
			#pred_tar:decoder_max_len*batch_size*decoder_ntoken, 在每个位置上是对目标端词库的概率分布
			#pred_src:decoder_max_len*batch_size*encoder_max_len, 在每个位置上是对源端序列的概率分布
			#pred_att:decoder_max_len*batch_size*2, 在每个位置上是对实体和关系的注意力
			#pred_gate:decoder_max_len*batch_size*3, 在每个位置上是对三种模式的概率分布
			
			pred_tar, pred_src, pred_gate, pred_att = model(batch_size, encoder_inputs, decoder_inputs, tar_in_srcs, feed_previous=False)
			batch_loss = None

			for time_step in xrange(decoder_max_len):
				for batch_step in xrange(batch_size):
					pred_tar_pos = pred_tar[time_step][batch_step]
					prob_tran = pred_tar_pos[decoder_targets_tran[batch_step][time_step].data[0]]

					pred_src_pos = pred_src[time_step][batch_step]
					prob_copy = torch.dot(pred_src_pos, tar_in_srcs[batch_step][time_step])
					
					prob_refer = tar_src_entity_probs[batch_step][time_step]*pred_att[time_step][batch_step][0] +\
					tar_src_relation_probs[batch_step][time_step]*pred_att[time_step][batch_step][1]

					prob = prob_tran*pred_gate[time_step][batch_step][0] + prob_copy*pred_gate[time_step][batch_step][1] + prob_refer*pred_gate[time_step][batch_step][2]
					loss = -torch.log(prob)

					if batch_loss is None:
						batch_loss = loss
					else:
						batch_loss += loss

			batch_loss /= batch_size
			print (batch_loss.data[0])
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			epoch_loss += batch_loss.data[0]
		'''
		torch.save(model.state_dict(), os.path.join(model_path, file_name + "_" + start_time + "_model" + str(e+1) + ".dat"))
		save_config(os.path.join(config_path, file_name + "_" + start_time + "_config" + ".txt"), \
			attention, dropout_p, batch_size, encoder_ntoken, decoder_ntoken, encoder_max_len, decoder_max_len, \
			encoder_embedding_size, decoder_embedding_size, encoder_hidden_size, decoder_hidden_size, attention_hidden_size,\
			init_range, learning_rate, decay_rate, epoch, message)