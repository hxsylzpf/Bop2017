# -*- coding: utf-8 -*-
import os
import random
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from six.moves import xrange
# import data_utils_rel_pat_with_neg_sample_and_copy_ls
from qamodel_linear_init_mlp_and_complex_attention import QAModel
import math
import data_utils_rel_pat_with_neg_sample_and_copy
import pickle
# from imp import reload
# import sys
# reload(sys)
# sys.setdefaultencoding("utf8")
# # import importlib,sys
# # importlib.reload(sys)

def lookupEmbed(encoder_inputs, embedding_dict, batch_size, max_len, embedding_size):
	num = 0
	encoder_inputs_embedding = []
	encoder_input_embedding = []
	for batch_step in range(batch_size):
		for id in encoder_inputs[batch_step]:
			if id not in embedding_dict:
				encoder_input_embedding.append(embedding_dict[1])
				num += 1
			else:
				encoder_input_embedding.append(embedding_dict[id])
		encoder_inputs_embedding.append(encoder_input_embedding)
		encoder_input_embedding = []
	# print(len(encoder_inputs_embedding))
	
	return encoder_inputs_embedding

if __name__ == '__main__':
	attention = True
	dropout_p = 0
	encoder_ntoken = 265521
	decoder_ntoken = 21959
	encoder_max_len = 60
	decoder_max_len = 30
	encoder_embedding_size = 200
	decoder_embedding_size = 200
	encoder_hidden_size = 64
	decoder_hidden_size = 64
	attention_hidden_size = 64
	data_path = "./data"
	model_path = "./model/train_model_linear_init_mlp_and_complex_attention_with_neg_sample.py_2017-05-30 09:13:29_model7.dat"
	analysis_path = model_path.replace("./model", "./analysis").replace(".dat", ".analysis")
	mode = "test"
	result_path = './data/result_score_3'
	# s_r_o_triple_path = data_path + '/freebase-FB5M_triple.pkl'
	# alias_dict_path = data_path + '/all_merge_dict'

	# with open(s_r_o_triple_path, 'rb') as f:
	# 	s_r_o_triple = pickle.load(f)
	# with open(alias_dict_path) as f:
	# 	alias_dict = pickle.load(f)
	src_embedding_dict_path = "./data/src_embedding_dict"
	tar_embedding_dict_path = "./data/tar_embedding_dict"
	src_embedding_dict = pickle.load(open(src_embedding_dict_path, "rb"))
	tar_embedding_dict = pickle.load(open(tar_embedding_dict_path, "rb"))

	test, content = data_utils_rel_pat_with_neg_sample_and_copy.prepare_data(data_path, encoder_ntoken, decoder_ntoken, mode)
	
	model = QAModel(encoder_ntoken, decoder_ntoken, encoder_embedding_size, decoder_embedding_size, encoder_hidden_size, decoder_hidden_size,\
				 attention_hidden_size, encoder_max_len, decoder_max_len, attention=attention, dropout_p=dropout_p)
	if torch.cuda.is_available():
		model.cuda()
	if os.path.exists(model_path):
		print("load already")
		saved_state = torch.load(model_path)
		model.load_state_dict(saved_state)

	example_num = 0
	correct_num = 0
	called_num = 0
	tolerance = 100

	f_result = open(result_path, 'w')
	with open(analysis_path, "w") as als_f:
		for example in test:
			example_info = []
			example_num += 1
			print("Testing the %d question:" % example_num)
			ground_truth = []
			predict= []
			example_content = content[example_num-1]
			batch_size = len(example)
			print("\tCandidates number: %d." % batch_size)
			rounds = [example[i:i+tolerance] for i in xrange(0, batch_size, tolerance)]
			print("\tRounds: %d." % len(rounds))
			for single_round in rounds:
				encoder_inputs = []
				decoder_inputs = []
				decoder_targets_tran = []
				tar_in_srcs = []
				
				round_size = len(single_round)
				print("\tThis round has %d entries." % round_size)
				for entry in single_round:
					encoder_input, decoder_input, tar_in_src, tar_src_entity_prob, tar_src_relation_prob, is_answer = entry
					ground_truth.append(is_answer)

					if len(encoder_input) > encoder_max_len:
						encoder_input = encoder_input[0:encoder_max_len-1]
					if len(decoder_input) > decoder_max_len -1:
						decoder_input = decoder_input[0:decoder_max_len-2]
					if len(tar_in_src) > decoder_max_len:
						tar_in_src = tar_in_src[0:decoder_max_len-1]
					
					for i in xrange(len(tar_in_src)):
						if len(tar_in_src[i]) > encoder_max_len:
							tar_in_src[i] = tar_in_src[i][0:encoder_max_len-1]
					encoder_input = [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (encoder_max_len - len(encoder_input)) + encoder_input
					decoder_target_tran = decoder_input + [data_utils_rel_pat_with_neg_sample_and_copy.end_id] + [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (decoder_max_len - len(decoder_input) - 1)
					decoder_input = [data_utils_rel_pat_with_neg_sample_and_copy.go_id] + decoder_input + [data_utils_rel_pat_with_neg_sample_and_copy.pad_id] * (decoder_max_len - len(decoder_input) - 1)
					for i in xrange(len(tar_in_src)):
						tar_in_src[i] = [0] * (encoder_max_len - len(tar_in_src[i])) + tar_in_src[i]
					tar_in_src += [[0] * encoder_max_len] * (decoder_max_len - len(tar_in_src))
					

					encoder_inputs.append(encoder_input)
					decoder_inputs.append(decoder_input)
					decoder_targets_tran.append(decoder_target_tran)
					tar_in_srcs.append(tar_in_src)
				
				encoder_inputs_embedding = lookupEmbed(encoder_inputs, src_embedding_dict, round_size, encoder_max_len, encoder_embedding_size)
				decoder_inputs_embedding = lookupEmbed(decoder_inputs, tar_embedding_dict, round_size, decoder_max_len, decoder_embedding_size)	

				encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs = Variable(torch.FloatTensor(encoder_inputs_embedding)), \
					Variable(torch.FloatTensor(decoder_inputs_embedding)), Variable(torch.FloatTensor(tar_in_srcs))
					

				if torch.cuda.is_available():
					encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs = \
											encoder_inputs_embedding.cuda(), decoder_inputs_embedding.cuda(), tar_in_srcs.cuda()

				pred_tar, pred_src, pred_gate = model(round_size, encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs, feed_previous=False)
				for batch_step in xrange(round_size):
					example_prob = 0
					for time_step in xrange(decoder_max_len):
						pred_tar_pos = pred_tar[time_step][batch_step]
						prob_tran = pred_tar_pos[decoder_targets_tran[batch_step][time_step]]

						pred_src_pos = pred_src[time_step][batch_step]
						prob_copy = torch.dot(pred_src_pos, tar_in_srcs[batch_step][time_step])

						prob = prob_tran*pred_gate[time_step][batch_step][0] + prob_copy*pred_gate[time_step][batch_step][1]
						example_prob += prob.data[0]
					predict.append(example_prob)
					example_info.append([ground_truth[batch_step], example_content[batch_step][1], example_prob, example_content[batch_step][0]])
			
			example_info = sorted(example_info, key = lambda x: x[2], reverse = True)
			for line in example_info:
				f_result.write(str(line[0]) + '\t' + str(line[1]).strip() + '\t' + str(line[2]) + '\t' + str(line[3]).strip()  + '\n')
			f_result.write('\n')
			predict_index = predict.index(max(predict))
			correct_index = ground_truth.index(max(ground_truth))
			# print(correct_index)
			if max(ground_truth) == 1:
				called_num += 1
			else:
				correct_index = -1
				print(example_content[0][1])
			if predict_index == correct_index:
				correct_num += 1
			# else:
			# 	# print(example_content)
			# 	question = example_content[correct_index][1]
			# 	als_f.write("Question num: %d" % example_num)
			# 	als_f.write("Question content: %s" % question)
			# 	als_f.write("prect: %d correct: %d\n" % (predict_index, correct_index))
			# 	if correct_index == -1:
			# 		als_f.write("The correct answer is uncalled!\n")
			# 	else:
			# 		correct_answer = example_content[correct_index][0]
			# 		correct_sub = example_content[correct_index][2]
			# 		correct_objId = s_r_o_triple[correct_answer]
			# 		# correct_obj = alias_dict[correct_objId]
			# 		correct_rel = example_content[correct_index][0].split(' ')[1]
			# 		wrong_rel = example_content[predict_index][0].split(' ')[1]

			# 		wrong_answer = example_content[predict_index][0]
			# 		wrong_sub = example_content[predict_index][2]
			# 		wrong_objId = s_r_o_triple[wrong_answer]
			# 		# wrong_obj = alias_dict[wrong_objId]
			# 		# if(correct_rel == wrong_rel and correct_sub == wrong_sub):
			# 		if(correct_objId == wrong_objId):
			# 			correct_num += 1
			# 			if(max(ground_truth) != 1):
			# 				called_num += 1

			# 		else:
			# 			als_f.write("The correct answer: %s\n" % (correct_answer))
			# 			als_f.write("The wrongly predicted answer: %s\n" % (wrong_answer))
			# 			als_f.write("The correct subject is: %s\n" % correct_sub)
			# 			als_f.write("The wrongly predicted subject is: %s\n" % wrong_sub)
			# 			als_f.write("The correct obj is: %s\n" % (correct_objId))
			# 			als_f.write("The wrong obj is: %s\n" % (wrong_objId))
			# 	als_f.write("\n")

			accuracy = correct_num/example_num
			print ("Accuracy is %f" % accuracy)
			recall = called_num/example_num
			print ("Recall is %f" % recall)
