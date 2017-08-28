import data_utils_rel_pat_with_neg_sample_and_copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from six.moves import xrange
 
def repackage_state(h):
	"""Wraps hidden states in new Variables, to detach them from their history."""
	if type(h) == Variable:
		return Variable(h.data)
	else:
		return tuple(repackage_state(v) for v in h)

class QAModel(nn.Module):
	def __init__(self, encoder_ntoken, decoder_ntoken, encoder_embedding_size, decoder_embedding_size, 
		encoder_hidden_size, decoder_hidden_size,  attention_hidden_size, encoder_max_len, decoder_max_len,
		attention, dropout_p=0.5, nlayers=1, bias=False):
		super(QAModel, self).__init__()

		self.dropout_p = dropout_p
		if dropout_p > 0:
			self.dropout = nn.Dropout(p=dropout_p)

		# self.encoder_embedding = nn.Embedding(encoder_ntoken, encoder_embedding_size)
		self.encoder = nn.LSTM(encoder_embedding_size, encoder_hidden_size, nlayers, bias=bias, batch_first=True, bidirectional=True)
		# self.decoder_embedding = nn.Embedding(decoder_ntoken, decoder_embedding_size)
		self.decoder = nn.LSTM(decoder_embedding_size+encoder_hidden_size*4, decoder_hidden_size, nlayers, bias=bias, batch_first=True)
		self.encoder_linear_att = nn.Linear(encoder_hidden_size*2, attention_hidden_size)
		self.decoder_linear_att = nn.Linear(decoder_hidden_size, attention_hidden_size)
		self.weight_linear_att = nn.Linear(attention_hidden_size, 1)
		self.decoder_linear_tran = nn.Linear(decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4, decoder_ntoken)
		self.decoder_linear_copy = nn.Linear(decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4, encoder_hidden_size*2)
		self.decoder_linear_gate1 = nn.Linear(decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4, \
			int((decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4)/4))
		self.decoder_linear_gate2 = nn.Linear(int((decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4)/4), \
			int((decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4)/16))
		self.decoder_linear_gate3 = nn.Linear(int((decoder_hidden_size+decoder_embedding_size+encoder_hidden_size*4)/16), 2)
		self.decoder_linear_hinit = nn.Linear(encoder_hidden_size*2, decoder_hidden_size)
		self.decoder_linear_cinit = nn.Linear(encoder_hidden_size*2, decoder_hidden_size)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax()
		self.sequential = nn.Sequential(self.decoder_linear_gate1, self.tanh, self.decoder_linear_gate2, self.tanh, self.decoder_linear_gate3)

		self.encoder_ntoken = encoder_ntoken
		self.decoder_ntoken = decoder_ntoken
		self.encoder_embedding_size = encoder_embedding_size
		self.decoder_embedding_size = decoder_embedding_size
		self.encoder_hidden_size = encoder_hidden_size
		self.decoder_hidden_size = decoder_hidden_size
		self.attention_hidden_size = attention_hidden_size
		self.encoder_max_len = encoder_max_len
		self.decode_max_len = decoder_max_len
		self.attention = attention
		self.nlayers = nlayers
		self.bias = bias

	def init_weights(self, initrange):
		for param in self.parameters():
			param.data.uniform_(-initrange, initrange)

	def find_subject_relation_index(self, encoder_inputs):
		# print("hi")
		subject_idx = []
		relation_idx = []
		for x in encoder_inputs:
			s = []
			status_s = 0
			r = []
			status_r = 0
			for y in x:
				# if y.data[0] == ll:
				# 	status_s = 0
				s.append(status_s)
				# if y.data[0] == data_utils_rel_pat_with_neg_sample_and_copy.bos_id:
				# 	status_s = 1
				# if y.data[0] == data_utils_rel_pat_with_neg_sample_and_copy.eor_id:
				# 	status_r = 0
				r.append(status_r)
				# if y.data[0] == data_utils_rel_pat_with_neg_sample_and_copy.bor_id:
				# 	status_r = 1
			subject_idx.append(s)
			relation_idx.append(r)
		return Variable(torch.FloatTensor(subject_idx)), Variable(torch.FloatTensor(relation_idx))


	def attention_func(self, batch_size, encoder_hiddens, decoder_hidden):
		# encoder_hiddens: batch_size * encoder_max_len * 2encoder_hidden_size, 每个位置是“实体-关系”对某词经双向LSTM编码的隐状态
		# decoder_hidden:batch_size * decoder_hidden_size, 每个位置是将用于生成问题词的单向LSTM解码的隐状态
		# attention:batch_size * encoder_max_len
		# 这块儿有好多问题，主要是Linear和想象的不一样，它要求输入必须是matrix，所以要进行一些捋平再回复的操作。
		encoder_hiddens_flat = encoder_hiddens.contiguous().view(-1, self.encoder_hidden_size*2)
		encoder_hiddens_after_linear = self.encoder_linear_att(encoder_hiddens_flat).view(batch_size, self.encoder_max_len, self.attention_hidden_size)
		decoder_hidden_after_linear = self.decoder_linear_att(decoder_hidden).unsqueeze(1).expand(batch_size, self.encoder_max_len, self.attention_hidden_size)
		attention = self.softmax(self.weight_linear_att(self.tanh(encoder_hiddens_after_linear +
			decoder_hidden_after_linear).view(-1, self.attention_hidden_size)).squeeze(1).view(batch_size, self.encoder_max_len))
		context = torch.bmm(torch.transpose(encoder_hiddens, 1, 2), attention.unsqueeze(2)).squeeze(2)
		#context:batch_size * 2encoder_hidden_size
		return attention, context

	def encode(self, batch_size, encoder_inputs_embedding):
		weight = next(self.parameters()).data
		init_state = (Variable(weight.new(self.nlayers * 2, batch_size, self.encoder_hidden_size).zero_()),
			Variable(weight.new(self.nlayers * 2, batch_size, self.encoder_hidden_size).zero_()))
		embedding = encoder_inputs_embedding
		if self.dropout_p > 0:
			embedding = self.dropout(embedding)
		# print(embedding)
		encoder_hiddens, encoder_state = self.encoder(embedding, init_state)
		#print(encoder_hiddens)

		return encoder_hiddens, encoder_state

	


	def decode(self, batch_size, encoder_inputs_embedding, encoder_hiddens, encoder_state, decoder_inputs_embedding, tar_in_srcs, feed_previous):
		
		pred_tar = []
		pred_src = []
		pred_gate = []
		pred_att = []

		state = (self.decoder_linear_hinit(torch.cat((encoder_state[0][0], encoder_state[0][1]), 1)).unsqueeze(0), \
			self.decoder_linear_cinit(torch.cat((encoder_state[1][0], encoder_state[1][1]), 1)).unsqueeze(0))
		# embedding = self.decoder_embedding(decoder_inputs[:, 0])
		
		embedding = decoder_inputs_embedding[:, 0]
		# print(embedding.size())
		#这块儿还可以再优化，包括（1）attention由state产生；(2)attention的变化过程。
		attention = Variable(torch.ones(batch_size, self.encoder_max_len))
		attention = torch.div(attention, torch.sum(attention, 1).expand_as(attention))
		if torch.cuda.is_available():
			attention = attention.cuda()

		context = torch.bmm(torch.transpose(encoder_hiddens, 1, 2), attention.unsqueeze(2)).squeeze(2)
		copytext = Variable(torch.zeros(batch_size, self.encoder_hidden_size*2))
		if torch.cuda.is_available():
			copytext = copytext.cuda()
		# subject_idx, relation_idx = self.find_subject_relation_index(encoder_inputs)
		# if torch.cuda.is_available():
		# 	subject_idx, relation_idx = subject_idx.cuda(), relation_idx.cuda()

		if not feed_previous:
			for time_step in xrange(self.decode_max_len):
				emb_contx_coptx = torch.cat((embedding, context, copytext), 1).unsqueeze(1)
				decoder_hidden, state = self.decoder(emb_contx_coptx, state)
				decoder_hidden = decoder_hidden.squeeze(1)
				attention, context = self.attention_func(batch_size, encoder_hiddens, decoder_hidden)

				# 给目标词库打分
				softmax_tar = self.softmax(self.decoder_linear_tran(torch.cat((decoder_hidden, embedding, context, copytext), 1)))
				# 给源端词打分
				softmax_src = self.softmax(torch.bmm(encoder_hiddens, self.decoder_linear_copy(torch.cat((decoder_hidden, embedding, context, copytext), 1)).unsqueeze(2)).squeeze(2))
				softmax_gate = self.softmax(self.sequential(torch.cat((decoder_hidden, embedding, context, copytext), 1)))
				# _att = torch.cat((torch.diag(torch.mm(attention, torch.transpose(subject_idx, 0, 1))).unsqueeze(1), \
				# 	torch.diag(torch.mm(attention, torch.transpose(relation_idx, 0, 1))).unsqueeze(1)), 1)

				if time_step < self.decode_max_len-1:
					x = torch.mul(softmax_src, tar_in_srcs[:, time_step])
					y = torch.div(x, torch.add(torch.sum(x, 1), 1e-8).expand_as(x)).unsqueeze(2).expand_as(encoder_hiddens)
					copytext = torch.sum(torch.mul(encoder_hiddens, y), 1).squeeze(1)
					# embedding = self.decoder_embedding(decoder_inputs[:, time_step+1])
					embedding = decoder_inputs_embedding[:, time_step+1]

				pred_tar.append(softmax_tar)
				pred_src.append(softmax_src)
				pred_gate.append(softmax_gate)
				# pred_att.append(_att)
		return pred_tar, pred_src, pred_gate

		'''
		subject_idx, relation_idx = self.find_subject_relation_index(encoder_inputs)
		if torch.cuda.is_available():
			subject_idx, relation_idx = subject_idx.cuda(), relation_idx.cuda()
		if not feed_previous:
			for time_step in xrange(self.decode_max_len):
				# embedding:batch_size * 1 * embedding_size, 虽然pytorch要求LSTM一次要处理一批且为序列，但我们可以让这序列的长度为1
				embedding = self.decoder_embedding(decoder_inputs[:, time_step].unsqueeze(1))
				embedding_expand = embedding, context

				decoder_hidden, state = self.decoder(embedding, state)
				decoder_hidden = decoder_hidden.squeeze(1)
				
				attention, context = self.attention_func(batch_size, encoder_hiddens, decoder_hidden)

				#copy反馈，1e-8是为了避免除0
				tar_in_src_step_exp = tar_in_srcs[:, time_step].unsqueeze(2).expand_as(encoder_hiddens)
				copy_feed = torch.div(torch.sum(torch.mul(encoder_hiddens, tar_in_src_step_exp), 1), torch.add(torch.sum(tar_in_src_step_exp, 1), 1e-8)).squeeze(1)

				# batch_size*deocder_ntoken
				softmax_tar = self.softmax(self.decoder_linear_tran(torch.cat((decoder_hidden, context, copy_feed), 1)))
				softmax_src = self.softmax(torch.bmm(encoder_hiddens, self.decoder_linear_copy(torch.cat((decoder_hidden, context, copy_feed), 1)).unsqueeze(2)).squeeze(2))
				
				# 运用多层感知机
				softmax_gate = self.softmax(self.sequential(torch.cat((decoder_hidden, embedding.squeeze(1), context), 1)))

				_att = torch.cat((torch.diag(torch.mm(attention, torch.transpose(subject_idx, 0, 1))).unsqueeze(1), 
					torch.diag(torch.mm(attention, torch.transpose(relation_idx, 0, 1))).unsqueeze(1)), 1)

				pred_tar.append(softmax_tar)
				pred_src.append(softmax_src)
				pred_gate.append(softmax_gate)
				pred_att.append(_att)

		#真能生成问题的模型，后面再补充
		return pred_tar, pred_src, pred_gate, pred_att
		'''

	def forward(self, batch_size, encoder_inputs_embedding, decoder_inputs_embedding, tar_in_srcs, feed_previous=False):
		# encoding
		encoder_hiddens, encoder_state = self.encode(batch_size, encoder_inputs_embedding)
		# decoding
		pred_tar, pred_src, pred_gate = self.decode(batch_size, encoder_inputs_embedding, encoder_hiddens, encoder_state, decoder_inputs_embedding, tar_in_srcs, feed_previous)
		return pred_tar, pred_src, pred_gate





