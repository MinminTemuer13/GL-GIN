# -*- coding: utf-8 -*-#

import math
from concurrent.futures import ProcessPoolExecutor

from numba import njit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter
import numpy as np
from utils.process import normalize_adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__('attention_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.__args = args

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,
            self.__args.attention_hidden_dim,
            self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        return hiddens

def _isdac_worker_function_single_arg(args_tuple):
    """
    Accepts a single tuple: (processor_instance, log_probs_tensor, seq_len_val)
    """
    processor_instance, log_probs_tensor, seq_len_val = args_tuple # Unpack
    return processor_instance.process_single_sentence(log_probs_tensor, seq_len_val)


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize ISDACProcessor with parameters from args
        isdac_tau = getattr(args, 'isdac_tau', 0.01)
        isdac_min_len = getattr(args, 'isdac_min_len', 2)
        isdac_max_len = getattr(args, 'isdac_max_len', 10)
        isdac_avg_log_conf_threshold = getattr(args, 'isdac_avg_log_conf_threshold', math.log(0.6))
        isdac_overlap_iou = getattr(args, 'isdac_overlap_threshold_iou', 0.3)

        self.isdac_processor = ISDACProcessor(
            num_intent_labels=num_intent,  # Use the passed num_intent
            tau=isdac_tau,
            min_len=isdac_min_len,
            max_len=isdac_max_len,
            avg_log_conf_threshold=isdac_avg_log_conf_threshold,
            overlap_threshold_iou=isdac_overlap_iou
        )

        # Initialize an embedding object.
        self.__embedding = nn.Embedding(
            self.__num_word,
            self.__args.word_embedding_dim
        )
        self.G_encoder = Encoder(args)
        # Initialize an Decoder object for intent.
        self.__intent_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                      self.__args.encoder_hidden_dim + self.__args.attention_output_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim, self.__num_intent),
        )

        self.__intent_embedding = nn.Parameter(
            torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim))  # 191, 32
        nn.init.normal_(self.__intent_embedding.data)

        self.__slot_lstm = LSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim + num_intent,
            self.__args.slot_decoder_hidden_dim,
            self.__args.dropout_rate
        )
        self.__intent_lstm = LSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

        self.__slot_decoder = LSTMDecoder(
            args,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate)

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def generate_global_adj_gat(self, seq_len, index, batch, window):
        global_intent_idx = [[] for i in range(batch)]
        global_slot_idx = [[] for i in range(batch)]
        for item in index:
            global_intent_idx[item[0]].append(item[1])

        for i, len in enumerate(seq_len):
            global_slot_idx[i].extend(list(range(self.__num_intent, self.__num_intent + len)))

        adj = torch.cat([torch.eye(self.__num_intent + seq_len[0]).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in global_intent_idx[i]:
                adj[i, j, global_slot_idx[i]] = 1.
                adj[i, j, global_intent_idx[i]] = 1.
            for j in global_slot_idx[i]:
                adj[i, j, global_intent_idx[i]] = 1.

        for i in range(batch):
            for j in range(self.__num_intent, self.__num_intent + seq_len[i]):
                adj[i, j, max(self.__num_intent, j - window):min(seq_len[i] + self.__num_intent, j + window + 1)] = 1.

        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def generate_slot_adj_gat(self, seq_len, batch, window):
        slot_idx_ = [[] for i in range(batch)]
        adj = torch.cat([torch.eye(seq_len[0]).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(seq_len[i]):
                adj[i, j, max(0, j - window):min(seq_len[i], j + window + 1)] = 1.
        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def forward(self, text, seq_lens, n_predicts=None):
        word_tensor = self.__embedding(text)
        g_hiddens = self.G_encoder(word_tensor, seq_lens)
        intent_lstm_out = self.__intent_lstm(g_hiddens, seq_lens)
        intent_lstm_out = F.dropout(intent_lstm_out, p=self.__args.dropout_rate, training=self.training)
        pred_intent = self.__intent_decoder(intent_lstm_out)

        token_intent_log_probs = F.logsigmoid(pred_intent)

        batch_size = pred_intent.shape[0]
        all_batch_intents_tuples = []

        # .detach() is important if pred_intent has requires_grad=True and ISDAC is non-differentiable
        # and you don't want gradients flowing back from intent_index through ISDAC during training.
        token_intent_log_probs_cpu = token_intent_log_probs.cpu().detach()

        map_args_list = [
            (self.isdac_processor,
             token_intent_log_probs_cpu[i, :seq_lens[i], :],
             seq_lens[i])
            for i in range(batch_size)
        ]

        num_workers = getattr(self.__args, 'isdac_workers', 8)
        detected_intents_per_sample = [set() for _ in range(batch_size)]

        if batch_size > 0:
            if num_workers > 0:
                try:
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        results_iterator = executor.map(
                            _isdac_worker_function_single_arg,
                            map_args_list
                        )
                        for i, detected_set in enumerate(results_iterator):
                            detected_intents_per_sample[i] = detected_set
                except Exception as e:
                    print(f"ProcessPoolExecutor failed: {e}. Falling back to sequential processing.")
                    for i in range(batch_size):
                        args_for_sample_tuple = map_args_list[i]
                        detected_intents_per_sample[i] = _isdac_worker_function_single_arg(
                            args_for_sample_tuple
                        )
            else:
                for i in range(batch_size):
                    args_for_sample_tuple = map_args_list[i]
                    detected_intents_per_sample[i] = _isdac_worker_function_single_arg(
                        args_for_sample_tuple
                    )

        for i, detected_set in enumerate(detected_intents_per_sample):
            for intent_idx_val in detected_set:
                all_batch_intents_tuples.append((i, intent_idx_val))

        # Determine device for intent_index based on pred_intent's original device
        # This ensures intent_index is on GPU if the model is on GPU.
        output_device = pred_intent.device

        if not all_batch_intents_tuples:
            intent_index = torch.empty((0, 2), dtype=torch.long, device=output_device)
        else:
            intent_index = torch.tensor(all_batch_intents_tuples, dtype=torch.long, device=output_device)
        # --- ISDAC VOTING END ---

        slot_lstm_out = self.__slot_lstm(torch.cat([g_hiddens, pred_intent], dim=-1), seq_lens)
        global_adj = self.generate_global_adj_gat(seq_lens, intent_index, len(pred_intent),
                                                  self.__args.slot_graph_window)
        slot_adj = self.generate_slot_adj_gat(seq_lens, len(pred_intent), self.__args.slot_graph_window)
        pred_slot = self.__slot_decoder(
            slot_lstm_out, seq_lens,
            global_adj=global_adj,
            slot_adj=slot_adj,
            intent_embedding=self.__intent_embedding
        )
        if n_predicts is None:
            # Return log_softmax for slots, and raw logits for intent (as it was before)
            # pred_intent here are the raw logits from __intent_decoder
            return F.log_softmax(pred_slot, dim=-1), pred_intent  # Corrected dim to -1 for slots
        else:
            # For top-k predictions
            # pred_slot are logits
            _, slot_indices_tensor = torch.topk(pred_slot, n_predicts, dim=-1)  # Corrected dim to -1 for slots

            # THE OLD INTENT VOTING CODE BELOW IS REMOVED:
            # intent_index_sum = torch.cat(
            #     [
            #         torch.sum(torch.sigmoid(pred_intent[i, 0:seq_lens[i], :]) > self.__args.threshold, dim=0).unsqueeze(
            #             0)
            #         for i in range(len(seq_lens))
            #     ],
            #     dim=0
            # )
            # # seq_lens_tensor needs to be defined if this old code were to be kept
            # # seq_lens_tensor = torch.tensor(seq_lens, device=pred_intent.device)
            # intent_index = (intent_index_sum > (seq_lens_tensor // 2).unsqueeze(1)).nonzero()

            # We ALREADY have `intent_index` computed by ISDAC earlier in the forward pass.
            # We just need to return it in the specified format.

            # The `intent_index` from ISDAC is already an (N, 2) tensor where N is the
            # total number of (batch_idx, intent_idx) pairs detected across the batch.
            # The original code returns intent_index.cpu().data.numpy().tolist()

            return slot_indices_tensor.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, args, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        """

        super(LSTMDecoder, self).__init__()
        self.__args = args
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)

        self.__slot_graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate, self.__args.alpha, self.__args.n_heads,
            self.__args.n_layers_decoder_global)

        self.__global_graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate, self.__args.alpha, self.__args.n_heads,
            self.__args.n_layers_decoder_global)

        self.__linear_layer = nn.Sequential(
            nn.Linear(self.__hidden_dim,
                      self.__hidden_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__hidden_dim, self.__output_dim),
        )

    def forward(self, encoded_hiddens, seq_lens, global_adj=None, slot_adj=None, intent_embedding=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :return: is distribution of prediction labels.
        """

        input_tensor = encoded_hiddens
        output_tensor_list, sent_start_pos = [], 0

        batch = len(seq_lens)
        slot_graph_out = self.__slot_graph(encoded_hiddens, slot_adj)
        intent_in = intent_embedding.unsqueeze(0).repeat(batch, 1, 1)
        global_graph_in = torch.cat([intent_in, slot_graph_out], dim=1)
        global_graph_out = self.__global_graph(global_graph_in, global_adj)
        out = self.__linear_layer(global_graph_out)
        num_intent = intent_embedding.size(0)
        for i in range(0, len(seq_lens)):
            output_tensor_list.append(out[i, num_intent:num_intent + seq_lens[i]])
        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


@njit
def calculate_iou_numba_jit(seg1_start: int, seg1_end: int, seg2_start: int, seg2_end: int) -> float:
    """
    Numba JIT compiled function to calculate Intersection over Union (IoU).
    Accepts basic integer types.
    """
    intersect_start = max(seg1_start, seg2_start)
    intersect_end = min(seg1_end, seg2_end)

    intersect_len = max(0, intersect_end - intersect_start + 1)
    if intersect_len == 0:
        return 0.0

    len1 = seg1_end - seg1_start + 1
    len2 = seg2_end - seg2_start + 1
    union_len = len1 + len2 - intersect_len

    if union_len == 0:
        return 0.0 if intersect_len == 0 else 1.0

    return intersect_len / union_len


@njit
def sum_log_conf_for_segment_numba_jit(log_probs_1d_np_array: np.ndarray,
                                       start_idx: int,
                                       end_idx: int) -> float:
    """
    Numba JIT compiled function to sum log confidences over a segment.
    Accepts a 1D NumPy array and start/end indices.
    """
    current_sum = 0.0
    for i in range(start_idx, end_idx + 1):
        current_sum += log_probs_1d_np_array[i]
    return current_sum


class ISDACProcessor:
    def __init__(self,
                 num_intent_labels: int,
                 tau: float = 0.01,  # This 'tau' is not directly used if log_probs are pre-calculated
                 min_len: int = 2,
                 max_len: int = 10,
                 avg_log_conf_threshold: any = math.log(0.6),
                 overlap_threshold_iou: float = 0.3):

        self.num_intent_labels = num_intent_labels
        # self.tau = tau # tau is applied when creating input log_probs, not here.
        self.min_len = min_len
        self.max_len = max_len

        if isinstance(avg_log_conf_threshold, (float, int)):
            self.avg_log_conf_thresholds_list = [float(avg_log_conf_threshold)] * num_intent_labels
        elif isinstance(avg_log_conf_threshold, (list, tuple)):
            if len(avg_log_conf_threshold) != num_intent_labels:
                raise ValueError(f"Length of avg_log_conf_threshold list ({len(avg_log_conf_threshold)}) "
                                 f"must match num_intent_labels ({num_intent_labels}).")
            self.avg_log_conf_thresholds_list = [float(t) for t in avg_log_conf_threshold]
        else:
            raise TypeError("avg_log_conf_threshold must be a float, int, list, or tuple.")

        self.overlap_threshold_iou = overlap_threshold_iou

    def _get_threshold_for_intent(self, intent_idx: int) -> float:
        if 0 <= intent_idx < self.num_intent_labels:
            return self.avg_log_conf_thresholds_list[intent_idx]
        else:
            raise IndexError(f"Invalid intent_idx {intent_idx} for num_intent_labels {self.num_intent_labels}")

    def _calculate_iou(self, seg1_start: int, seg1_end: int, seg2_start: int, seg2_end: int) -> float:
        """Wrapper method to call the Numba JITted IoU function."""
        return calculate_iou_numba_jit(seg1_start, seg1_end, seg2_start, seg2_end)

    def process_single_sentence(self,
                                token_intent_log_probs_single_cpu_tensor: torch.Tensor,
                                seq_len_single: int
                                ) -> set[int]:
        # Validate input shapes
        if token_intent_log_probs_single_cpu_tensor.shape[0] != seq_len_single:
            raise ValueError(
                f"Shape mismatch on seq_len_single: tensor has {token_intent_log_probs_single_cpu_tensor.shape[0]}, expected {seq_len_single}")
        if token_intent_log_probs_single_cpu_tensor.shape[1] != self.num_intent_labels:
            raise ValueError(
                f"Shape mismatch on num_intent_labels: tensor has {token_intent_log_probs_single_cpu_tensor.shape[1]}, expected {self.num_intent_labels}")

        # Convert the input CPU PyTorch Tensor to a NumPy array ONCE.
        log_probs_numpy_2d = token_intent_log_probs_single_cpu_tensor.numpy()

        candidate_segments = []

        for k in range(self.num_intent_labels):
            intent_threshold = self._get_threshold_for_intent(k)
            log_probs_1d_for_intent_k_np = log_probs_numpy_2d[:, k]  # This is a 1D NumPy array view

            for t in range(seq_len_single):
                s_loop_min = max(0, t - self.max_len + 1)
                s_loop_max = t - self.min_len + 1

                for s in range(s_loop_min, s_loop_max + 1):
                    if s > t:
                        continue
                    current_segment_len = t - s + 1

                    sum_log_conf = sum_log_conf_for_segment_numba_jit(
                        log_probs_1d_for_intent_k_np, s, t
                    )

                    avg_log_conf = sum_log_conf / current_segment_len
                    if avg_log_conf >= intent_threshold:
                        candidate_segments.append((s, t, k, avg_log_conf))

        if candidate_segments:
            filtered_candidates_map = {}
            for s_val, t_val, k_val, score_val in candidate_segments:
                segment_key = (s_val, t_val)
                if segment_key not in filtered_candidates_map or score_val > filtered_candidates_map[segment_key][3]:
                    filtered_candidates_map[segment_key] = (s_val, t_val, k_val, score_val)
            candidate_segments = list(filtered_candidates_map.values())

        final_detected_intents_for_sentence = set()
        if not candidate_segments:
            return final_detected_intents_for_sentence

        candidate_segments.sort(key=lambda x: x[3], reverse=True)
        selected_segments_for_nms = []
        for cand_s, cand_t, cand_k, cand_score in candidate_segments:
            is_overlapping = False
            for sel_s, sel_t, _, _ in selected_segments_for_nms:
                iou = self._calculate_iou(cand_s, cand_t, sel_s, sel_t)  # Calls Numba JITted version
                if iou > self.overlap_threshold_iou:
                    is_overlapping = True;
                    break
            if not is_overlapping:
                selected_segments_for_nms.append((cand_s, cand_t, cand_k, cand_score))
                final_detected_intents_for_sentence.add(cand_k)
        return final_detected_intents_for_sentence