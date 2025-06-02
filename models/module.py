# -*- coding: utf-8 -*-#

import math
import torch
import torch.nn as nn

from models.GCNModule import GCNInteractionModule
from models.HDAEncoder import HierarchicalDiffEncoderWithRoPE, build_hierarchical_attention_mask
from models.SVIDecoder import SoftVotingIntentDecoder
from models.slotDecoder import SlotDecoder


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args  # 存储args


        # d_k_neighbor calculation logic remains, but inputs come from self.__args
        d_k_neighbor_val = self.__args.d_model // self.__args.d_k_neighbor_divisor \
            if self.__args.d_k_neighbor_divisor > 0 \
            else int(math.sqrt(self.__args.d_model))

        d_ff_val = self.__args.d_model * self.__args.d_ff_multiplier

        self.__HDA_encoder = HierarchicalDiffEncoderWithRoPE(
            num_encoder_layers=self.__args.num_encoder_layers,
            d_model=self.__args.d_model,
            n_heads=self.__args.num_attention_heads,  # Note: Renamed in config, using new name
            d_k_neighbor=int(d_k_neighbor_val) if d_k_neighbor_val > 0 else None,
            d_ff=d_ff_val,
            input_vocab_size=self.__num_word,
            max_seq_len=self.__args.max_seq_len,
            d_intent_hidden_dim=self.__args.d_intent_specific_hidden_dim,
            d_slot_hidden_dim=self.__args.d_slot_specific_hidden_dim,
            rope_theta=self.__args.rope_theta,
            dropout_rate=self.__args.dropout_rate_encoder,  # Using specific encoder dropout
            padding_idx=self.__args.padding_idx,
        )

        # --- GCN Interaction Module (if启用) from self.__args ---
        self.use_gcn = self.__args.use_gcn_interaction  # Assuming this is in args
        if self.use_gcn:
            self.__GCN_interaction_module = GCNInteractionModule(
                d_intent=self.__args.d_intent_specific_hidden_dim,
                d_slot=self.__args.d_slot_specific_hidden_dim,
                num_gcn_layers=self.__args.num_gcn_layers,
                gcn_epsilon=self.__args.gcn_epsilon,
                gcn_activation_fn_str=self.__args.gcn_activation_fn_str,
                gcn_affinity_power=self.__args.gcn_affinity_power,
            )
        else:
            self.__GCN_interaction_module = None

        # --- Decoders Parameters from self.__args ---
        self.__intent_decoder = SoftVotingIntentDecoder(
            hidden_dim=self.__args.d_intent_specific_hidden_dim,
            num_intents=self.__num_intent,
            dropout_rate=self.__args.dropout_rate_intent_decoder
        )

        self.__slot_decoder = SlotDecoder(
            hidden_dim=self.__args.d_slot_specific_hidden_dim,
            num_slot_labels=self.__num_slot,
            dropout_rate=self.__args.dropout_rate_slot_decoder
        )

    def show_summary(self):
        """
        print the abstract of the defined model.
        """
        print('Model parameters are listed as follows:\n')
        print(f'\tNumber of word:                            {self.__num_word}')
        print(f'\tNumber of slot:                            {self.__num_slot}')
        print(f'\tNumber of intent:                          {self.__num_intent}')

        # HDA Encoder Params
        hda_config = self.__HDA_encoder
        print(f'\tEncoder Type:                              HierarchicalDiffEncoderWithRoPE')
        print(f'\t  Encoder Layers:                          {len(hda_config.layers)}')
        print(f'\t  Model Dimension (d_model):               {hda_config.d_model}')
        print(f'\t  Number of Attention Heads:               {hda_config.n_heads}')
        print(f'\t  Feed-Forward Dimension (d_ff):           {hda_config.layers[0].ffn.w_1.out_features if hda_config.layers else "N/A"}')
        print(f'\t  Max Sequence Length (for RoPE):          {hda_config.rope_emb.cos_cached.shape[0]}')
        print(f'\t  RoPE Theta:                              {hda_config.rope_emb.freqs_cis.real.exp().pow(-hda_config.rope_emb.dim / 2).mean().item() if hasattr(hda_config.rope_emb, "freqs_cis") else "N/A"}') # 估算theta
        print(f'\t  Dropout Rate (Encoder):                  {hda_config.embedding_dropout.p}')
        print(f'\t  Padding Index:                           {hda_config.padding_idx}')
        print(f'\t  d_k_neighbor (Affinity Calc):            {hda_config.d_k_neighbor if hda_config.d_k_neighbor is not None else "Disabled"}')
        if hda_config.task_feature_transformer:
            print(f'\t  Intent Specific Hidden Dim (HDA out):  {hda_config.task_feature_transformer.d_intent_hidden_dim}')
            print(f'\t  Slot Specific Hidden Dim (HDA out):    {hda_config.task_feature_transformer.d_slot_hidden_dim}')
        else:
            print(f'\t  Intent Specific Hidden Dim (HDA out):  N/A (Direct from d_model)')
            print(f'\t  Slot Specific Hidden Dim (HDA out):    N/A (Direct from d_model)')


        # GCN Module Params
        if self.__GCN_interaction_module:
            gcn_config = self.__GCN_interaction_module
            print(f'\tGCN Interaction Module:                  Enabled')
            print(f'\t  GCN Layers:                            {len(gcn_config.gcn_layers)}')
            print(f'\t  GCN Activation:                        {gcn_config.gcn_layers[0].activation_fn.__name__ if gcn_config.gcn_layers else "N/A"}')
            print(f'\t  GCN Affinity Power:                    {gcn_config.gcn_layers[0].affinity_power if gcn_config.gcn_layers else "N/A"}')
        else:
            print(f'\tGCN Interaction Module:                  Disabled')

        # Intent Decoder Params
        intent_dec_config = self.__intent_decoder
        print(f'\tIntent Decoder Type:                       SoftVotingIntentDecoder')
        print(f'\t  Intent Decoder Hidden Dim:               {intent_dec_config.hidden_dim}')
        print(f'\t  Dropout Rate (Intent Decoder):           {intent_dec_config.dropout.p}')

        # Slot Decoder Params
        slot_dec_config = self.__slot_decoder
        print(f'\tSlot Decoder Type:                         SlotDecoder')
        print(f'\t  Slot Decoder Hidden Dim:                 {slot_dec_config.hidden_dim}')
        print(f'\t  Dropout Rate (Slot Decoder):             {slot_dec_config.dropout.p}')

        print('\nEnd of parameters show. Now training begins.\n\n')


    def forward(self, text_token_ids):
        # 1. HDA Encoder
        #    - hda_encoder_outputs 是一个字典，包含:
        #      "encoder_output": (B, S, D_model) - 最终的共享编码器输出 (可能不需要直接使用)
        #      "intent_specific_hidden_states": (B, S, D_intent_specific) 或 (B, D_intent_specific) 如果池化
        #      "slot_specific_hidden_states": (B, S, D_slot_specific)
        #      "final_affinity_scores_a": (B, S-1) - 最后一层的 'a' 分数，用于GCN
        #      "all_layer_attention_weights": 列表，包含每层的 (attn_w1, attn_w2)
        #      "source_padding_mask": (B, S) - True为非padding
        hda_encoder_outputs = self.__HDA_encoder(text_token_ids)

        h_intent_for_decoder = hda_encoder_outputs["intent_specific_hidden_states"]
        h_slot_for_decoder = hda_encoder_outputs["slot_specific_hidden_states"]
        final_affinity_scores_a = hda_encoder_outputs["final_affinity_scores_a"]
        padding_mask_bool = hda_encoder_outputs["source_padding_mask"] # (B,S) True for non-padding

        # 2. Affinity-based GCN Interaction Module (如果启用)
        if self.__GCN_interaction_module and final_affinity_scores_a is not None:
            # GCN期望的affinity_matrix是 (B, S, S)
            # 我们从HDA得到的是 (B, S-1) 的 a_k,k+1
            # 需要从 final_affinity_scores_a 构建完整的亲和力矩阵 C
            # 注意: build_hierarchical_attention_mask 返回的是 [B, 1, S, S]
            # GCN 需要的是 [B, S, S]
            affinity_matrix_C_for_gcn = build_hierarchical_attention_mask(
                final_affinity_scores_a, text_token_ids.size(1), text_token_ids.device
            ).squeeze(1) # 移除head维度

            h_intent_gcn_out, h_slot_gcn_out = self.__GCN_interaction_module(
                h_intent_input=h_intent_for_decoder,
                h_slot_input=h_slot_for_decoder,
                affinity_matrix=affinity_matrix_C_for_gcn,
                padding_mask=padding_mask_bool
            )
            # 更新用于解码器的隐藏状态
            h_intent_for_decoder = h_intent_gcn_out
            h_slot_for_decoder = h_slot_gcn_out
        elif self.__GCN_interaction_module and final_affinity_scores_a is None:
            # 如果启用了GCN但没有亲和力分数 (例如，序列太短或HDA的d_k_neighbor为None)
            # 可以选择跳过GCN，或者使用一个默认的全连接/单位亲和力矩阵
            print("警告: GCN模块已启用，但最终亲和力分数为None。将跳过GCN交互。")
            pass # h_intent_for_decoder 和 h_slot_for_decoder 保持不变


        # 3. Intent Decoder
        # SoftVotingIntentDecoder 需要 (B, S, D_intent) 的输入
        # 以及 (B, S, 1) 的padding_mask (1 for non-padding)
        intent_padding_mask_for_decoder = padding_mask_bool.unsqueeze(-1).float()
        intent_logits, intent_probs, _, _ = self.__intent_decoder(
            token_encodings=h_intent_for_decoder,
            padding_mask=intent_padding_mask_for_decoder
        )

        # 4. Slot Decoder
        # SlotDecoder 需要 (B, S, D_slot) 的输入
        slot_logits, slot_probs = self.__slot_decoder(
            token_encodings=h_slot_for_decoder
            # SlotDecoder的padding_mask是可选的，主要用于损失计算或预测提取
        )

        # 返回训练所需的logits
        return intent_logits, slot_logits # 或者根据训练目标返回probs

    @torch.no_grad()
    def predict(self, text_token_ids, seq_lens=None, intent_threshold=0.5):
        """
        进行预测。
        参数:
            text_token_ids (torch.Tensor): 输入的token ID, 形状 [batch_size, seq_len]。
            seq_lens (torch.Tensor, optional): 每个序列的实际长度，当前未使用，但保留接口。
            intent_threshold (float): 用于意图二元化的阈值。
        返回:
            dict: 包含预测结果的字典。
                - "intent_labels": 预测的意图标签 (0或1), 形状 [batch_size, num_intents]。
                - "intent_probs": 预测的意图概率, 形状 [batch_size, num_intents]。
                - "slot_labels": 预测的槽位标签ID, 形状 [batch_size, seq_len]。
                - "slot_probs": 预测的槽位概率, 形状 [batch_size, seq_len, num_slots]。
        """
        self.eval() # 设置为评估模式

        hda_encoder_outputs = self.__HDA_encoder(text_token_ids)
        h_intent_for_decoder = hda_encoder_outputs["intent_specific_hidden_states"]
        h_slot_for_decoder = hda_encoder_outputs["slot_specific_hidden_states"]
        final_affinity_scores_a = hda_encoder_outputs["final_affinity_scores_a"]
        src_padding_mask_bool = hda_encoder_outputs["source_padding_mask"]
        gcn_padding_mask = ~src_padding_mask_bool

        if self.__GCN_interaction_module and final_affinity_scores_a is not None:
            affinity_matrix_C_for_gcn = HierarchicalDiffEncoderWithRoPE.build_hierarchical_attention_mask(
                final_affinity_scores_a, text_token_ids.size(1), text_token_ids.device
            ).squeeze(1)
            h_intent_gcn_out, h_slot_gcn_out = self.__GCN_interaction_module(
                h_intent_input=h_intent_for_decoder,
                h_slot_input=h_slot_for_decoder,
                affinity_matrix=affinity_matrix_C_for_gcn,
                padding_mask=gcn_padding_mask
            )
            h_intent_for_decoder = h_intent_gcn_out
            h_slot_for_decoder = h_slot_gcn_out
        elif self.__GCN_interaction_module and final_affinity_scores_a is None:
            pass

        # Intent Prediction
        intent_padding_mask_for_decoder = src_padding_mask_bool.unsqueeze(-1).float()
        _, intent_probs, _, _ = self.__intent_decoder(
            token_encodings=h_intent_for_decoder,
            padding_mask=intent_padding_mask_for_decoder
        )
        predicted_intent_labels = (intent_probs > intent_threshold).long()

        # Slot Prediction
        _, slot_probs = self.__slot_decoder(
            token_encodings=h_slot_for_decoder
        )
        predicted_slot_labels = torch.argmax(slot_probs, dim=-1) # (B, S)

        # 在padding位置将slot_labels设为padding_idx (例如0)
        # src_padding_mask_bool: True for non-padding, False for padding
        # 我们希望在padding位置的slot label是padding_idx
        # 首先创建一个全是padding_idx的张量
        slot_padding_token_id = getattr(self.__args, 'slot_padding_token_id', self.__HDA_encoder.padding_idx) # 通常与输入padding一致
        padded_slot_labels = torch.full_like(predicted_slot_labels, fill_value=slot_padding_token_id)
        # 只在非padding位置填充预测的slot labels
        padded_slot_labels[src_padding_mask_bool] = predicted_slot_labels[src_padding_mask_bool]


        self.train() # 恢复为训练模式 (如果后续有训练)
        return {
            "intent_labels": predicted_intent_labels,
            "intent_probs": intent_probs,
            "slot_labels": padded_slot_labels, # 返回处理过padding的slot labels
            "slot_probs": slot_probs
        }