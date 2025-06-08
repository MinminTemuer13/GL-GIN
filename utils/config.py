# -*- coding: utf-8 -*-#
import argparse
import torch

parser = argparse.ArgumentParser(description='Configuration for Hierarchical Differential SLU Model')

# --- Dataset and Other General Parameters ---
parser.add_argument('--data_dir', '-dd', help='Dataset file path', type=str, default='./data/MixSNIPS_clean')
parser.add_argument('--save_dir', '-sd', help='Directory to save models', type=str, default='./save/MixSNIPS_HDA_GCN')
parser.add_argument('--load_dir', '-ld', help='Directory to load models from', type=str, default=None)
parser.add_argument('--log_dir', '-lod', help='Directory to save logs', type=str, default='./log/MixSNIPS_HDA_GCN')
parser.add_argument('--log_name', '-ln', help='Log file name', type=str, default='training_log.txt')
parser.add_argument("--random_state", '-rs', help='Random seed', type=int, default=72)
parser.add_argument('--gpu', '-g', action='store_true', help='Use GPU if available', default=torch.cuda.is_available())
parser.add_argument('--padding_idx', help='Index of the padding token', type=int, default=0)



# --- Training Parameters ---
parser.add_argument('--num_epoch', '-ne', help='Number of training epochs', type=int, default=50)
parser.add_argument('--batch_size', '-bs', help='Batch size for training', type=int, default=16)
parser.add_argument('--l2_penalty', '-l2p', help='L2 regularization penalty', type=float, default=1e-5)
parser.add_argument("--learning_rate", '-lr', help='Initial learning rate', type=float, default=0.0005) # 0.0001


parser.add_argument('--early_stop', action='store_true', help='Enable early stopping', default=True) # 建议启用
parser.add_argument('--patience', '-pa', help='Patience for early stopping', type=int, default=10)
parser.add_argument('--slot_padding_token_id', help='ID for padding slot tokens in prediction output', type=int, default=0)



parser.add_argument('--slot_loss_alpha', '-sla', help='Weight for slot loss', type=float, default=0.9)
parser.add_argument('--intent_loss_alpha', '-ila', help='Weight for intent loss', type=float, default=0.1)

parser.add_argument('--intent_threshold', '-ithr', help='Threshold for binarizing intent probabilities', type=float, default=0.7)
parser.add_argument('--d_ff_multiplier', '-dffm', help='Multiplier for d_model to get FFN intermediate dim',
                    type=int, default=2) # SwiGLU, 论文中FFN size = 8/3 * d_model

# --- Hierarchical Differential Encoder (HDA) Parameters ---
parser.add_argument('--num_encoder_layers', '-nel', help='Number of HDA encoder layers',
                    type=int, default=10)
parser.add_argument('--d_model', '-dm', help='Core model dimension (word embedding, encoder hidden)',
                    type=int, default=256)
parser.add_argument('--num_attention_heads', '-nah', help='Number of attention heads in HDA',
                    type=int, default=4)
parser.add_argument('--d_k_neighbor_divisor', '-dknd', help='Divisor for d_model to get d_k_neighbor',
                    type=int, default=16) # 设为num_attention_heads, int(math.sqrt(d_model))
parser.add_argument('--max_seq_len', '-msl', help='Maximum sequence length for RoPE', type=int, default=80)
parser.add_argument('--rope_theta', '-rt', help='Theta parameter for RoPE', type=float, default=10000.0)
parser.add_argument('--dropout_rate_encoder', '-dre', help='Dropout rate for HDA encoder components',
                    type=float, default=0.2)



# --- Task-Specific Feature Transformation (after HDA) Parameters ---
parser.add_argument('--d_intent_specific_hidden_dim', '-dishd', help='Dimension of intent-specific hidden states from HDA',
                    type=int, default=256)
parser.add_argument('--d_slot_specific_hidden_dim', '-dsshd', help='Dimension of slot-specific hidden states from HDA',
                    type=int, default=512)



# --- GCN Interaction Module Parameters ---
parser.add_argument('--use_gcn_interaction', '-ugcn', action='store_true', help='Use GCN interaction module after HDA', default=True)
parser.add_argument('--num_gcn_layers', '-ngl', help='Number of GCN layers',
                    type=int, default=4)
parser.add_argument('--gcn_epsilon', '-geps', help='Epsilon for GCN normalization', type=float, default=1e-12)
parser.add_argument('--gcn_activation_fn_str', '-gafs', help='Activation function for GCN (relu, gelu, tanh)', type=str, default="relu")
parser.add_argument('--gcn_affinity_power', '-gap', help='Power for affinity matrix in GCN to enhance contrast',
                    type=float, default=1)



# --- Decoder Parameters ---
parser.add_argument('--dropout_rate_intent_decoder', '-drid', help='Dropout rate for intent decoder',
                    type=float, default=0.2)
parser.add_argument('--dropout_rate_slot_decoder', '-drsd', help='Dropout rate for slot decoder',
                    type=float, default=0.2)
parser.add_argument('--use_crf', '-uc', help='Using CRF as loss function or not',
                    type=bool, default=True)


args = parser.parse_args()


# Ensure GPU is used if specified and available
if args.gpu and not torch.cuda.is_available():
    print("Warning: GPU was requested but is not available. Using CPU instead.")
    args.gpu = False
elif not args.gpu and torch.cuda.is_available():
    print("Information: GPU is available but not requested. Using CPU. Add -g or --gpu to use GPU.")


print("--- Parsed Configuration ---")
for arg_name, value in sorted(vars(args).items()):
    print(f"  {arg_name}: {value}")
print("--------------------------")