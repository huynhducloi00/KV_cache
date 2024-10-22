import argparse
def get_parser():
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    # Basic Configs
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--model_size', type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:0')
    
    # Data Collection 
    parser.add_argument('--seq_len', type=int, default=-1) 
    parser.add_argument('--sampling_batch_size', type=int, default=1)
    parser.add_argument('--seqs_to_collect', type=int, default=512)
    parser.add_argument('--half_precision', action='store_true') # QKVO will be collect in half precision if this is toggled
    
    # Sparse Cache 
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--fix_heavy_to_initial_tokens", action='store_true') # for Lambda masking

    # Kernels
    parser.add_argument("--ker_dim", type=int, default=8)
    parser.add_argument("--ker_hid", type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2)

    # Training Parallelism
    parser.add_argument('--from_layer', type=int, default=0)
    parser.add_argument('--to_layer', type=int, default=9999)
    return parser