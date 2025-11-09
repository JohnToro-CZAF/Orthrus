import torch
import argparse
import os

from orthrus.eval_utils import load_model
from orthrus.gk_utils import seq_to_oh


def main():
    parser = argparse.ArgumentParser(description='Generate RNA sequence embeddings using Orthrus model')
    parser.add_argument('--sequence', type=str, required=True, help='RNA sequence to encode')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory to save representations')
    parser.add_argument('--checkpoint', type=str, default='orthrus_v1_4_track/epoch=6-step=20000.ckpt', 
                        help='Path to model checkpoint') # 
# orthrus_v0_small_4_track/epoch=18-step=20000.ckpt
# orthrus_v1_4_track/epoch=6-step=20000.ckpt

# orthrus_v0_6_track/epoch=22-step=20000.ckpt
# orthrus_v1_6_track/epoch=6-step=20000.ckpt
# orthrus_v1_small_6_track/epoch=6-step=20000.ckpt
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sequence
    seq = args.sequence
    
    # One-hot encode sequence
    one_hot = seq_to_oh(seq)
    
    # Transpose to get (4, seq_len) shape
    one_hot = one_hot.T
    
    # Convert to torch tensor
    torch_one_hot = torch.tensor(one_hot, dtype=torch.float32)
    
    # Add batch dimension
    torch_one_hot = torch_one_hot.unsqueeze(0)
    
    print(f"Input shape: {torch_one_hot.shape}")
    
    # Move to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_one_hot = torch_one_hot.to(device=device)
    
    # Create lengths tensor
    lengths = torch.tensor([torch_one_hot.shape[2]]).to(device=device)
    
    # Load model
    model_dir, checkpoint_name = args.checkpoint.split('/')
    model = load_model(f"./models/{model_dir}", checkpoint_name=checkpoint_name)
    model = model.to(torch.device(device))
    
    print("Model loaded successfully")
    print(model)
    
    # Generate representations
    with torch.no_grad():
        reps = model.representation(torch_one_hot, lengths)
    
    print(f"Representations shape: {reps.shape}")
    
    # Save representations
    output_path = os.path.join(args.output_dir, "representations.pt")
    torch.save(reps.cpu(), output_path)
    print(f"Representations saved to {output_path}")


if __name__ == "__main__":
    main()
