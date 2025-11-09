import torch
import argparse
import os
import numpy as np

from orthrus.eval_utils import load_model


def seq_to_oh(seq):
    """One-hot encode DNA/RNA sequence.
    
    Args:
        seq (str): DNA/RNA sequence
        
    Returns:
        np.ndarray: One-hot encoded sequence of shape (L, 4)
    """
    oh = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq.upper()):
        if base in ['A', 'a']:
            oh[i, 0] = 1
        elif base in ['C', 'c']:
            oh[i, 1] = 1
        elif base in ['G', 'g']:
            oh[i, 2] = 1
        elif base in ['T', 't', 'U', 'u']:
            oh[i, 3] = 1
    return oh


def create_manual_six_track(sequence, cds_positions, splice_positions):
    """Create 6-track encoding manually from sequence and position lists.
    
    Args:
        sequence (str): RNA/DNA sequence
        cds_positions (list): List of CDS positions (0-indexed)
        splice_positions (list): List of splice site positions (0-indexed)
        
    Returns:
        np.ndarray: 6-track encoding of shape (6, L)
    """
    # One-hot encode sequence: (L, 4)
    oh = seq_to_oh(sequence)
    L = len(sequence)
    
    # Create CDS track: (L,)
    cds_track = np.zeros(L, dtype=int)
    if cds_positions:
        for pos in cds_positions:
            if 0 <= pos < L:
                cds_track[pos] = 1
    
    # Create splice track: (L,)
    splice_track = np.zeros(L, dtype=int)
    if splice_positions:
        for pos in splice_positions:
            if 0 <= pos < L:
                splice_track[pos] = 1
    
    # Transpose one-hot from (L, 4) to (4, L)
    oh = oh.T
    
    # Reshape tracks from (L,) to (1, L)
    cds_track = cds_track[None, :]
    splice_track = splice_track[None, :]
    
    # Concatenate to get (6, L)
    six_track = np.concatenate([oh, cds_track, splice_track], axis=0)
    
    return six_track


def create_transcript_six_track(genome, gene_name):
    """Create 6-track encoding from GenomeKit transcript.
    
    Args:
        genome: GenomeKit Genome object
        gene_name (str): Gene name to look up
        
    Returns:
        np.ndarray: 6-track encoding of shape (6, L)
    """
    from orthrus.gk_utils import find_transcript_by_gene_name, create_six_track_encoding
    
    transcripts = find_transcript_by_gene_name(genome, gene_name)
    if not transcripts:
        raise ValueError(f"No transcripts found for gene {gene_name}")
    
    t = transcripts[0]
    from genome_kit import Interval
    print(f"Using transcript: {t.id}")
    print(f"Sequence: {genome.dna(Interval(t.chromosome, t.strand, t.start, t.end, genome))}")
    
    # create_six_track_encoding returns (6, L) when channels_last=False
    six_track = create_six_track_encoding(t, genome=genome, channels_last=False)
    print(f"6-track encoding shape: {six_track.shape}")
    return six_track

def main():
    parser = argparse.ArgumentParser(
        description='Generate RNA sequence embeddings using Orthrus 6-track model'
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['manual', 'transcript_based'],
                        help='Input mode: manual (provide sequence+positions) or transcript_based (use GenomeKit)')
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory to save representations')
    parser.add_argument('--checkpoint', type=str, default='orthrus_v1_6_track/epoch=6-step=20000.ckpt',
                        help='Path to model checkpoint (format: model_dir/checkpoint_name)')
    
    # Transcript-based mode arguments
    parser.add_argument('--genome', type=str, default='gencode.v29',
                        help='Genome version (gencode.v29, hg38.p12, ncbi_refseq.v109, mm39, gencode.VM31, ncbi_refseq.m39.v109, rn6, Rnor_6.0.88)')
    parser.add_argument('--gene_name', type=str,
                        help='Gene name for transcript_based mode')
    
    # Manual mode arguments
    parser.add_argument('--sequence', type=str,
                        help='RNA/DNA sequence for manual mode')
    parser.add_argument('--cds_positions', type=str,
                        help='Comma-separated CDS positions (0-indexed), e.g., "0,3,6,9"')
    parser.add_argument('--splice_positions', type=str,
                        help='Comma-separated splice site positions (0-indexed), e.g., "99,199"')
    
    args = parser.parse_args()
    
    # Validate mode-specific arguments
    if args.mode == 'transcript_based':
        if not args.gene_name:
            parser.error("--gene_name is required for transcript_based mode")
    elif args.mode == 'manual':
        if not args.sequence:
            parser.error("--sequence is required for manual mode")
        if not args.cds_positions:
            parser.error("--cds_positions is required for manual mode")
        if not args.splice_positions:
            parser.error("--splice_positions is required for manual mode")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate 6-track encoding based on mode
    if args.mode == 'transcript_based':
        print(f"Loading genome: {args.genome}")
        from genome_kit import Genome
        genome = Genome(args.genome)
        
        print(f"Creating 6-track encoding for gene: {args.gene_name}")
        six_track = create_transcript_six_track(genome, args.gene_name)
        
    else:  # manual mode
        # Parse position lists
        cds_positions = [int(x.strip()) for x in args.cds_positions.split(',')]
        splice_positions = [int(x.strip()) for x in args.splice_positions.split(',')]
        
        print(f"Creating 6-track encoding for sequence of length {len(args.sequence)}")
        print(f"CDS positions: {cds_positions}")
        print(f"Splice positions: {splice_positions}")
        
        six_track = create_manual_six_track(args.sequence, cds_positions, splice_positions)
    
    # Convert to torch tensor
    torch_six_track = torch.tensor(six_track, dtype=torch.float32)
    
    # Add batch dimension: (6, L) -> (1, 6, L)
    torch_six_track = torch_six_track.unsqueeze(0)
    
    print(f"Input shape: {torch_six_track.shape}")
    
    # Move to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_six_track = torch_six_track.to(device=device)
    
    # Create lengths tensor
    lengths = torch.tensor([torch_six_track.shape[2]]).to(device=device)
    
    # Load model
    model_dir, checkpoint_name = args.checkpoint.split('/')
    model = load_model(f"./models/{model_dir}", checkpoint_name=checkpoint_name)
    model = model.to(torch.device(device))
    
    print("Model loaded successfully")
    print(model)
    
    # Generate representations
    with torch.no_grad():
        reps = model.representation(torch_six_track, lengths)
    
    print(f"Representations shape: {reps.shape}")
    
    # Save representations
    output_path = os.path.join(args.output_dir, "representations.pt")
    torch.save(reps.cpu(), output_path)
    print(f"Representations saved to {output_path}")


if __name__ == "__main__":
    main()
