#!/usr/bin/env python3
import sys
import argparse
import gzip
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate transcript-level poly(A) length estimates to the gene level."
    )
    parser.add_argument("-i", "--input", required=True, help="Input transcript poly(A) TSV file (gzipped or raw)")
    parser.add_argument("-m", "--map", required=True, help="Mapping file containing 'tx_name' and 'gene_id' columns (gzipped or raw TSV)")
    parser.add_argument("-o", "--output", required=True, help="Output gene-level TSV file path")
    parser.add_argument("--gzip", action="store_true", help="Compress the output TSV file using gzip")
    return parser.parse_args()

def get_open_func(filename):
    """Returns the correct open function based on the file extension suffix."""
    if filename.endswith(".gz"):
        return lambda f: gzip.open(f, "rt", encoding="utf-8")
    return lambda f: open(f, "r", encoding="utf-8")

def load_gene_mapping(map_file):
    """
    Parses the mapping file and builds a dictionary mapping tx_name -> gene_id.
    Handles files with arbitrary column layouts as long as the headers are present.
    """
    print(f"Reading mapping file from {map_file}...", file=sys.stderr)
    tx_to_gene = {}
    
    open_func = get_open_func(map_file)
    with open_func(map_file) as f:
        header = f.readline().strip().split('\t')
        
        if "tx_name" not in header or "gene_id" not in header:
            print("Error: Mapping file must contain both 'tx_name' and 'gene_id' column headers.", file=sys.stderr)
            sys.exit(1)
            
        tx_col = header.index("tx_name")
        gene_col = header.index("gene_id")
        
        for line_num, line in enumerate(f, 2):
            parts = line.strip().split('\t')
            if len(parts) <= max(tx_col, gene_col):
                continue
            
            tx_to_gene[parts[tx_col]] = parts[gene_col]
            
    print(f"Loaded mappings for {len(tx_to_gene)} unique transcripts.", file=sys.stderr)
    return tx_to_gene

def main():
    args = parse_args()
    
    # Step 1: Load transcript-to-gene relationships
    tx_to_gene = load_gene_mapping(args.map)
    
    # Step 2: Read transcript data and pool by gene
    # Format: gene_id -> {'probs': [...], 'pa_lens': [...]}
    gene_pools = defaultdict(lambda: {'probs': [], 'pa_lens': []})
    unmapped_transcripts = set()
    
    print(f"Processing transcript poly(A) lengths from {args.input}...", file=sys.stderr)
    open_input = get_open_func(args.input)
    
    with open_input(args.input) as f:
        header = f.readline().strip().split('\t')
        if "tx_name" not in header or "probs" not in header or "pa_lens" not in header:
            print("Error: Input file must contain 'tx_name', 'probs', and 'pa_lens' headers.", file=sys.stderr)
            sys.exit(1)
            
        tx_col = header.index("tx_name")
        probs_col = header.index("probs")
        lens_col = header.index("pa_lens")
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= max(probs_col, lens_col):
                continue
                
            tx_name = parts[tx_col]
            
            # Identify which gene this transcript corresponds to
            if tx_name in tx_to_gene:
                gene_id = tx_to_gene[tx_name]
                
                # Parse strings of comma-separated values back to numbers
                probs = [float(p) for p in parts[probs_col].split(',')]
                pa_lens = [int(l) for l in parts[lens_col].split(',')]
                
                gene_pools[gene_id]['probs'].extend(probs)
                gene_pools[gene_id]['pa_lens'].extend(pa_lens)
            else:
                unmapped_transcripts.add(tx_name)
                
    if unmapped_transcripts:
        print(f"Warning: Ignored {len(unmapped_transcripts)} transcripts that were missing from the mapping file.", file=sys.stderr)
        
    # Step 3: Compute gene-level statistics and write output
    output_filename = args.output
    if args.gzip:
        if not output_filename.endswith(".gz"):
            output_filename += ".gz"
        open_output = lambda f: gzip.open(f, "wt", encoding="utf-8")
    else:
        open_output = lambda f: open(f, "w", encoding="utf-8")
        
    print(f"Writing gene-level metrics to {output_filename}...", file=sys.stderr)
    
    with open_output(output_filename) as out_f:
        # Write format-specified header (dropped tx_idx as requested)
        out_f.write("gene_id\tn_reads\tpa_wlen\tprobs\tpa_lens\n")
        
        # Sort outputs alphabetically by gene_id
        for gene_id in sorted(gene_pools.keys()):
            probs = gene_pools[gene_id]['probs']
            pa_lens = gene_pools[gene_id]['pa_lens']
            
            n_reads = len(probs)
            
            # Global re-estimation of the weighted average length
            sum_prob = sum(probs)
            if sum_prob > 0:
                pa_wlen = sum(p * l for p, l in zip(probs, pa_lens)) / sum_prob
            else:
                pa_wlen = 0.0
                
            probs_str = ",".join(f"{p:.5g}" for p in probs)
            pa_lens_str = ",".join(str(l) for l in pa_lens)
            
            out_f.write(f"{gene_id}\t{n_reads}\t{pa_wlen:.2f}\t{probs_str}\t{pa_lens_str}\n")
            
    print("Gene-level aggregation completed successfully!", file=sys.stderr)

if __name__ == "__main__":
    main()
