#!/usr/bin/env python3
import sys
import argparse
import lz4.frame
import pysam
import uuid
import hashlib

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate transcript isoform-specific poly(A) length using Oarfish assignments and Dorado BAM."
    )
    parser.add_argument("-o", "--oarfish", required=True, help="Oarfish read assignment probability file (.lz4)")
    parser.add_argument("-b", "--bam", required=True, help="Raw reads BAM file containing pt:i tags")
    parser.add_argument("-out", "--output", required=True, help="Output TSV file")
    # >>> CHANGE: Optional gzip argument added <<<
    parser.add_argument("--gzip", action="store_true", help="Compress the output TSV file using gzip")
    return parser.parse_args()

def read_id_to_int(read_id_str):
    """
    Converts a UUID string to a 128-bit integer to save memory.
    """
    try:
        return uuid.UUID(read_id_str).int
    except ValueError:
        return int(hashlib.md5(read_id_str.encode('utf-8')).hexdigest(), 16)

def main():
    args = parse_args()

    print(f"Parsing Oarfish assignments from {args.oarfish}...", file=sys.stderr)
    tx_idx_to_name = {}
    read_assignments = {}
    
    # Step 1: Parse the lz4 compressed assignment prob file
    with lz4.frame.open(args.oarfish, 'rt') as f:
        first_line = f.readline().strip().split()
        if not first_line or len(first_line) < 1:
            print("Error: Oarfish file is empty or malformed.", file=sys.stderr)
            sys.exit(1)
            
        num_targets = int(first_line[0])
        
        # Next num_targets lines are the transcript IDs
        for i in range(num_targets):
            tx_name = f.readline().strip()
            tx_idx_to_name[i] = tx_name
            
        # Remaining lines are the read assignments
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            read_id_int = read_id_to_int(parts[0])
            num_mappings = int(parts[1])
            
            tx_indices = parts[2 : 2 + num_mappings]
            probs = parts[2 + num_mappings : 2 + 2 * num_mappings]
            
            assignments = []
            for tx_idx_str, prob_str in zip(tx_indices, probs):
                assignments.append((int(tx_idx_str), float(prob_str)))
                
            if assignments:
                read_assignments[read_id_int] = assignments

    print(f"Loaded {len(tx_idx_to_name)} transcripts and {len(read_assignments)} reads with assignments.", file=sys.stderr)
    
    if not read_assignments:
        print("0 reads with assignments found. Exiting early without parsing the BAM file.", file=sys.stderr)
        sys.exit(0)
    
    # Step 2: Initialize a dict to store poly(A) information mapped to transcripts
    tx_data = {tx_idx: [] for tx_idx in tx_idx_to_name.keys()}
    
    print(f"Processing BAM file {args.bam}...", file=sys.stderr)
    processed_reads = set()
    reads_scanned = 0
    
    # Step 3: Read BAM file and extract pt:i tags
    with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            reads_scanned += 1
            
            if reads_scanned % 200000 == 0:
                print(f"  ...scanned {reads_scanned} reads from BAM so far...", file=sys.stderr)

            read_id_int = read_id_to_int(read.query_name)
            
            if read_id_int in processed_reads:
                continue
                
            if read_id_int in read_assignments:
                if read.has_tag("pt"):
                    pt_val = read.get_tag("pt")
                    
                    if pt_val > 0:
                        processed_reads.add(read_id_int)
                        
                        for tx_idx, prob in read_assignments[read_id_int]:
                            tx_data[tx_idx].append((prob, pt_val))

    print(f"Finished BAM parsing. Scanned {reads_scanned} total reads.", file=sys.stderr)
    print(f"Successfully extracted poly(A) lengths for {len(processed_reads)} mapped reads.", file=sys.stderr)
    
    # Step 4: Compute metrics and generate output TSV
    # >>> CHANGE: Set up file handling based on compression needs <<<
    output_filename = args.output
    if args.gzip:
        import gzip
        if not output_filename.endswith(".gz"):
            output_filename += ".gz"
        open_func = lambda f: gzip.open(f, "wt", encoding="utf-8")
    else:
        open_func = lambda f: open(f, "w", encoding="utf-8")

    print(f"Writing results to {output_filename}...", file=sys.stderr)
    with open_func(output_filename) as out_f:
        out_f.write("tx_name\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n")
        
        for tx_idx, tx_name in tx_idx_to_name.items():
            data = tx_data.get(tx_idx, [])
            
            if not data:
                continue
                
            probs = [item[0] for item in data]
            pa_lens = [item[1] for item in data]
            
            n_reads = len(data)
            
            sum_prob = sum(probs)
            if sum_prob > 0:
                pa_wlen = sum(p * l for p, l in data) / sum_prob
            else:
                pa_wlen = 0.0
                
            probs_str = ",".join(f"{p:.5g}" for p in probs)
            pa_lens_str = ",".join(str(l) for l in pa_lens)
            
            out_f.write(f"{tx_name}\t{tx_idx}\t{n_reads}\t{pa_wlen:.3f}\t{probs_str}\t{pa_lens_str}\n")

    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
