# Information about file format

## Oarfish read assignment probabilities

https://github.com/COMBINE-lab/oarfish#read-level-assignment-probabilities


## bam
CIGAR

| Operator | Description | Consumes Read | Consumes Reference |
| :--- | :--- | :---: | :---: |
| M | Match or mismatch (alignment of a base to a base) | Yes | Yes |
| I | Insertion (bases in read not in reference) | Yes | No |
| D | Deletion (bases in reference not in read) | No | Yes |
| N | Skipped region (e.g., intron) | No | Yes |
| S | Soft clipping (bases in read not aligned) | Yes | No |
| H | Hard clipping (bases removed from read) | No | No |
| P | Padding (silent positions in multiple alignment) | No | No |
| = | Exact match (read base equals reference base) | Yes | Yes |
| X | Mismatch (read base differs from reference base) | Yes | Yes |

tags:
explained in `/docs/SAMtags.pdf` in the project folder.