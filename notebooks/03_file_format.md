# Information about file format

## Oarfish read assignment probabilities

(below is copy-pasted from )
Read-level assignment probabilities
oarfish has the ability to output read-level assignment probabilities. That is, for each input read, what is the probability, conditioned on the final estimate of transcript abundances, that the read was sequenced from each transcript to which it aligned. By default, this information is not recorded (as it's not required, or commonly used, for most standard analyses). To enable this output, you should pass the --write-assignment-probs option to oarfish. Optionally, you may also pass --write-assignment-probs=compressed to write the output to a compressed (lz4) stream --- the default output is to an uncompressed text file.

The format of the read assignment probabilities is as follows --- where all fields below on a single line are \t delimited:

```
<m> <n>
<tname_1>
<tname_2>
...
<tname_m>
<rname_1> <k_1> <tid_11> <tid_21> ... <tid_{k_1}1> <p_11> <p_21> ... <p_{k_1}1>
...
<rname_n> <k_1> <tid_1n> <tid_2n> ... <tid_{k_n}n> <p_1n> <p_2n> ... <p_{k_n}n>
```

Here, `<m>` is the number of transcripts in the reference, `<n>` is the number of mapped reads. The following m lines consist of the names of the transcripts in the order they will be referred to in the file. The following n lines after that provide the actual alignment information and assignment probabilities for each read.

The format of each of these lines is as follows; `<rname> `is the name of the read, `<k>` is the number of alignments for which probabilities are reported (if it is determined an alignment is invalid under the model, it may not be reported). Subsequently, there is a list of k integers, and k floating point numbers. Each of the k integers is the index of some transcript in the list of m transcripts given at the start of the file, and the subsequent list of k floating point numbers are the assignment probabilities of this read to each of the transcripts.

For example:

```
5 3
txpA
txpB
txpC
txpD
txpE
readX 2 0 2 0.4 0.6
readY 3 1 3 4 0.1 0.2 0.7
readZ 1 4 1.0
```

This file provides an example with 5 reference transcripts and 3 mapped reads. The first read (readX) maps to transcripts 0 and 2 (so txpA and txpC) with probabilities 0.4 and 0.6 respectively. The next read (readY) maps to 3 transcripts, txpB, txpD and txpE with probabilities 0.1, 0.2, and 0.7 respectively. Finally, the last read (readZ) maps uniquely to transcript txpE with probability 1.

The compressed output (i.e. what is generated if one passes --write-assignment-probs=compressed) is exactly the same format, except instead of residing in a plain text file, it is written to an lz4 compressed text file. You can either decompress this file first with an lz4 decompressor, or decompress it on-the-fly as you are parsing the file using the lz4 library in your favorite language.

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