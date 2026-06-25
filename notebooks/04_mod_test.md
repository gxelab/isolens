## RNA modification analysis

```bash
# example 2 test
uv run python -m isolens.mod_scan -b examples/example2.txmap.bam -a examples/example2.lz4 -o examples/tmp_example2.isolens.mod_scan.h5 -c 0.95 -v -t 2
uv run python -m isolens.mod_sites -i examples/tmp_example2.isolens.mod_scan.h5 -o examples/tmp_example2.isolens.mod_sites.tsv.gz -f tsv -z -v

uv run python -m isolens.mod_corr -i examples/example2.isolens.mod_scan.h5 -s examples/example2.isolens.mod_sites.tsv.gz -o examples/example2.isolens.mod_corr.tsv.gz -f tsv -z -v -P tmp


# example 4 prepare
uv run python scripts/asp_extract.py -i examples/example3.lz4 -o examples/example4.lz4 -p 0.9 -n 1000 -t FBtr0076934 --compress
uv run python scripts/bam_filter.py -i examples/example3.txmap.bam -a examples/example4.lz4 -o examples/example4.txmap.bam
samtools index examples/example4.txmap.bam

# example 4 test
uv run python -m isolens.mod_scan -b examples/example4.txmap.bam -a examples/example4.lz4 -o examples/example4.isolens.mod_scan.h5 -c 0.95 -v
uv run python -m isolens.mod_sites -i examples/example4.isolens.mod_scan.h5 -o examples/example4.isolens.mod_sites.tsv.gz -f tsv -z -v

uv run python -m isolens.mod_corr -i examples/example4.isolens.mod_scan.h5 -s examples/example4.isolens.mod_sites.tsv.gz -o examples/example4.isolens.mod_corr.tsv.gz -f tsv -z -v -P tmp

# compare with modkit
~/miniforge3/envs/lrs/bin/modkit pileup examples/example4.txmap.bam examples/example4.modkit.pileup.tsv \
    --modified-bases m6A inosine m5C pseU 2OmeA 2OmeC 2OmeG 2OmeU \
    --filter-threshold 0.95 --reference examples/final_isoforms.fa
```
