### poly(A) tail length analysis

```bash
uv run python -m isolens.polya_calc -a examples/example.lz4 -b examples/example.txmap.bam -o examples/example.isolens.polya_calc.tsv.gz -z -g examples/final_isoforms.gtf.gz -l

uv run python -m isolens.polya_calc -a examples/example2.lz4 -b examples/example2.txmap.bam -o examples/example2.polya_calc.tsv.gz -z -g examples/final_isoforms.gtf.gz -l
uv run python -m isolens.polya_calc -a examples/example2.lz4 -b examples/example2.txmap.bam -o examples/example2.polya_calc.pq -f parquet -g examples/final_isoforms.gtf.gz -l
python scripts/tsv2pq.py -i examples/example2.polya_calc.tsv.gz -o examples/example2.polya_calc.pq2tsv.pq
```
