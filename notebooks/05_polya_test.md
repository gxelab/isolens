## polyA calc with oarfish assignment probs

```bash
# pwd: /nfs_data/zhangh/mzt_translation/polya
# env: miniforge -> lrs
# using minimap2 genomic mapping
lz4 -c /nfs_data/liumy/05.flytrans/16.dorado_v100/04.quantify/oarfish/drs01_r1.prob >drs01_r1.prob.lz4
lz4 -c /nfs_data/liumy/05.flytrans/16.dorado_v100/04.quantify/oarfish/drs01_r2.prob >drs01_r2.prob.lz4
lz4 -c /nfs_data/liumy/05.flytrans/16.dorado_v100/04.quantify/oarfish/drs45_r1.prob >drs45_r1.prob.lz4
lz4 -c /nfs_data/liumy/05.flytrans/16.dorado_v100/04.quantify/oarfish/drs45_r2.prob >drs45_r2.prob.lz4

python polya_calc.py -o drs01_r1.prob.lz4 -b /nfs_data/liumy/05.flytrans/16.dorado_v100/01.bam/basecall_drs01_r1.bam -out drs01_r1.polya_calc.tsv.gz --gzip
python polya_calc.py -o drs01_r2.prob.lz4 -b /nfs_data/liumy/05.flytrans/16.dorado_v100/01.bam/basecall_drs01_r2.bam -out drs01_r2.polya_calc.tsv.gz --gzip

python polya_calc.py -o drs45_r1.prob.lz4 -b /nfs_data/liumy/05.flytrans/16.dorado_v100/01.bam/basecall_drs45_r1.bam -out drs45_r1.polya_calc.tsv.gz --gzip
python polya_calc.py -o drs45_r2.prob.lz4 -b /nfs_data/liumy/05.flytrans/16.dorado_v100/01.bam/basecall_drs45_r2.bam -out drs45_r2.polya_calc.tsv.gz --gzip

python polya_merge.py -i1 drs01_r1.polya_calc.tsv.gz -i2 drs01_r2.polya_calc.tsv.gz -o drs01_merge.polya_calc.tsv.gz --gzip
python polya_merge.py -i1 drs45_r1.polya_calc.tsv.gz -i2 drs45_r2.polya_calc.tsv.gz -o drs45_merge.polya_calc.tsv.gz --gzip

python polya_t2g.py -i drs01_merge.polya_calc.tsv.gz -m /nfs_data/liumy/05.flytrans/16.dorado_v100/07.analyze/12.supp_tbl/new_genome_annotation.gtf.txt -o drs01_merge.polya_gene.tsv.gz --gzip

python polya_t2g.py -i drs45_merge.polya_calc.tsv.gz -m /nfs_data/liumy/05.flytrans/16.dorado_v100/07.analyze/12.supp_tbl/new_genome_annotation.gtf.txt -o drs45_merge.polya_gene.tsv.gz --gzip
```
