## RNA modification analysis

initial isolens test
``` bash
# using minimap2 transcriptomic mapping
minimap2 -x map-ont -d all_isoforms.mmi final_isoforms.fa.gz

# -T and -y keeps poly(A) tail and modification tags from uBAM to final BAM.
samtools fastq -T MM,ML,pt drs01_r1.bam |  minimap2 --eqx -N 100 -ax map-ont -y -t8 ../data/all_isoforms.mmi - | samtools view -@ 8 -b -o drs01_r1.txmap.bam
samtools fastq -T MM,ML,pt drs01_r2.bam |  minimap2 --eqx -N 100 -ax map-ont -y -t8 ../data/all_isoforms.mmi - | samtools view -@ 8 -b -o drs01_r2.txmap.bam

../isolens/target/release/isolens -b drs01_r2.txmap.bam -p drs01_r2.prob.lz4 -z -v --out1 drs01_r2.isolens.mod.tsv.gz --out2 drs01_r2.isolens.pa.tsv.gz --out-per-base drs01_r2.isolens.modpb.tsv.gz

python isolens.py -b drs01_r1.txmap.bam -p drs01_r1.prob.lz4 --out1 drs01_r1.isolens.mod.tsv.gz --out2 drs01_r1.isolens.pa.tsv.gz --out-per-base drs01_r1.isolens.modpb.tsv.gz
```

## dev with sample
```bash
samtools sort -@8 -o drs01_r2.txmap.sorted.bam drs01_r2.txmap.bam
samtools index drs01_r2.txmap.sorted.bam
samtools view -b drs01_r2.txmap.sorted.bam FBtr0073078 FBtr0073079 >example.txmap.bam

python -m isolens.mod_scan -b example.txmap.bam -p example.lz4 -o example.isolens.mod_scan.h5 -c 0.95 -v

samtools view -b drs01_r2.txmap.sorted.bam FBtr0076952 FBtr0345025 FBtr0076950 FBtr0076951 FBtr0073748 FBtr0479993 FBtr0340303 FBtr0330396 FBtr0479992 FBtr0330397 FBtr0073747 FBtr0071460 FBtr0071461 FBtr0332953 FBtr0307488 FBtr0081392 FBtr0305586 FBtr0305587 FBtr0305584 FBtr0305585 >example2.txmap.bam

python asp_extract.py -i drs01_r2.prob.lz4 -t "FBtr0076952,FBtr0345025,FBtr0076950,FBtr0076951,FBtr0073748,FBtr0479993,FBtr0340303,FBtr0330396,FBtr0479992,FBtr0330397,FBtr0073747,FBtr0071460,FBtr0071461,FBtr0332953,FBtr0307488,FBtr0081392,FBtr0305586,FBtr0305587,FBtr0305584,FBtr0305585" -o example2.lz4 --compress
```
