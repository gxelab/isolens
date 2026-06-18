## RNA modification analysis
## dev with sample
```bash
samtools sort -@8 -o drs01_r2.txmap.sorted.bam drs01_r2.txmap.bam
samtools index drs01_r2.txmap.sorted.bam
samtools view -b drs01_r2.txmap.sorted.bam FBtr0073078 FBtr0073079 >example.txmap.bam

python -m isolens.mod_scan -b example.txmap.bam -a example.lz4 -o example.isolens.mod_scan.h5 -c 0.95 -v

samtools view -b drs01_r2.txmap.sorted.bam FBtr0076952 FBtr0345025 FBtr0076950 FBtr0076951 FBtr0073748 FBtr0479993 FBtr0340303 FBtr0330396 FBtr0479992 FBtr0330397 FBtr0073747 FBtr0071460 FBtr0071461 FBtr0332953 FBtr0307488 FBtr0081392 FBtr0305586 FBtr0305587 FBtr0305584 FBtr0305585 >example2.txmap.bam

python asp_extract.py -i drs01_r2.prob.lz4 -t "FBtr0076952,FBtr0345025,FBtr0076950,FBtr0076951,FBtr0073748,FBtr0479993,FBtr0340303,FBtr0330396,FBtr0479992,FBtr0330397,FBtr0073747,FBtr0071460,FBtr0071461,FBtr0332953,FBtr0307488,FBtr0081392,FBtr0305586,FBtr0305587,FBtr0305584,FBtr0305585" -o example2.lz4 --compress
```
