#!/bin/bash

# This code returns a 2kb region upstream of the promoter region for all genotypes in the "Genomes" folder

# Must load samtools and bedtools on MSI before running
# module load samtools
# module load bedtools


# Fasta files were renamed for ease
#
# mv BrapaO_302V_711_v1.0.fa BrapaO_302V.fa
# mv Brapassp_chinensisvar_communisPCGlu_712_v2.0.fa BrapaPCGlu.fa
# mv Brapassp_chinensisvar_parachinensisL58_709_v1.0.fa BrapaL58.fa
# mv Brapassp_oleiferavar_oleiferaWO_83_715_v1.0.fa BrapaWO_83.fa
# mv Brapassp_pekinensisvar_pekinensisA03_708_v1.0.fa BrapaA03.fa
# mv Brapassp_rapavar_rapaVT123_714_v1.0.fa BrapaVT123.fa
# mv Brapassp_trilocularisR500_795_v2.0.fa BrapaR500.fa
#

cd /home/myersc/jbarreto/CSCI5461_Project
mkdir -p output
mkdir -p output/Regions

# Path to folder with .fa files
fasta_files="Genomes"

# Loop over all .fa files in folder
for fa_file in "$fasta_files"/*.fa; do
    # Extract genotype name
    genotype=$(basename "$fa_file" .fa)
    # echo "fa_file: $fa_file, and Genotype: $genotype"
    # # Create index file
    samtools faidx "$fa_file"
    # # Create chrom.sizes file
    cut -f 1,2 "$fa_file.fai" | awk '{print $1"\t"$2}' > "./output/$genotype.chrom.sizes"
# # Get .gff3 file for genotype
    genotype2=$(echo "$genotype" | sed 's/Brapa//')
    gff3_path=$(find "./$fasta_files" -name "*$genotype2*.gff3" -type f)
    grep "gene" "$gff3_path" > "./output/$genotype2.genes.gff3"
    echo "New gff3 is: ./output/$genotype2.genes.gff3"

    # Move column 3 (name) to 9 (attributes), and vice versa in .gff3
    # This is to print attributes in final output
    # cp "./output/$genotype2.genes.gff3" "./output/$genotype2.backup.genes.gff3"
    awk 'BEGIN {OFS="\t"} {t=$3; $3=$9; $9=t; print}' "./output/$genotype2.genes.gff3" > temp.gff3 && mv temp.gff3 "./output/$genotype2.genes.gff3"

# # Identify promoter regions (2kbp upstream)
    bedtools flank -i "./output/$genotype2.genes.gff3" -g "./output/$genotype.chrom.sizes" -l 2000 -r 0 -s > "./output/$genotype.promoters.bed"
    # # Extract the 2kb sequences
    bedtools getfasta -fi "$fa_file" -bed "./output/$genotype.promoters.bed" -fo "./output/Regions/$genotype.genes.2kb.promoters.bed" -name -tab

done
