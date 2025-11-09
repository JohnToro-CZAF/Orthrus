import genome_kit as gk

# l = ['Rnor_6.0.88', 'ensembl.Macfas_5.0.95', 'ensembl.Rnor_6.0.88', 'ensembl.Sscrofa11.1.98', 'gencode.v19', 'gencode.v25', 'gencode.v25.basic', 'gencode.v25lift37', 'gencode.v25lift37.basic', 'gencode.v26', 'gencode.v26.basic', 'gencode.v26lift37', 'gencode.v26lift37.basic', 'gencode.v27', 'gencode.v27.basic', 'gencode.v27lift37', 'gencode.v27lift37.basic', 'gencode.v29', 'gencode.v29.basic', 'gencode.v29lift37', 'gencode.v29lift37.basic', 'gencode.v30', 'gencode.v30.basic', 'gencode.v30lift37', 'gencode.v30lift37.basic', 'gencode.v41', 'gencode.v41.basic', 'gencode.v41lift37', 'gencode.v41lift37.basic', 'gencode.v46', 'gencode.v46.basic', 'gencode.v46lift37', 'gencode.v46lift37.basic', 'gencode.v47', 'gencode.v47.basic', 'gencode.v47lift37', 'gencode.v47lift37.basic', 'gencode.vM15', 'gencode.vM15.basic', 'gencode.vM19', 'gencode.vM19.basic', 'gencode.vM30', 'gencode.vM30.basic', 'gencode.vM31', 'gencode.vM31.basic', 'gencode.vM36', 'gencode.vM36.basic', 'hg19', 'hg19.p13.plusMT', 'hg38', 'hg38.p10', 'hg38.p12', 'hg38.p13', 'hg38.p14', 'hg38.p7', 'macFas5', 'mm10', 'mm10.p4', 'mm10.p5', 'mm10.p6', 'mm39', 'ncbi_refseq.Macfas_5.0.v101', 'ncbi_refseq.m38.v106', 'ncbi_refseq.m39.v109', 'ncbi_refseq.v105.20190906', 'ncbi_refseq.v108', 'ncbi_refseq.v109', 'ncbi_refseq.v110', 'rn6', 'susScr11', 'ucsc_refseq.2017-06-25']

l = gk.gk_data.data_manager.list_available_genomes()
print(f"Available genomes: {l}")

valid_annotation_genomes = []
mapping_valid_genomes_to_example_gene = {}
for g in l:
    genome = gk.Genome(g) # downloading the genome data
    try:
        print(f"Genome {g} has {len(genome.genes)} genes")
        valid_annotation_genomes.append(g)
        mapping_valid_genomes_to_example_gene[g] = [(genome.genes[0].name, genome.genes[0].id), (genome.genes[1].name, genome.genes[1].id), (genome.genes[2].name, genome.genes[2].id), (genome.genes[3].name, genome.genes[3].id)]
    except Exception as e:
        print(f"Error downloading genome {g}: {e}")

print(f"Valid annotation genomes: {valid_annotation_genomes}")
print(f"Mapping valid genomes to example gene: {mapping_valid_genomes_to_example_gene}")