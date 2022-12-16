# Inpactor2: LTR retrotransposon detector and classificator using Deep Learning

## Table of Contents  
* [Introduction](#introduction)  
* [Installation](#installation)  
* [Testing](#testing)  
* [Usage](#usage) 
* [Inputs](#input) 
* [Outputs](#output)
* [Inpactor2's cycles of analysis](#cycles) 
* [Computacional resources needed](#Resources)
* [Inpactor2_utils](#inpactor2_utils) 
* * [K-mer counting utility](#inpactor2_utils_kmer) 
* * [CLASSIFY utility](#inpactor2_utils_classify) 
* * [Sequence filter utility](#inpactor2_utils_filter)
* [Neural network architectures](#architectures)
* [For advance GPU users only](#gpu_users)
* [References and related works](#references) 
* [Other useful resources](#resources) 
* [Citation](#citation)

## Introduction
<a name="introduction"/>

Inpactor2 was designed and developed to detect reference LTR retrotransposons (LTR-RTs), filtering out those elements that correspond to fragments or have nested insertions. In addition, Inpactor2 classifies LTR-RTs down to the lineage/family level. Optionally, this tool annotates the elements discovered through RepeatMasker.

Inpactor2 uses neural networks to perform its tasks faster than other software (up to 7 times faster), accurately while maintaining high sensitivity and a low false positive rate.

<p align="center">
  <img src="https://github.com/simonorozcoarias/Inpactor2/blob/main/NN_architectures/simple_Inpactor2_diagram.png">
</p>

Inpactor2 receives as input a genomic assembly and generates a library of semi-curated and classified LTR-RTs (in fasta format). In addition, it generates a tabular file listing the predictions made by each neural network to verify the confidentiality of each detected LTR-RT. If the RepeatMasker option is active, it will generate the same files as a traditional run of this software.

## Installation:
<a name="installation"/>

We highly recommend to use and install Python packages within an Anaconda environment. First, download the lastest version of Inpactor2

```
git clone https://github.com/simonorozcoarias/Inpactor2.git
```
Go to the Inpactor2 folder and find the file named "Inpactor2.yml". Then, install the environment: 
```
conda env create -f Inpactor2/Inpactor2.yml
```

## Testing:
<a name="testing"/>

After successfully installing Inpactor2, you can test it using the testing data contained in this repository. To do so, first you must activate the conda environment:
```
conda activate Inpactor2
```
Then, you must run the following command:
```
python3 Inpactor2.py -f Testing/toy_genome.fasta -o Testing/toy_execution -a no
```
Finally compare your results in the folder Testing/toy_execution with the files in the folder Testing 'Inpactor2_library_successfull_run.fasta' and 'Inpactor2_predictions_successfull_run.tab'. If you obtain similar (or also the same) results, congrats! Inpactor2 is now installed and funcional.

## Usage:
<a name="usage"/>

Activate the anaconda environment:
```
conda activate Inpactor2
```
Then execute Inpactor2 with default parameters:
```
python3 Inpactor2.py -f genome_file.fasta -o outputDir
```
Please note that the unique required parameter is -f (the genome file in fasta format). The outputDir is a folder where Inpactor2 will put some temporal files and also the final results. It is mandatory that outputDir folder exists before running Inpactor2. The complete options are the following:
* -h or --help: show this help message and exit.
* -f FASTA_FILE or --file FASTA_FILE: Fasta file containing DNA sequences **(required)**.
* -o OUTPUTDIR or --output-dir OUTPUTDIR: Path of the output directory. Default: current path.
* -t THREADS or --threads THREADS: Number of threads to be used by Inpactor2. Default: all available threads.
* -a ANNOTATE or, --annotate ANNOTATE: Annotate LTR retrotransposons using RepeatMasker? [yes or no]. Default: yes.
* -m MAX_LEN_THRESHOLD or --max-len MAX_LEN_THRESHOLD: Maximum length for detecting LTR-retrotransposons [1 - 50000]. Default: 15000.
* -n MIN_LEN_THRESHOLD or --min-len MIN_LEN_THRESHOLD: Minimum length for detecting LTR-retrotransposons [1 - 50000]. Default: 1000.
* -i TG_CA or --tg-ca TG_CA: Keep only elements with TG-CA-LTRs? [yes or no]. Default: no.
* -d TSD or --tsd TSD: Keep only elements with TDS? [yes or no]. Default: no.
* -c CURATION or --curated CURATION: keep on only intact elements? [yes or no]. Default: yes.
* -C (upper case) CYCLES or --cycles CYCLES: Number of analysis cycles [1 - 5]. Default: 1.
* -V (upper case) VERBOSE or --verbose VERBOSE: activate verbose? [yes or no]. Default: no.
* --version: show program's version number and exit.

It is highly recommended to create and use an output directory in each execution to avoid the substitution of result files from different runs of Inpactor2.

## Inpactor2's Input
<a name="input"/>

The input required for Inpactor2 is a fasta file containing an assembly of the genome of the species to be analyzed. It is important to mention that the quality of Inpactor2 results depends directly on the input data. If the input genome has a very low N50 (less than 10-20 kb), contains a lot of Ns, or it is very fragmented it is possible that Inpactor2 will not detect LTR-RTs.

Inpactor2 does not work on short sequencing reads. It is required that you first assemble the reads (using your favorite assembly program) and then use Inpactor2 on the final assembly.

## Inpactor2's Output
<a name="output"/>

Inpactor2 produces two main outputs: a library of LTR-retrotransposons called "Inpactor2_library.fasta" and a file with the predictions of each neural network in a tabular format named "Inpactor2_predictions.tab".

### LTR-retrotransposon library
The library will be done in fasta format. Each sequence has a identifier with follows the structure, where RLC and RLG means Copia and Gypsy superfamilies:
```
>ContainingSequence#LTR/RL[C-G]/PredictedLineage
```
### Predictions file
Additionally, The software writes in a file the probabilities obtained by each neural network (Inpactor2_Detect, Inpactor2_Filter and Inpactor2_Class) separated by tabulations. An example of the predictions file will be seen following:

```
Chr1	3780765	3785720	4955	RLC/IVANA/ORYCO	0.99999654	0.99999964	0.99986255
Chr1	7717356	7722547	5191	RLC/IVANA/ORYCO	0.99971956	0.9999194	0.9999558
Chr1	11789452	11794077	4625	RLC/IVANA/ORYCO	0.9967033	0.99992347	0.99087137
Chr1	13334465	13339869	5404	RLG/REINA	0.99110717	0.9932672	0.9977406
Chr1	13507779	13515963	8184	RLG/TEKAY/DEL	1.0	0.99970347	0.99928516
```

This file has the following columns:

* Containing sequence
* LTR-RT initial position in the sequence.
* LTR-RT end position. 
* LTR-RT length. 
* Predicted superfamily (RLG or RLC) and lineage.
* Detection's probability obtained by Inpactor2_Detect.
* Filtering's probability obtained by Inpactor2_Filter (if the filter option was activated during the execution).
* Classification's probrability obtained by Inpactor2_Class.

NOTE: This file will be useful for knowning how reliable were the predictions done by the software, and if you desire, remove them with probabilies lower than a given value. Please note that if you want to discard some elements, you need to do it manually since Inpactor2 does not have any function related to it.

### Annotation output (Repeat Masker, optional)
Optionally, Inpactor2 can use the library created during program execution to annotate LTR-retrotransposons in plant genomes using Repeat Masker software. Inpactor2 uses the following parameters:  -gff -nolow -no_is -norna. In addition, it will use the same number of cores specified in the -t flag of Inpactor2. This outputs will be generated if the flag "-a yes" is defined in the Inpactor2's execution. Due to the execution of Repeat Masker, five additional files will be created in the output directory indicated with the -o flag:
* Inpactor2_anno_summary.txt (The results of the Repeat Masker' script buildSummary.pl)
* genome_file.fasta.masked
* genome_file.fasta.cat.gz
* genome_file.fasta.out
* genome_file.fasta.out.gff
* genome_file.fasta.tbl

Where "genome_file.fasta" is the name of the input genome used in Inpactor2. For more information about RepeatMasker outputs and its operation, please consult its documentation: https://www.repeatmasker.org/webrepeatmaskerhelp.html

## Inpactor2's cycles of analysis
<a name="cycles"/>

Although Inpactor2 approach overcomes most of the challenges, there is still an issue with dividing the sequences into segments, because some elements could be split and thus the structure-based approach would not be able to detect them. However, in LTR_FINDER_PARALLEL tool is proposed that most of these undetected LTR-RTs are represented by complete copies identified in other segments, with a loss of less than 1 %. Additionally, in order to reduce this problem as much as possible, the approach proposed in this tool can be executed in different cycles (from 1 to 5), where each cycle divides differently the input sequences (see figure below) in order to predict the elements that remain split in any of the partitions. At the end, the result of all the cycles are unified, eliminating those elements detected in more than one cycle.

<p align="center">
  <img src="https://github.com/simonorozcoarias/Inpactor2/blob/main/NN_architectures/Inpctor2_cycles.png">
</p>

## Computational resources needed by Inpactor2
<a name="Resources"/>
Inpactor2 uses heterogeneous architectures to execute as fast as possible the detection of LTR-RTs. Therefore, Inpactor2 uses CPUs (using the -t option), RAM, disk and GPU (if available and accessible by Tensorflow). If the server where Inpactor2 is running does not have GPU, the program will use only CPU and may take a little longer. In order to allow the user to estimate the resources required to run Inpactor2, we have run Inpactor2 with plants of different sizes and ratios of LTR-RTs. All these executions were done using a Workstation with 64 threads, a 8 Gb GPU Nvidia RTX 2080 super, with SSD and 128 Gb of RAM.


| Species | Genome size (Mb)  | Number of LTR-RTs | Time (Sec) | Max Ram used (Gb) | Max disk usage (Mb) |
| :------------: |:---------------:| :-----:| :-----:|  :-----:|  :-----:| 
| A. thaliana | 115 | 107 | 179.60 | 0.86 | 571.48 |
| O. sativa | 358 | 1090 | 328.54 | 3.00 | 1794.81 |
| C. canephora | 553 | 730 | 345.12 | 5.07 | 2718.68 |
| S. lycopersicum | 791 | 1401 | 855.31 | 8.87 | 4641.29 |
| C. arabica | 1126.4 | 3656 | 1568.59 | 9.97 | 5610.94 |
| Z. mays | 2150.4 | 35,274 | 10,512.24 | 24.80 | 10,487.07 |

# Inpactor2_utils
<a name="inpactor2_utils"/>

In addition to the main component of Inpactor2, Inpactor2_utils.py contains utilities in the LTR-RT analysis, such as delete characters different from nucleotides (A, C, T, G or N), calculate k-mer frequencies with 1 <= k <= 6, and re-train Inpactor2_Class to specialize the neural network for a certain group of species. 

## Usage
```
python3 Inpactor2_utils.py [-h] -u UTIL -o OUTPUTDIR [-t THREADS] [-f FASTAFILE] [-v]
```
Where the options are the following:
* -h or --help: show this help message and exit
* -u UTIL or --util UTIL: Utility to be used [FILTER, CLASSIFY, KMER] **(required)**.
* -o OUTPUTDIR or --output-dir OUTPUTDIR: Path of the output directory **(required)**.
* -t THREADS or --threads THREADS: Number of threads to be used by Inpactor2. Default: all available threads.
* -f FASTAFILE or --fasta-file FASTAFILE: Path of fasta file containg DNA sequences (for KMER and CLASSIFY utils).
* -l LINEAGE_NAMES or --lineage-names LINEAGE_NAMES: fasta file includes lineage names? [yes or not] (for KMER util). The IDs of the sequences must contain the lineage name followed by a "-" (See CLASSIFY utility for more information about the required format). If this option is yes, then a extra column will be added at the beginning of the result file, containing a numerical representation of the lineage, as following: 1: ALE/RETROFIT, 3: ANGELA, 4: BIANCA, 8: IKEROS, 9: IVANA/ORYCO, 11: TAR/TORK, 13: SIRE, 14: CRM, 16: GALADRIEL, 17: REINA, 18: TEKAY/DEL, 19: ATHILA, 20: TAT. 
* -v or --version: show program's version number and exit.

## Inpactor2_utils Execution
### k-mer counting utility  
<a name="inpactor2_utils_kmer"/>
This utility allows users to count k-mer frequencies in nucleotide sequences from 1 <= k <= 6 through a convolutional neural network called "Inpactor2_K-mers" in a time-efficient way.

To use this utility, execute:
```
python3 Inpactor2_utils.py -u KMER -o output_directory -t num_cores -f multiseq_file.fasta -l yes
```
Where -o output_directory is a folder where Inpactor2 will put final results (it must exists before running the script), num_cores is the number of cores available in your system (by default it will use all cores), multiseq_file.fasta is the fasta file that contains all the sequencies which will be analyze by the software, and -l flag indicates if your multiseq_file.fasta contains the lineage/family classification of each LTR-RT (see CLASSIFY utility to get more information about them).

### CLASSIFY utility  
<a name="inpactor2_utils_classify"/>
This utility lets users to re-train Inpactor2_Class neural network with custom LTR-RT libraries. This library must be in fasta format and sequence's IDs have to contain the lineage/family name followed by "-". Example: ">SIRE-NC_587496_58_17". Inpactor2_Class can receive the next lineage names:

* ALE-
* ALESIA-
* RETROFIT-
* ANGELA-
* BIANCA-
* IKEROS-
* IVANA-
* ORYCO-
* OSSER-
* TAR-
* TORK-
* SIRE-
* CRM-
* GALADRIEL-
* REINA-
* TEKAY- 
* DEL-
* ATHILA-
* TAT-

**NOTE**: Inpactor2_Class was designed and trained only for plant genomes, not for others organisms.  

To run this utility, execute the following:
```
python3 Inpactor2_utils.py -u CLASSIFY -o output_directory -t num_cores -f multiseq_file.fasta 
```
Where -o output_directory is a folder where Inpactor2 will put final results (it must exists before running the script), num_cores is the number of cores available in your system (by default it will use all cores), and multiseq_file.fasta is the fasta file that contains all the sequencies which will be analyze by the software.

### Sequence filter utility
<a name="inpactor2_utils_filter"/>
To avoid a possible error in Inpactor2 caused by a non-nucleotide character (a character different than A, C, T, G or N), This utility removes all those characters. The scripts produces a file with the same name of the input, but adding the extension ".filtered". This output can be used in Inpactor2.

To use this utility, please run:
```
python3 Inpactor2_utils.py -u FILTER -o output_directory -t num_cores -f multioutput_file.fasta 
```
Where -o output_directory is a folder where Inpactor2 will put final results (it must exists before running the script), num_cores is the number of cores available in your system (by default it will use all cores), and multiseq_file.fasta is the fasta file that contains all the sequencies which will be analyze by the software.

## Neural network Architectures
<a name="architectures"/>

To improve the reproducibility, a directory named "NN_architectures" is available with the four neural network architectures in jupyter notebooks. Thus, users can use whole or sections of the Inpactor2's netoworks, re-train the neural networks with their own data or reproduce the results shown.

## For advance GPU users only
<a name="gpu_users"/>
If when running Inpactor2 using GPU (especially on Nvidia RTX 2080 super) you get an error message similar to the following:

```
.... Failed to get convolution algorithm. This is probably because cuDNN failed to initialize ....
```

Try to uncomment the lines from 25 to 29 of Inpactor2.py script, removing the '#' character at the beginning of each line. This is caused by a issue in the GPU and Tensorflow framework.

## References and similar works
<a name="references"/>

* Orozco-Arias, S., Liu, J., Tabares-Soto, R., Ceballos, D., Silva Domingues, D., Garavito, A., ... & Guyot, R. (**2018**). Inpactor, integrated and parallel analyzer and classifier of LTR retrotransposons and its application for pineapple LTR retrotransposons diversity and dynamics. Biology, 7(2), 32.
* Orozco-Arias, S., Isaza, G., & Guyot, R. (**2019**). Retrotransposons in plant genomes: structure, identification, and classification through bioinformatics and machine learning. International journal of molecular sciences, 20(15), 3837.
* Orozco-Arias, S., Isaza, G., Guyot, R., & Tabares-Soto, R. (**2019**). A systematic review of the application of machine learning in the detection and classification of transposable elements. PeerJ, 7, e8311.
* Orozco-Arias, S., Piña, J. S., Tabares-Soto, R., Castillo-Ossa, L. F., Guyot, R., & Isaza, G. (**2020**). Measuring performance metrics of machine learning algorithms for detecting and classifying transposable elements. Processes, 8(6), 638.
* Orozco-Arias, S., Jaimes, P. A., Candamil, M. S., Jiménez-Varón, C. F., Tabares-Soto, R., Isaza, G., & Guyot, R. (**2021**). InpactorDB: a classified lineage-level plant LTR retrotransposon reference library for free-alignment methods based on machine learning. Genes, 12(2), 190.
* Orozco-Arias, S., Candamil-Cortés, M. S., Jaimes, P. A., Piña, J. S., Tabares-Soto, R., Guyot, R., & Isaza, G. (**2021**). K-mer-based machine learning method to classify LTR-retrotransposons in plant genomes. PeerJ, 9, e11456.
* Orozco-Arias, S., Candamil-Cortes, M. S., Jaimes, P. A., Valencia-Castrillon, E., Tabares-Soto, R., Guyot, R., & Isaza, G. (**2021**). Deep Neural Network to Curate LTR Retrotransposon Libraries from Plant Genomes. In International Conference on Practical Applications of Computational Biology & Bioinformatics (pp. 85-94). Springer, Cham.
* Orozco-Arias, S., Candamil-Cortés, M. S., Valencia-Castrillón, E., Jaimes, P. A., Tobón Orozco, N., Arias-Mendoza, M., Tabares-Soto, R., Guyot, R., & Isaza, G. (**2021**). SENMAP: A Convolutional Neural Network Architecture for Curation of LTR-RT Libraries from Plant Genomes. In 2021 IEEE 2nd International Congress of Biomedical Engineering and Bioengineering (CI-IB&BI) (pp. 1-4). IEEE.
* Orozco-Arias, S., Candamil-Cortes, M. S., Jaimes, P. A., Valencia-Castrillon, E., Tabares-Soto, R., Isaza, G., & Guyot, R. (**2022**). Automatic curation of LTR retrotransposon libraries from plant genomes through machine learning. Journal of Integrative Bioinformatics.

## Other useful resourcers
<a name="resources"/>

* Inpactor version 1 (non-DL implementation): [Inpactor V1 github](https://github.com/simonorozcoarias/Inpactor)
* LTR retrotransposon classification experiments using ML: [ML experiments github](https://github.com/simonorozcoarias/MachineLearningInTEs)
* Plant LTR retrotransposon reference library: [InpactorDB dataset](https://zenodo.org/record/5816833#.YdRXUXWZNH4)
* Dataset of genomic features other than LTR-RTs: [Negative Instances dataset](https://zenodo.org/record/4543905#.YdRXpnWZNH4)

## Citation
<a name="citation"/>
if you use this software, or any part of it (as the neural network architectures, functions, etc), please cite us as following:


Orozco-Arias, S., Humberto Lopez-Murillo, L., Candamil-Cortés, M. S., Arias, M., Jaimes, P. A., Rossi Paschoal, A., ... & Guyot, R. (2022). Inpactor2: a software based on deep learning to identify and classify LTR-retrotransposons in plant genomes. Briefings in Bioinformatics.
