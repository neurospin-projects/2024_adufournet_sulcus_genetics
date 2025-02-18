#!/bin/python					       #
# -*- coding: utf-8 -*-                                # 
#                                                      #
#  File author(s): 			               #
#     Antoine Dufournet                                #
#     Vincent Frouin                                   #
#     Inspired E Larsonneur <elise.larsonneur@cea.fr>  #
#                                                      #
#  2024                                                #
#                                                      #
########################################################

shell.prefix("set -eo pipefail; ")

onsuccess:
    print("+++++++++++++++++++++++++++++++++++++++++")
    print("+++++ Workflow successfully ended. ++++++")
    print("+++++++++++++++++++++++++++++++++++++++++")
onerror:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~ Workflow failed. ~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

###############################################################################
#target files
###############################################################################
rule all:
  input:
    # for the PRL rules
    expand(config["tempodir"] + "/{model}/{ancestry}/cov.pc", model=config["modeln"],ancestry=config["which_ancestry"]), 
    expand(config["tempodir"] + "/{model}/{ancestry}/sid.lst", model=config["modeln"],ancestry=config["which_ancestry"]),

    # for the GWA rules to generate the genomic files that are needed for GWAS
#    expand(config["dataimp"] + "/ukb_imp_chr{chr_n}_v3.bgen", chr_n=config["chrn"]),
#    expand(config["tempodir"]+ "/gen/imputed_chr{chr_n}_decim"+"_maf-{}.bed".format(config["imputemaf"]),
#                   chr_n=config["chrn"]),
#    expand(config["tempodir"] + "/gen/bedlist.lst", model=config["modeln"]),
#    expand(config["tempodir"] + "/gen/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"])),
#    expand(config["tempodir"] + "/gen/imputed_autosomes_decim_maf-{}.fam".format(config["imputemaf"])),

    # for the MOSTest rule
    expand(config["outdir"] + "/{model}/{ancestry}/pheno_pre_residualized.txt", 
                   model=config["modeln"], ancestry=config["which_ancestry"]),
    expand(config["tempodir"] + "/{model}/{ancestry}/list_IID_kept.txt", 
                   model=config["modeln"], ancestry=config["which_ancestry"]),

    expand(config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"]),
                   model=config["modeln"], ancestry=config["which_ancestry"]),
    expand(config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.fam".format(config["imputemaf"]),
                   model=config["modeln"], ancestry=config["which_ancestry"]),

    expand(config["outdir"] + "/{model}/{ancestry}/pheno_pre_residualized_reordered.txt", 
                   model=config["modeln"], ancestry=config["which_ancestry"]),
    expand(config["tempodir"] + "/{model}/{ancestry}/mostest/mostest_imputed_autosomes_decim_maf-{imputemaf}.mat", 
                  imputemaf=config["imputemaf"], ancestry=config["which_ancestry"], model=config["modeln"]),
    expand(config["outdir"] + "/{model}/{ancestry}/mostest_imputed_autosomes_decim_maf-{imputemaf}.most_orig.sumstats",
                  imputemaf=config["imputemaf"], ancestry=config["which_ancestry"], model=config["modeln"]),
    expand(config["outdir"] + "/{model}/{ancestry}/mostest_imputed_autosomes_decim_maf-{imputemaf}_most_orig.zmat.tsv",
                  imputemaf=config["imputemaf"], ancestry=config["which_ancestry"], model=config["modeln"]),

    # for the CKP  CHECKPOINT RULE -> uncover the dynamic part of the pipeline (line after)
    expand(config["tempodir"] + "/{model}/{ancestry}/pheno.phe", model=config["modeln"], ancestry=config["which_ancestry"]),
    expand(config["tempodir"] + "/{model}/{ancestry}/covar.cov", model=config["modeln"], ancestry=config["which_ancestry"]),
    expand(config["tempodir"] + "/{model}/{ancestry}/pheno.hdr", model=config["modeln"], ancestry=config["which_ancestry"])
    #pheno_flist_ckp


###############################################################################
#
# rules PRL: prologs
#    PRL_POP
#    PRL_DUMP: prepare sid list, prepare covar file:
#
###############################################################################
#localrules: PRL_POP
rule PRL_POP:
  input: 
     config["dataphe"] + "/{model}/" + config["infile"] + ".pc"
  output: 
     phephop = config["tempodir"] + "/{model}/{ancestry}/cov.pc"
  log:
    out = config["logdir"] + '/{model}/{ancestry}/PRL_POP.out',
    err = config["logdir"] + '/{model}/{ancestry}/PRL_POP.err',
    rc =  config["logdir"] + '/{model}/{ancestry}/PRL_POP.rc'
  shell: """
         (
           pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 ukb_phecov -- \
               create \
                  --modify addcovar  \
                  --pcfile {input} \
                  --outfile {output} \
                  --which_ancestry {wildcards.ancestry}

         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ; exit $(cat {log.rc});
         """

###############################################################################
#localrules: PRL_DUMP
rule PRL_DUMP:
  input: 
     config["tempodir"] + "/{model}/{ancestry}/cov.pc"
  output: 
     sidlist = config["tempodir"] + "/{model}/{ancestry}/sid.lst"
  log:
    out = config["logdir"] + '/{model}/{ancestry}/PRL_DUMP.out',
    err = config["logdir"] + '/{model}/{ancestry}/PRL_DUMP.err',
    rc =  config["logdir"] + '/{model}/{ancestry}/PRL_DUMP.rc'
  shell: """
         (

            SLST=`basename {output.sidlist}`
            DIR=`dirname {output.sidlist}`

           pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 ukb_phecov -- \
              dump \
                   --info IID_list \
   		   --GCTA_which pheno \
                   --pcfile {input} \
		   --outdir $DIR \
                   --fname $SLST

         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ; exit $(cat {log.rc});
         """

###############################################################################
#
# rules GWA: GWAS
#    GWA_DEC:  chrom_decimate:
#    GWA_LIS:  prepar_chrom list
#    GWA_AGG:  aggregate chrom with sid short list and filtering:
#
###############################################################################

#rule GWA_DEC:
#  input:
#     bgen         = config["dataimp"] + "/ukb_imp_chr{chr_n}_v3.bgen",
#     unordlistsubj = config["tempodir"] + "/{model}/{ancestry}/sid.lst", 
#  output: 
#     chr_dec       = config["tempodir"] + "/{model}" + "/gen/imputed_chr{chr_n}_decim.bed"
#  log:
#    out = config["tempodirlog"] + '/{model}/GWA_DEC_{chr_n}.out',
#    err = config["tempodirlog"] + '/{model}/GWA_DEC_{chr_n}.err',
#    rc = config["tempodirlog"] + '/{model}/GWA_DEC_{chr_n}.rc'
#  params:
#     threads       = config["threads"]
#  shell: """
#         ( 
#           pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
#              plink2 -- \
#                 --memory 16000 --threads {params.threads} \
#                 --bgen   {input.bgen} \
#                 --sample $(echo '{input.bgen}' | sed -e "s/.bgen/_s487395.sample/") \
#                 --make-bed \
#                 --keep-fam   {input.unordlistsubj} \
#                 --out $(echo '{output.chr_dec}' | sed -e "s/.bed//")
#
#         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ;  exit $(cat {log.rc});
#         """

##############################################################################
#rule GWA_MAF:
#  input:
#     imp_chr       = config["tempodir"] + "/{model}" + "/gen/imputed_chr{chr_n}_decim.bed"
#  output: 
#     imp_chrf      = config["tempodir"] + "/{model}" + "/gen/imputed_chr{chr_n}_decim"+"_maf-{}.bed".format(config["imputemaf"]),
#     imp_chr_bim   = config["tempodir"] + "/{model}" + "/gen/imputed_chr{chr_n}_decim"+"_maf-{}.bim".format(config["imputemaf"]),
#     imp_chr_fam   = config["tempodir"] + "/{model}" + "/gen/imputed_chr{chr_n}_decim"+"_maf-{}.fam".format(config["imputemaf"])
#  params:
#     threads       = config["threads"],
#     maf       = config["imputemaf"]
#  log:
#    out = config["tempodirlog"] + '/{model}/GWA_MAF_{chr_n}.out',
#    err = config["tempodirlog"] + '/{model}/GWA_MAF_{chr_n}.err',
#    rc = config["tempodirlog"]  + '/{model}/GWA_MAF_{chr_n}.rc'
#  shell: """
#         ( 
#
#	 pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
#              plink2-17-02-22 -- \
#                   --memory 16000 --threads {params.threads} \
#                   --bfile   $(echo '{input.imp_chr}' | sed -e "s/.bed//") \
#                   --make-bed \
#                   --snps-only \
#                   --maf {params.maf} \
#                   --out   $(echo '{output.imp_chrf}' | sed -e "s/.bed//")
#
#         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ;  exit $(cat {log.rc});
#	"""
#
##################################################################################
#localrules: GWA_LIS
#rule GWA_LIS:
#  input:
#    lambda wildcards: expand(
#        os.path.join(config["tempodir"], wildcards.model, "/gen/imputed_chr{chr_n}_decim_maf-{imputemaf}.bed"),
#        chr_n=config["chrn"],
#        imputemaf=config["imputemaf"]
#    )
#  output:
#    config["tempodir"] + "/{model}" + "/gen/bedlist.lst"
#  log:
#    out = config["tempodirlog"] + '/{model}/GWA_LIS.out',
#    err = config["tempodirlog"] + '/{model}/GWA_LIS.err',
#    rc = config["tempodirlog"] + '/{model}/GWA_LIS.rc'
#  shell: """
#         ( 
#           ls -C1 {input} > {output}
#           sed --in-place s/\.bed// {output}
#
#         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ;  exit $(cat {log.rc});
#	 """
#
##################################################################################
#rule GWA_AGG:
#  input:
#    bedlst = config["tempodir"] + "/{model}" + "/gen/bedlist.lst",
#  output:
#    bedout = config["tempodir"] + "/{model}" + "/gen/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"]),
#    famout = config["tempodir"] + "/{model}" + "/gen/imputed_autosomes_decim_maf-{}.fam".format(config["imputemaf"]),
#    bimout = config["tempodir"] + "/{model}" + "/gen/imputed_autosomes_decim_maf-{}.bim".format(config["imputemaf"]),
#  params:
#     maf           = config["imputemaf"],
#     threads       = config["threads"]
#  log:
#    out = config["tempodirlog"] + '/{model}/GWA_AGG.out',
#    err = config["tempodirlog"] + '/{model}/GWA_AGG.err',
#    rc = config["tempodirlog"]  + '/{model}/GWA_AGG.rc'
#  shell: """
#         ( pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
#              plink2-17-02-22 -- \
#                        --memory 16000 --threads {params.threads}  \
#	                    --pmerge-list {input.bedlst} bfile \
#	                    --make-bed \
#                        --max-alleles 2 \
#                        --maf {params.maf} \
#	                    --out   $(echo '{output.bedout}' | sed -e "s/.bed//")
#         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ;  exit $(cat {log.rc});
#	 """
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This rule create a file and is declared as ckpt to force re-running of 
# rule sets by snakemake
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#localrules: CKP
checkpoint CKP:
  input:
    pheno = config["tempodir"] + "/{model}/{ancestry}/cov.pc",
  output:
     phelist = config["tempodir"] + "/{model}/{ancestry}/pheno.hdr",
     pheno =  config["tempodir"] + "/{model}/{ancestry}/pheno.phe", 
     covar =  config["tempodir"] + "/{model}/{ancestry}/covar.cov", 
  log:
    out = config["logdir"] + '/{model}/{ancestry}/CKP_.out',
    err = config["logdir"] + '/{model}/{ancestry}/CKP_.err',
    rc  = config["logdir"] + '/{model}/{ancestry}/CKP_.rc'
  shell: """
         (

           COV=`basename {output.covar}`
           PHE=`basename {output.pheno}`
           DIR=`dirname {output.pheno}`

            pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 ukb_phecov -- \
		dump \
                   --info PLINK2_pheno_list --PLINK2_which pheno \
                   --pcfile {input} --outdir $DIR --fname $PHE

            pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 ukb_phecov -- \
            	dump \
                   --info PLINK2_covar_list --PLINK2_which covar \
                   --pcfile {input} --outdir $DIR --fname $COV


         ) 1>{log.out} 2>{log.err} && echo $? > {log.rc} || echo $? > {log.rc} ;  exit $(cat {log.rc});
	 """

##################################################################################
# For the MOSTest, it is adviced to pre-residualize the phenotypes, 
# to see how it is done, check the pre_residualize.py file
# The script will also extract the IIDs common to the intersection
# of the covar.cov file and the pheno.phe files.
##################################################################################
rule PreResidualizePheno:
  input:
    pheno = config["tempodir"] + "/{model}/{ancestry}/pheno.phe", 
    covar = config["tempodir"] + "/{model}/{ancestry}/covar.cov", 
  output:
    pre_residu_pheno=config["outdir"] + "/{model}/{ancestry}/pheno_pre_residualized.txt",
    list_selected_IID=config["tempodir"] + "/{model}/{ancestry}/list_IID_kept.txt",
  params:
    script=config["most_python_scripts"] + "/pre_residualize.py",
  log:
    out=config["logdir"] + "/{model}/{ancestry}/mostest_pre_residu_pheno.out",
    err=config["logdir"] + "/{model}/{ancestry}/mostest_pre_residu_pheno.err",
  shell: """
        (
        pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
            python3 {params.script} \
                {input.pheno} \
                {input.covar} \
                {output.pre_residu_pheno}
        ) > {log.out} 2> {log.err}
        """

##################################################################################
# For the MOSTest, the pheno and the .fam file must be in the same order. 
# We will apply a filter to the .bed .fam .bim files to select only the 
# IIDs that are in the list_IID_kept.txt given by the covariates list.
# The bfile used as input here can be pre-calculated and saved at config["tempodirgen"]
# for instance, but can also be calculated once if all the phenotypes you are working
# with are derived from a same global list of participants.
##################################################################################
rule FilterGenetics:
  input:
    list_selected_IID=config["tempodir"] + "/{model}/{ancestry}/list_IID_kept.txt", 
    bfile=config["tempodirgen"] + "/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"]), 
    #bfile=config["tempodir"] + "/gen/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"]), 
  output:
    bedfile=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"]),
    famfile=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.fam".format(config["imputemaf"]),
    bimfile=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.bim".format(config["imputemaf"]),
  params:
     threads=config["threads"]
  log:
    out=config["logdir"] + "/{model}/{ancestry}/mkbfile.out",
    err=config["logdir"] + "/{model}/{ancestry}/mkbfile.err",
  shell: """
        (
        # Extract base name and directory of the bfile, removing the .bed extension
        bfile_base=$(basename {input.bfile} .bed)
        bfile_dir=$(dirname {input.bfile})
        bfile_path="$bfile_dir/$bfile_base"

        # Extract base name and directory of the output mat file, removing the .mat extension
        outfile_base=$(basename {output.bedfile} .bed)
        outfile_dir=$(dirname {output.bedfile})
        outfile_path="$outfile_dir/$outfile_base"

        pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
            plink2 -- \
                --memory 16000 --threads {params.threads}  \
                --bfile $bfile_path \
                --keep {input.list_selected_IID} \
                --make-bed \
                --out $outfile_path \
        ) > {log.out} 2> {log.err}
        """

###################################################################################
# As asked for the MOSTest command, the pheno must be reorderd given the order of the 
# subjects in the .fam files. The .fam file is here chosen as the extrated .fam file
# from the global bfile. (All the necessary bfiles are extracted at the same time).
# Then the IID and the FID columns must me removed (see check_and_reorder_pheno.py)
####################################################################################
rule ReOrderPheno:
  input:
    pre_residu_pheno=config["outdir"] + "/{model}/{ancestry}/pheno_pre_residualized.txt",
    famfile=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.fam".format(config["imputemaf"]),
    reorder_pheno=config["most_python_scripts"] + "/reorder_pheno_autosome.py",
  output:
    reordered_pheno=config["outdir"] + "/{model}/{ancestry}/pheno_pre_residualized_reordered.txt",
  log:
    out=config["logdir"] + "/mostest_{model}/{ancestry}_reorder_pheno.out",
    err=config["logdir"] + "/mostest_{model}/{ancestry}_reorder_pheno.err",
  shell: """
        (
        pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
            python3 {input.reorder_pheno} \
                {input.pre_residu_pheno} \
                {input.famfile}
        ) > {log.out} 2> {log.err}
        """


# #################################################################################
# Extract the base name of the file, removing the .bed extension
# Extract the directory part of the file path
# Construct the path without the .bed extension
# Do the MOSTest for each chromosome separetly with the mostest command
###################################################################################
rule MOSTest:
  input:
    pheno=config["outdir"] + "/{model}/{ancestry}/pheno_pre_residualized_reordered.txt",
    bfile=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.bed".format(config["imputemaf"]),
  output:
    mat=config["tempodir"] + "/{model}/{ancestry}/mostest" + "/mostest_imputed_autosomes_decim"+"_maf-{}.mat".format(config["imputemaf"]),
    zmatmat=config["tempodir"] + "/{model}/{ancestry}/mostest" + "/mostest_imputed_autosomes_decim"+"_maf-{}_zmat.mat".format(config["imputemaf"]),
  log:
    out=config["logdir"] + "/{model}/{ancestry}/mostest_imputed_autosomes"+"_decim_maf-{}.out".format(config["imputemaf"]),
    err=config["logdir"] + "/{model}/{ancestry}/mostest_imputed_autosomes"+"_decim_maf-{}.err".format(config["imputemaf"]),
  shell: """
    # Extract base name and directory of the bfile, removing the .bed extension
    bfile_base=$(basename {input.bfile} .bed)
    bfile_dir=$(dirname {input.bfile})
    bfile_path="$bfile_dir/$bfile_base"

    # Extract base name and directory of the output mat file, removing the .mat extension
    outfile_base=$(basename {output.mat} .mat)
    outfile_dir=$(dirname {output.mat})
    outfile_path="$outfile_dir/$outfile_base"
    
    # Run MOSTest command and redirect output and error logs
    (
    pcocc-rs run --env UKB_HOME n4h00001rs:mphen-genetics-0.1 -- mostest \
         {input.pheno} \
         $bfile_path \
         $outfile_path 
    ) > {log.out} 2> {log.err}

    """

##############################################################################
# This rules convert the .mat files to text files. 
# See the doc on https://github.com/precimed/mostest/tree/master, about 
# process_results.py to get the mostest results, and 
# process_results_ext.py to get the zscores from each GWAS
##############################################################################
rule mat_to_txt1:
    input:
        bim=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.bim".format(config["imputemaf"]),
        mat=config["tempodir"] + "/{model}/{ancestry}/mostest" + "/mostest_imputed_autosomes_decim"+"_maf-{imputemaf}.mat",
    output:
        most_orig_stats=config["outdir"] + "/{model}/{ancestry}/mostest_imputed_autosomes_decim_maf-{imputemaf}.most_orig.sumstats",
    params:
        script_dir=config["most_python_scripts"],
    log:
        out=config["logdir"] + "/{model}/{ancestry}/convert_mat_to_txt_imputed_autosomes_maf-{imputemaf}.out",
        err=config["logdir"] + "/{model}/{ancestry}/convert_mat_to_txt_imputed_autosomes_maf-{imputemaf}.err"
    shell:
        """
        matfile_base=$(basename {input.mat} .mat)
        matfile_dir=$(dirname {input.mat})
        matfile_path="$matfile_dir/$matfile_base"

        outfile_base=$(basename {output.most_orig_stats} .most_orig.sumstats)
        outfile_dir=$(dirname {output.most_orig_stats})
        outfile_path="$outfile_dir/$outfile_base"

        # Run process_results.py to produce MOSTest and MinP p-values text files
	# Gives the .minp_perm.sumstats and the .most_perm.sumstats
        # Gives the .minp_orig.sumstats and the .most_orig.sumstats
        (
        pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
              python3 {params.script_dir}/process_results.py \
                      {input.bim} \
                      $matfile_path \
                      $outfile_path \
        )> {log.out} 2> {log.err}

        # Run process_results_ext.py to produce univariate GWAS results text file        
        """

rule mat_to_txt2:
    input:
        bim=config["tempodir"] + "/{model}/{ancestry}"+"/gen/imputed_autosomes_decim_maf-{}.bim".format(config["imputemaf"]),
        mat=config["tempodir"] + "/{model}/{ancestry}/mostest/" + "mostest_imputed_autosomes_decim_maf-{imputemaf}.mat",
        zmatmat=config["tempodir"] + "/{model}/{ancestry}/mostest/" + "mostest_imputed_autosomes_decim_maf-{imputemaf}_zmat.mat",
    output:
        most_orig_zmat=config["outdir"] + "/{model}/{ancestry}/mostest_imputed_autosomes_decim_maf-{imputemaf}_most_orig.zmat.tsv",

    params:
        script_dir=config["most_python_scripts"],
    log:
        out=config["logdir"] + "/{model}/{ancestry}/convert_zmat_to_txt_imputed_autosomes_maf-{imputemaf}.out",
        err=config["logdir"] + "/{model}/{ancestry}/convert_zmat_to_txt_imputed_autosomes_maf-{imputemaf}.err"
    shell:
        """
        matfile_base=$(basename {input.mat} .mat)
        matfile_dir=$(dirname {input.mat})
        matfile_path="$matfile_dir/$matfile_base"
        zstat_base=$(basename {output.most_orig_zmat} _most_orig.zmat.tsv)
        zstat_dir=$(dirname {output.most_orig_zmat})
        zstat_path="$zstat_dir/$zstat_base"

        # Run process_results.py to produce MOSTest p-values text files
	# Gives the ._most.zmat.tsv 
        (
        pcocc-rs run --env UKB_HOME n4h00001rs:quant-genetics-0.3 \
              python3 {params.script_dir}/process_results_ext.py \
                      {input.bim} \
                      $matfile_path \
                      $zstat_path \
        )> {log.out} 2> {log.err}

        # Run process_results_ext.py to produce univariate GWAS results text file        
        """


