Genotyping tools -- Example of usage
=============

Manhattan plot of one or several models
--------------------------

To compute a Manhattan plot without using FUMA, for one or several summary statistics given by the MOSTtest or PLINK, you can use:

.. code-block:: shell

    python3 multi_model_manhattan_plot.py \
        -p path_to_results/ChampollionV0/ORBITAL_left/20-56-02_1/white.British.ancestry/* \
        -f most_orig.sumstats

or

.. code-block:: shell

    python3 multi_model_manhattan_plot.py \
        -p path_to_results/ChampollionV0/ORBITAL_left/20-56-02_1/white.British.ancestry/*most_orig.sumstats

or 

.. code-block:: shell

    python3 multi_model_manhattan_plot.py \
        -p results/GWAS/sstats/Left_Prob_Pred.glm.linear \
        -f glm.linear

Manhattan plot for one or several epochs (one model)
--------------------------

To compute a Manhattan plot without using FUMA, for one model for different epochs, you can use:
(only the parent folder of the epoch0, epoch10, ..., epoch100 folders is mandatory).
For now, it only looks for summary statistics that ends with 'most_orig.sumstats'. Needs to be changed. 

.. code-block:: shell

    python3 multi_epoch_manhattan_plot.py \
        path_to_results/ChampollionV0/FIP_right/20-16-33_0/epoch*

Heuristic to get lead SNPs
-------------------

For now, it only looks for summary statistics that ends with 'most_orig.sumstats'. Needs to be changed. 

.. code-block:: shell

    python3 get_lead_SNP.py path_to_folder_with_sumstats


To compare significant SNPs from different summary statistics
-------------------

For now, it only looks for summary statistics that ends with 'most_orig.sumstats'. Needs to be changed.

.. code-block:: shell

    python3 compare_signi_SNPs.py path_to_parent_folder_containing_sumstats

Or

.. code-block:: shell

    python3 compare_signi_SNPs.py \
                    path_to_sumstats1 \
                    path_to_sumstats2


