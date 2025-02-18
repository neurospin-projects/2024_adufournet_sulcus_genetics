Genotyping tools -- Example of usage
=============

Manhattan plot of one or several models
--------------------------

To compute a Manhattan plot without using FUMA, for a summary statistic given by the MOSTtest, you can use:

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

.. code-block:: shell



Heuristic to get lead SNPs
-------------------

.. code-block:: shell



To compare significant SNPs from different summary statistics
-------------------

.. code-block:: shell

