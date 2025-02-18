
Evaluation and interpretation of sulcus genetics
###########################################################################

This repository is based on https://github.com/neurospin-projects/2023_jlaval_STSbabies. 
It aims to use the embeddings of the self-supervised deep learning pipepline to preterm-specific folding pattern analysis.
Especially on sulcus genetics.
Official Pytorch implementation for Unsupervised Learning and Cortical Folding (`paper <https://openreview.net/forum?id=ueRZzvQ_K6u>`_).


Dependencies
------------
- python >= 3.6
- pytorch >= 1.4.0
- numpy >= 1.16.6
- pandas >= 0.23.3


Set up the work environment
---------------------------
First, the repository can be cloned thanks to:

.. code-block:: shell

    git clone https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics
    cd 2024_adufournet_sulcus_genetics

Then, install a virtual environment through the following command lines:

.. code-block:: shell

    python3 -m venv venv
    . venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -e .

Note that you might need a `BrainVISA <https://brainvisa.info>`_ environment to run
some of the functions or notebooks.

For instance, to compute the average of the sulcal shape given a phenotype., you need the BrainVISA environment, that you can run with `bv bash`.
In this example, we work with the number of allele C as the phenotype.
The region is the anterior cingulate cortex (CINGULATE.).
The hemisphere is the left (L).
The subjects ID are in the IID column (IID).
The phenotype is in the column projection.
The 200-subjects averages will be plot on 2 columns, 1 row.


.. code-block:: shell
   bv bash
   cd notebooks
   python3 MOStest/Interpretation/Moving_average.py -p path_to_regression_on_rs4842267_C.csv \
                                                    -r CINGULATE. \
                                                    -i L \
                                                    -s IID \
                                                    -e projection \
                                                    -n 2 \
                                                    -l 1 \
                                                    -t 200 

To access to the code used to run the MOSTest, go to the MOSTest folder.

.. code-block:: shell
    cd notebooks/MOStest

