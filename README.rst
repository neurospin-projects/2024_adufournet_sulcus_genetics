
Evaluation and interpretation of sulcus genetics
###########################################################################

This repository is based on https://github.com/neurospin-projects/2023_jlaval_STSbabies. 
It aims to use the embeddings of the self-supervised deep learning pipepline to perform folding pattern analysis, especially in relation to genetics.

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


To perform GWAS and the MOSTest on a High-performance computing (HPC) cluster (TGCC in this case)
---------------------------

You can find the code used for the Anterior Cingulate Cortex genetic analysis in the `AD_ACC <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/tree/ad279118/TGCC_scripts/AD_ACC>`_ folder,  
relying on tools that can be found in the `Tools <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/tree/ad279118/TGCC_scripts/Tools>`_ folder.

To use a classifier on the left Anterior Cingulate Cortex (ACC) to detect the ParaCingulate Sulcus (PCS)
---------------------------

The notebook `Left_Classifier <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/blob/ad279118/notebooks/LEFT_CINGULATE/SVM_classifier_on_left_cingulate_dim256.ipynb>`_ allows 
a generalization of the ACC dataset classification to the UKBioBank subjects.
Same idea can be found for the right hemisphere at `Right_Classifier <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/blob/ad279118/notebooks/RIGHT_CINGULATE/SVM_classifier_on_right_cingulate_dim256.ipynb>`_ .


To perform an average on the masked sulcal skeleton
---------------------------

For instance, to compute the average of the sulcal shape given a phenotype with `Average <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/blob/ad279118/notebooks/MOStest/Interpretation/Moving_average.py, you need the BrainVISA environment>`_.
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

To access the code used to run the MOSTest, go to the `TGCC_scripts <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/tree/ad279118/TGCC_scripts>`_ folder.

To observe the masked sulcal skeleton subject by subject
---------------------------

In a BrainVISA environment (bv bash), use the notebook `UKB_crops.ipynb <https://github.com/neurospin-projects/2024_adufournet_sulcus_genetics/blob/ad279118/notebooks/Figures/UKB_crops.ipynb>`_ to open subjects' sulcal skeleton.



