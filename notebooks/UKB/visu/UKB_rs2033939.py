import anatomist.api as ana
from soma.qt_gui.qtThread import QtThreadCall
from soma.qt_gui.qt_backend import Qt

a = ana.Anatomist()

from soma import aims
import pandas as pd
import numpy as np
import os

def f_id(x):
    return "sub-"+str(x)

def main():
    dataset = 'UkBioBank40'
    region = 'S.C.-S.Po.C.'
    side = 'R' #"L"

    sample = [1061134, 1067132, 1070151, 1148898, 1150582, 1163252, 1214460, 1251127, 1296415, 1299659, 1337348, 1379399, 1387002, 1393668, 1394959, 1396149, 1396920, 1438199, 1449967, 1476187, 1503440, 1513646, 1518199, 1532274, 1592703, 1598549, 1626146, 1640360, 1653434, 1700491, 1721635, 1740958, 1753033, 1778862, 1779551, 1796448, 1804005, 1805021, 1820008, 1834920, 1865110, 1895787, 1904024, 1914514, 1921308, 1951561, 1987373, 2010885, 2015340, 2025563, 2026620, 2035144, 2049148, 2068104, 2083797, 2108189, 2139279, 2150891, 2151619, 2185112, 2200735, 2236846, 2237335, 2244839, 2248021, 2257245, 2269186, 2286232, 2299342, 2302189, 2306503, 2393843, 2413555, 2436981, 2471329, 2484597, 2485237, 2527206, 2528063, 2537086, 2585646, 2619637, 2621329, 2627867, 2640715, 2652496, 2652683, 2665121, 2681368, 2696974, 2719989, 2767639, 2797457, 2839602, 2901712, 2965080, 2970959, 2974101, 3005985, 3016611, 3028136, 3065441, 3067647, 3080995, 3105840, 3111927, 3157244, 3166647, 3189095, 3198927, 3205376, 3207315, 3238236, 3274235, 3307386, 3311281, 3322226, 3338256, 3347638, 3375793, 3381901, 3409020, 3409841, 3415807, 3422739, 3435095, 3460877, 3486237, 3551658, 3597524, 3653991, 3719895, 3785882, 3808751, 3816031, 3834482, 3846762, 3849149, 3867120, 3875541, 3885263, 3905368, 3917851, 3938633, 3950444, 3971744, 3975160, 4021106, 4106062, 4117872, 4126645, 4129426, 4137534, 4163564, 4167529, 4171049, 4175498, 4178592, 4178742, 4182626, 4219150, 4283526, 4332304, 4381907, 4407113, 4414515, 4426546, 4426552, 4443397, 4446388, 4461088, 4479280, 4525289, 4586783, 4633238, 4644362, 4678355, 4712321, 4725088, 4736156, 4753358, 4776480, 4814576, 4820084, 4824127, 4841663, 4859726, 4872681, 4899779, 4934534, 4936497, 4938154, 4950376, 4961039, 4971562, 5042216, 5045197, 5068532, 5081476, 5085357, 5096308, 5112577, 5145933, 5157167, 5227377, 5227722, 5287248, 5304667, 5321290, 5336163, 5336896, 5347181, 5488693, 5539717, 5565362, 5665510, 5681111, 5726299, 5733158, 5798726, 5803608, 5807356, 5849208, 5866457, 5938636, 5938669, 5940675, 5949734, 5993417, 6003592, 6009693]

    sample = [f_id(x) for x in sample]
    sample = sample[0:12]

    volume=True
    nb_columns=4
    block = a.createWindowsBlock(nb_columns) # nb of columns
    dic_windows = {}

    referential1 = a.createReferential()

    mm_skeleton_path = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{region}/mask/{side}crops'
    dic_windows['Sulci_color']=a.loadObject('/casa/host/build/share/brainvisa-share-5.2/nomenclature/hierarchy/sulcal_root_colors.hie')
    for i, subject_id in enumerate(sample):
        if volume:
            volume_path = f"{mm_skeleton_path}/{subject_id}_cropped_skeleton.nii.gz"
            
            if os.path.isfile(volume_path):
                vol = aims.read(volume_path)
                
                dic_windows[f'a_vol{nb_columns*i}'] = a.toAObject(vol)
                #dic_windows[f'a_vol{i}'].setPalette(absoluteMode=True)
                dic_windows[f'rvol{nb_columns*i}'] = a.fusionObjects(objects=[dic_windows[f'a_vol{nb_columns*i}']], method='VolumeRenderingFusionMethod')
                dic_windows[f'rvol{nb_columns*i}'].releaseAppRef()
                dic_windows[f'rvol{nb_columns*i}'].assignReferential(referential1)
                dic_windows[f'wvr{nb_columns*i}'] = a.createWindow('3D', block=block) #geometry=[100+400*(i%3), 100+440*(i//3), 400, 400])
                dic_windows[f'wvr{nb_columns*i}'].addObjects(dic_windows[f'rvol{nb_columns*i}'])
            else:
                print(f"{volume_path} is not a correct path, or the .nii.gz doesn't exist")

        path_to_t1mri = f'/home/ad279118/tmp1/{subject_id}/ses-2/anat/t1mri/default_acquisition'
        white_matter_path = f'{path_to_t1mri}/default_analysis/segmentation/mesh/{subject_id}_{side}white.gii'
        spam_labelled_sulci_path = f'{path_to_t1mri}/default_analysis/folds/3.1/spam_session_auto/{side}{subject_id}_spam_session_auto.arg'
        deep_labelled_sulci_path = f'{path_to_t1mri}/default_analysis/folds/3.1/deepcnn_session_auto/{side}{subject_id}_deepcnn_session_auto.arg'

        if os.path.isfile(white_matter_path):
            # To visualize the white matter for specific people
            dic_windows[f'white_{subject_id}'] = a.loadObject(white_matter_path)
            #dic_windows[f'white_{subject_id}'].loadReferentialFromHeader()
            dic_windows[f'white_{subject_id}'].assignReferential(referential1)
        else:
            print(f"{white_matter_path} is not a correct path, or the .white.gii doesn't exist")


        
        if os.path.isfile(spam_labelled_sulci_path):
            # To visualize the annotated sulci for specific people
            dic_windows[f'sulci_labelled_{subject_id}'] = a.loadObject(spam_labelled_sulci_path)
            #dic_windows[f'sulci_labelled_{subject_id}'].loadReferentialFromHeader()
            dic_windows[f'sulci_labelled_{subject_id}'].assignReferential(referential1)
        else:
            print(f"{spam_labelled_sulci_path} is not a correct path, or the .arg doesn't exist")
            print("Automatic try with 'deepcnn_session_auto' instead of 'spam_session_auto'")
            if  os.path.isfile(deep_labelled_sulci_path):
                # To visualize the annotated sulci for specific people
                dic_windows[f'sulci_labelled_{subject_id}'] = a.loadObject(deep_labelled_sulci_path)
                #dic_windows[f'sulci_labelled_{subject_id}'].loadReferentialFromHeader()
                dic_windows[f'sulci_labelled_{subject_id}'].assignReferential(referential1)
        
        dic_windows[f'wvr{nb_columns*i+1}'] = a.createWindow('3D', block=block)
        dic_windows[f'wvr{nb_columns*i+1}'].addObjects([dic_windows[f'white_{subject_id}'], dic_windows[f'sulci_labelled_{subject_id}']])

    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication()

    app.exec_()  # Start the Qt event loop

if __name__ == "__main__":
    main()