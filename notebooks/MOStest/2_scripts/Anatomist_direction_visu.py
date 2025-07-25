# Antoine Dufournet 2024

"""

Exemple of command:

bv python3 Anatomist_direction_visu.py 
              CINGULATE. \
              R \
              Sorted_projection/regression_on_latent/rs3020595_G.csv \
              6
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

import anatomist.api as ana
#from soma.qt_gui.qtThread import QtThreadCall
from soma.qt_gui.qt_backend import Qt
from soma import aims


a = ana.Anatomist()


def to_bucket(obj):
    """Converts an object to a bucket if it isn't one already."""
    if obj.type() == obj.BUCKET:
        return obj
    avol = a.toAimsObject(obj)
    c = aims.Converter(intype=avol, outtype=aims.BucketMap_VOID)
    abck = c(avol)
    bck = a.toAObject(abck)
    bck.releaseAppRef()
    return bck


def build_gradient(pal):
    """Builds a gradient palette."""
    gw = ana.cpp.GradientWidget(None, 'gradientwidget', pal.header()['palette_gradients'])
    gw.setHasAlpha(True)
    nc = pal.shape[0]
    rgbp = gw.fillGradient(nc, True)
    rgb = rgbp.data()
    npal = pal.np['v']
    pb = np.frombuffer(rgb, dtype=np.uint8).reshape((nc, 4))
    npal[:, 0, 0, 0, :] = pb
    npal[:, 0, 0, 0, :3] = npal[:, 0, 0, 0, :3][:, ::-1]  # Convert BGRA to RGBA
    pal.update()


def buckets_average(subject_id_list, dataset_name_list, region, side):
    """Computes the average bucket volumes for a list of subjects."""
    dic_vol = {}
    dim = 0
    rep = 0

    if not subject_id_list:
        return False

    # Find a valid volume for dimension checking
    while dim == 0 and rep < len(subject_id_list):
        dataset = 'UkBioBank' if dataset_name_list[rep].lower() in ['ukb', 'ukbiobank', 'projected_ukb'] else 'UkBioBank40'
        mm_skeleton_path = f"/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{region}/mask/{side}crops"
        
        file_path = f"{mm_skeleton_path}/{subject_id_list[rep]}_cropped_skeleton.nii.gz"
        if os.path.isfile(file_path):
            sum_vol = aims.read(file_path).astype(float)
            dim = sum_vol.shape
            sum_vol.fill(0)
        else:
            print(f'FileNotFound: {file_path}')
        rep += 1

    # Process each subject
    for subject_id, dataset_name in zip(subject_id_list, dataset_name_list):
        dataset = 'UkBioBank' if dataset_name.lower() in ['ukb', 'ukbiobank', 'projected_ukb'] else 'UkBioBank40'
        mm_skeleton_path = f"/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{region}/mask/{side}crops"

        file_path = f"{mm_skeleton_path}/{subject_id}_cropped_skeleton.nii.gz"
        if os.path.isfile(file_path):
            vol = aims.read(file_path)
            if vol.np.shape != dim:
                raise ValueError(f"{subject_id_list[0]} and {subject_id} must have the same dimensions")
                
            # Convert to binary structure
            struc3D = (vol.np > 0).astype(int)  
            dic_vol[subject_id] = struc3D
            # Accumulate binary volumes
            sum_vol.np[:] += struc3D  
        else:
            print(f'FileNotFound: {file_path}')

    # Normalize the accumulated volume
    sum_vol.np[:] /= len(subject_id_list)
    print(sum_vol.shape)
    return sum_vol


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process brain imaging projections with Anatomist.")
    parser.add_argument("region", type=str, help="Region of the brain to analyze (e.g., 'S.F.int.-F.C.M.ant.').")
    parser.add_argument("side", type=str, choices=["L", "R"], help="Side of the brain (left or right).")
    parser.add_argument("sorted_projections", type=str, help="Path to the sorted projections file.")
    parser.add_argument("nb_columns", type=int, help="Number of columns for the Anatomist windows block.")

    args = parser.parse_args()

    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication(sys.argv)

    # Load sorted projections (a .csv file)
    sorted_projections = pd.read_csv(args.sorted_projections)

    if 'IID' in sorted_projections:
        sorted_projections = sorted_projections.set_index('IID')

    average_dic = {}
    block = a.createWindowsBlock(args.nb_columns)

    # Define the step size for processing projections
    step = 20
    dic_packages = {}
    for i in range(0, len(sorted_projections), step):
        list_idx = sorted_projections.index[i:i + step].to_numpy()
        dic_packages[i // step] = [f'sub-{idx}' for idx in list_idx]

    list_database = ['UkBioBank40'] * step
    n_pack = len(dic_packages)

    # Process each package of subjects
    for i in range(0, n_pack, max(n_pack // (3 * args.nb_columns - 1), 1)):
        sum_vol = buckets_average(dic_packages[i], list_database, args.region, args.side)
        average_dic[f'a_sum_vol{i}'] = a.toAObject(sum_vol)
        average_dic[f'a_sum_vol{i}'].setPalette(minVal=0, absoluteMode=True)

        average_dic[f'rvol{i}'] = a.fusionObjects(objects=[average_dic[f'a_sum_vol{i}']], method='VolumeRenderingFusionMethod')
        average_dic[f'rvol{i}'].releaseAppRef()

        # custom palette
        pal = a.createPalette('VR-palette')
        pal.header()['palette_gradients'] = '0;0.459574;0.497872;0.910638;1;1#0;0;0.52766;0.417021;1;1#0;0.7;1;0#0;0;0.0297872;0.00851064;0.587179;0.0666667;0.838462;0.333333;0.957447;0.808511;1;1'
        build_gradient(pal)
        average_dic[f'rvol{i}'].setPalette('VR-palette', minVal=0.05, maxVal=0.35, absoluteMode=True)
        pal2 = a.createPalette('slice-palette')
        pal2.header()['palette_gradients'] = '0;0.459574;0.497872;0.910638;1;1#0;0;0.52766;0.417021;1;1#0;0.7;1;0#0;0;0.0297872;0.00851064;0.587179;0.0666667;0.838462;0.333333;0.957447;0.808511;1;1'
        build_gradient(pal2)
        average_dic[f'a_sum_vol{i}'].setPalette('slice-palette')

        # Create a 3D window and add the volume rendering object
        average_dic[f'wvr{i}'] = a.createWindow('3D', block=block)
        average_dic[f'wvr{i}'].addObjects(average_dic[f'rvol{i}'])

    app.exec_()  # Start the Qt event loop


if __name__ == "__main__":
    main()
