from soma import aims
from scipy import ndimage
import numpy as np
import glob
import argparse
import json
import os


def compute_bbox_mask(arr, verbose):

    # Gets location of bounding box as slices
    objects_in_image = ndimage.find_objects(arr)
    if verbose:
        print(f"ndimage.find_objects(arr) = {objects_in_image}")
    if not objects_in_image:
        raise ValueError("There are only 0s in array!!!")

    loc = objects_in_image[0]
    bbmin = []
    bbmax = []

    for slicing in loc:
        bbmin.append(slicing.start)
        bbmax.append(slicing.stop)

    return np.array(bbmin), np.array(bbmax)

def initialize_empty_volume(path):
    # Read the original to get correct dims + header
    ref = aims.read(path)
    ref_arr = np.asarray(ref)
    
    # Create a new empty volume with the same geometry
    full = aims.Volume(ref_arr.shape, dtype='float32')
    full_arr = np.asarray(full)
    full_arr[:] = 0  # or NaN if preferred

    return full_arr


def uncrop_volume(full_arr, decoded_crop, bbmin, bbmax):
    """
    Places the decoded sub-volume back into the original spatial position.
    Takes voxel-wise max between existing values and decoded_crop.
    """
    try:
        # Extract the region where the crop will be inserted
        region = full_arr[
            bbmin[0]:bbmax[0],
            bbmin[1]:bbmax[1],
            bbmin[2]:bbmax[2]
        ]

        # Compute voxel-wise max
        full_arr[
            bbmin[0]:bbmax[0],
            bbmin[1]:bbmax[1],
            bbmin[2]:bbmax[2]
        ] = np.maximum(region, np.asarray(decoded_crop))

    except Exception as e:
        print("Shape conflict! Check the bbox:", e)
    return full_arr

def main():
    parser = argparse.ArgumentParser(description="To save global decoded volumes in nifty.")
    verbose = False
    subjects = [100206,
    100307,
    100408,
    100610,
    101006,
    101107,
    101309,
    101410,
    101915,
    102008,
    102109,
    102311,
    102513,
    102614,
    102715,
    102816,
    103010,
    103111,
    103212,
    103414,
    103515,
    103818,
    104012,
    104416,
    104820,
    105014,
    105115,
    105216,
    105620,
    105923,
    106016,
    106319,
    106521,
    106824,
    107018,
    107220,
    107321,
    107422,
    107725,
    108020,
    108121,
    108222,
    108323,
    108525,
    108828,
    109123,
    109325,
    109830,
    110007,
    110411,
    110613,
    111009,
    111211,
    111312,
    111413,
    111514,
    111716,
    112112,
    112314,
    112516,
    112819,
    112920,
    113215,
    113316]

    subjects=[197550]

    antero_posteror =['F.Coll.-S.Rh.',
        'S.F.median-S.F.pol.tr.-S.F.sup.',
        'S.F.inf.-BROCA-S.Pe.C.inf.',
        'S.Po.C.',
        'S.C.-S.Po.C.',
        'S.F.inter.-S.F.sup.',
        'S.Call.',
        'S.Call.-S.s.P.-S.intraCing.',
        'F.C.M.post.-S.p.C.',
        'S.s.P.-S.Pa.int.',
        'S.Or.-S.Olf.',
        'F.P.O.-S.Cu.-Sc.Cal.',
        'S.F.marginal-S.F.inf.ant.',
        'S.F.int.-F.C.M.ant.',
        'S.T.i.-S.T.s.-S.T.pol.',
        'S.F.int.-S.R.',
        'Lobule_parietal_sup.',
        'S.T.i.-S.O.T.lat.',
        'S.Pe.C.',
        'S.T.s.br.',
        'F.I.P.-F.I.P.Po.C.inf.',
        'Sc.Cal.-S.Li.',
        'S.T.s.',
        'F.C.L.p.-subsc.-F.C.L.a.-INSULA.',
        'S.C.-sylv.',
        'S.C.-S.Pe.C.',
        'OCCIPITAL',
        'S.Or.']

    SIDE = "R"

    if SIDE == "R":
        side = "right"
    elif SIDE == "L":
        side = "left"

    PATH_LIST_REGIONS = "/neurospin/dico/data/deep_folding/current/sulci_regions_champollion_V1.json"
    with open(PATH_LIST_REGIONS) as f:
        d = json.load(f)

    list_regions = [w.replace('_left', '').replace('_right', '') for w in list(d['brain'].keys())]
    list_regions = list(set(list_regions))

    root_path = "/neurospin/dico/data/deep_folding/current/datasets/ABCD/crops/2mm"
    decode_root_path = "/neurospin/dico/adufournet/2025_Champollion_Decoder/runs/Champollion_V1_after_ablation_256"

    path_initial_volume = f"{root_path}/S.C.-sylv./mask/{SIDE}mask_skeleton.nii.gz"

    for subject in subjects:
        counter = 0
        print(subject)
        initial_volume = aims.read(path_initial_volume)
        full_arr = initialize_empty_volume(path_initial_volume)
        for region in antero_posteror:
            if verbose:
                print(region)
            # read the mask of the given region
            path_to_mask  = f"{root_path}/{region}/mask/{SIDE}mask_skeleton.nii.gz"
            mask_volume = aims.read(path_to_mask)
            mask_arr = np.asarray(mask_volume)
            # compute the box for the given region
            bbmin, bbmax = compute_bbox_mask(mask_arr, verbose)
            # get the reconstruction of the given region
            region = region.replace(".", "")
            if verbose:
                print(bbmax-bbmin)
            npy_recon = glob.glob(f"{decode_root_path}/*_{region}_{side}*/reconstruction_best_model/{subject}_decoded.npy")
            if npy_recon:
                vol_npy = np.load(npy_recon[0]).astype(np.float32)
                if verbose:
                    print(npy_recon[0])
                vol_npy_shape = vol_npy.shape
                if verbose:
                    print(vol_npy_shape)
                vol_npy = vol_npy.reshape(list(vol_npy_shape)+[1])
                # place the reconstruction in the global volume
                full_arr = uncrop_volume(full_arr=full_arr, decoded_crop=vol_npy, bbmin=bbmin, bbmax=bbmax)
            else :
                print(f'Reconstruction not found for region {region}')
            
            file_out = f"/volatile/ad279118/2026_Noillopmahc/{SIDE}_{subject}_decoded_{counter}.nii.gz"
            vol_aims = aims.Volume(full_arr.reshape(96, 114, 96))
            vol_aims.copyHeaderFrom(initial_volume.header())
            aims.write(vol_aims, file_out)
            counter+=1

        # write the global reconstruction
        file_out = f"/volatile/ad279118/2026_Noillopmahc/{SIDE}_{subject}_decoded.nii.gz"
        vol_aims = aims.Volume(full_arr.reshape(96, 114, 96))
        vol_aims.copyHeaderFrom(initial_volume.header())
        aims.write(vol_aims, file_out)

if __name__ == "__main__":
    main()