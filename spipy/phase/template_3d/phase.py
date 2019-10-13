import numpy as np
import sys, os
import re
import copy

import phasing3d
import phasing3d.utils as utils

import multiprocessing as mp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split(r'(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def out_merge(out, I, good_pix):
    # average the background retrievals
    if out[0]['background'] is not None :
        background = np.mean([i['background'] for i in out], axis=0)
    else :
        background = 0
    
    # centre, flip and average the retrievals
    O, PRTF    = utils.merge.merge_sols(np.array([i['O'] for i in out]), True)
    support, _ = utils.merge.merge_sols(np.array([i['support'] for i in out]).astype(np.float), True)
       
    eMod    = np.array([i['eMod'] for i in out])
    eCon    = np.array([i['eCon'] for i in out])

    # mpi
    if rank == 0:
        Os       = np.empty([size]+list(O.shape), dtype=O.dtype)
        supports = np.empty([size]+list(support.shape), dtype=support.dtype)
        eMods    = np.empty([size]+list(eMod.shape), dtype=eMod.dtype)
        eCons    = np.empty([size]+list(eCon.shape), dtype=eCon.dtype)
        PRTFs    = np.empty([size]+list(PRTF.shape), dtype=PRTF.dtype)
        if background is not 0 :
            backgrounds = np.empty([size]+list(background.shape), dtype=background.dtype)
    else:
        Os       = None
        supports = None
        eMods    = None
        eCons    = None
        PRTFs    = None
        backgrounds = None

    comm.Gather(O, Os, root=0)
    comm.Gather(support, supports, root=0)
    comm.Gather(eMod, eMods, root=0)
    comm.Gather(eCon, eCons, root=0)
    comm.Gather(PRTF, PRTFs, root=0)
    if background is not 0 :
        comm.Gather(background, backgrounds, root=0)

    if rank == 0 :
        PRTFs       = np.abs(np.mean(np.array(PRTFs), axis=0))
        eMods       = np.array(eMods).reshape((size*eMods[0].shape[0], eMods[0].shape[1]))
        eCons       = np.array(eCons).reshape((size*eCons[0].shape[0], eCons[0].shape[1]))
        Os, _       = utils.merge.merge_sols(np.array(Os))
        supports, _ = utils.merge.merge_sols(np.array(supports), True)
        if background is not 0 :
            backgrounds = np.mean(np.array(backgrounds), axis=0)

    if rank == 0:
        # get the PSD
        PSD, PSD_I, PSD_phase = utils.merge.PSD(Os, I)

        out_m = out[0]
        out_m['I'] = np.abs(np.fft.fftn(Os))**2
        out_m['O'] = Os
        out_m['background'] = backgrounds
        out_m['PSD']      = PSD
        out_m['PSD_I']    = PSD_I
        out_m['PRTF']     = PRTFs
        out_m['PRTF_rav'] = np.array([0]) #PRTF_rav
        out_m['eMod']     = eMods
        out_m['eCon']     = eCons
        out_m['support']  = supports
        return out_m
    else:
    	return None

def phase(I, support, params, good_pix = None, sample_known = None):
    d   = {'eMod' : [],         \
           'eCon' : [],         \
           'O'    : None,       \
           'background' : None, \
           'B_rav' : None, \
           'support' : None     \
            }
    out = []

    if sample_known is None:
        params['phasing_parameters']['O'] = None
    else:
        c_dtype = (I[0,0,0] + 1J * I[0, 0, 0]).dtype
        params['phasing_parameters']['O'] = np.array(sample_known).astype(c_dtype)
    
    params['phasing_parameters']['mask'] = good_pix
    
    if params['phasing_parameters']['support'] is None :
        params['phasing_parameters']['support'] = support
    
    alg_iters = config_iters_to_alg_num(params['phasing']['iters'])
        
    # Repeats
    #---------------------------------------------
    for j in range(params['phasing']['repeats']):
        out.append(copy.deepcopy(d))
        params0 = copy.deepcopy(params)
        
        for alg, iters in alg_iters :
            
            if alg == 'ERA':
                O, info = phasing3d.ERA(I, iters, **params0['phasing_parameters'])
         
            if alg == 'DM':
                O, info = phasing3d.DM(I,  iters, **params0['phasing_parameters'])

            if alg == 'RAAR':
                O, info = phasing3d.RAAR(I,  iters, **params0['phasing_parameters'])
         
            out[j]['O']           = params0['phasing_parameters']['O']          = O
            out[j]['eMod']       += info['eMod']
            out[j]['eCon']       += info['eCon']
        
            if 'background' in info.keys():
                out[j]['background']  = params0['phasing_parameters']['background'] = info['background'] * good_pix
                out[j]['B_rav']       = info['r_av']
    
        out[j]['support']     = params0['phasing_parameters']['support']    = info['support']

    return out



if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, sample_known, params = utils.io_utils.read_input_h5(args.input)

    out = phase(diff, support, params, \
                        good_pix = good_pix, sample_known = sample_known)

    out = out_merge(out, diff, good_pix)
    
    # write the h5 file 
    if rank == 0 :
        utils.io_utils.write_output_h5(params['output']['path'], diff, out['I'], support, out['support'], \
                                      good_pix, sample_known, out['O'], out['eMod'], out['eCon'], None,   \
                                      out['PRTF'], out['PRTF_rav'], out['PSD'], out['PSD_I'], out['B_rav'])
        print("\nDone ! Phasing result is stored in " + params['output']['path'] + '/output.h5\n')

