import numpy as np
import scipy.io as sio
import h5py
import sys
import argparse
import os


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--emc", type=str, help=".emc file", required=True)
	parser.add_argument("--saveto", type=str, help="saving to h5 file", required=True)
	parser.add_argument("--quat", type=str, help="quaternion file", default="None")
	parser.add_argument("--det", type=str, help="detector file", default="None")
	parser.add_argument("--density", type=str, help="density map file", default="None")
	parser.add_argument("--intens", type=str, help="intensities file", default="None")
	args = parser.parse_args()

	if not os.path.isfile(args.emc):
		raise ValueError("Can't find input emc file !")
	readfile = args.emc

	if not (args.quat == "None" or os.path.isfile(args.quat)):
		raise ValueError("Can't find input quaternion file !")
	readquat = args.quat

	if not (args.det == "None" or os.path.isfile(args.det)):
		raise ValueError("Can't find input detector file !")
	readdet = args.det

	if not (args.density == "None" or os.path.isfile(args.density)):
		raise ValueError("Can't find input density file !")
	readdensity = args.density

	if not (args.intens == "None" or os.path.isfile(args.intens)):
		raise ValueError("Can't find input intensity file !")
	readinten = args.intens

	savedir = os.path.dirname(args.saveto)
	if savedir != "" and not os.path.exists(savedir):
		raise ValueError("Can't find saving folder !")
	savefile = args.saveto
	print('saving to '+savefile)

	data = np.fromfile(readfile,dtype='i4')
	num_data = data[0]
	pixs = data[1]
	ext = np.zeros((num_data,pixs),dtype='i4')

	one_photon_events = data[256:num_data+256]
	multi_photon_events = data[num_data+256:num_data*2+256]

	total_mul_events = np.sum(multi_photon_events)
	total_one_events = np.sum(one_photon_events)
	pc_one = num_data*2+256
	pc_mul = pc_one+total_one_events
	pc_mul_counts = pc_one+total_one_events+total_mul_events
	for i in range(num_data):
		one_event = one_photon_events[i]
		one_loca = data[pc_one:pc_one+one_event]
		ext[i][one_loca] = 1
		pc_one += one_event

		multi_event = multi_photon_events[i]
		multi_loca = data[pc_mul:pc_mul+multi_event]
		multi_counts = data[pc_mul_counts:pc_mul_counts+multi_event]
		ext[i][multi_loca] = multi_counts
		pc_mul += multi_event
		pc_mul_counts += multi_event
	image_size = int(np.round(np.sqrt(pixs)))
	ext.shape = ((num_data,image_size,image_size))

	f = h5py.File(savefile,'w')
	f.create_dataset('patterns',data=ext, chunks=True, compression="gzip")

	if readquat != "None":
		quat = np.fromfile(readquat,dtype='double')
		quat.shape = (num_data,4)
		f.create_dataset('quaternions',data=quat, chunks=True, compression="gzip")

	if readdensity != "None":
		density = np.fromfile(readdensity, dtype=float)
		density_size = np.int(np.round(len(density)**(1.0/3.0)))
		density.shape = (density_size, density_size, density_size)
		f.create_dataset('electron density',data=density, chunks=True, compression="gzip")

	if readinten != "None":
		intensity = np.fromfile(readinten, dtype=float)
		inten_size = np.int(np.round(len(intensity)**(1.0/3.0)))
		intensity.shape = (inten_size, inten_size, inten_size)
		f.create_dataset('scattering intensity',data=intensity, chunks=True, compression="gzip")

	if readdet != "None":
		detector = np.loadtxt(readdet, skiprows=1)
		f.create_dataset('detector mapping', data=detector, chunks=True, compression="gzip")

	f.close()
	
	
	
	
	
	
