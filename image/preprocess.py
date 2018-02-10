def _detect_artifact():
	pass

def fix_artifact(dataset, estimated_center=None, artifacts=None, mask=None):
	if type(dataset)==str and dataset=="help":
		print("This function reduces artifacts of an adu dataset, whose patterns have the same artifacts")
		print("    -> Input: dataset (FLOAT adu patterns, numpy.ndarray, shape=(Nd,Nx,Ny))")
		print("              estimated_center (estimated pattern center, (Cx,Cy))")
		print("              artifacts (artifact location in pattern, numpy.ndarray, shape=(Na,2))")
		print("     *option: mask (mask area of patterns, 0/1 numpy.ndarray where 1 means masked, shape=(Nx,Ny), default=None)")
		print("    -> Return: None (To save RAM, your input dataset is modified directly)")
		print("[Notice] This function cannot reduce backgroud noise, try preprocess.adu2photon instead")
		print("Help exit.")
		return
	import sys
	import numpy as np
	sys.path.append(__file__.split("/image/preprocess.py")[0] + "/analyse")
	import saxs
	import radp

	if estimated_center is None or artifacts is None:
		raise RuntimeError("no estimated_center or artifacts")
	try:
		dataset[0, artifacts[:,0], artifacts[:,1]]
	except:
		raise RuntimeError("Your input artifacts is not valid")

	print("\nAnalysing artifact locations ...")
	dataset[np.where(dataset<0)] = 0
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	powder = saxs.cal_saxs(dataset)
	center = np.array(saxs.frediel_search(powder, estimated_center))
	inv_art_loc = 2*center - artifacts
	print("Data center : " + str(center))
	# whether inv_art_loc exceed pattern size
	normal_inv_art_loc = (inv_art_loc[:,0]<powder.shape[0]).astype(int) & (inv_art_loc[:,0]>=0).astype(int) \
		& (inv_art_loc[:,1]<powder.shape[1]).astype(int) & (inv_art_loc[:,1]>=0).astype(int)
	# whether a pair of artifact points is symmetried by center point
	art_pat = np.zeros(powder.shape)
	art_pat[artifacts] = 1
	pair_inv_art_loc_index = np.where(art_pat[inv_art_loc[:,0],inv_art_loc[:,1]]==1)[0]
	# whether inv_art_loc is in mask area
	if mask is not None:
		mask_inv_art_loc_index = np.where(mask[inv_art_loc[:,0],inv_art_loc[:,1]]==1)[0]
	else:
		mask_inv_art_loc_index = None
	# normal and unique locations
	print("Fix normal artifacts ...")
	normal_inv_art_loc[pair_inv_art_loc_index] = 0
	if mask is not None:
		normal_inv_art_loc[mask_inv_art_loc_index] = 0
	uniq_inv_art_loc = 1 - normal_inv_art_loc
	normal_artifacts = np.where(normal_inv_art_loc==1)[0]
	uniq_artifacts = np.where(uniq_inv_art_loc==1)[0]
	# fix artifacts at normal locations
	dataset[:, artifacts[normal_artifacts,0], artifacts[normal_artifacts,1]] = \
			dataset[:, inv_art_loc[normal_artifacts,0], inv_art_loc[normal_artifacts,1]]
	# fix artifacts at unique locations
	print("Fix unique artifacts ...")
	for loc in artifacts[uniq_inv_art_loc]:
		r = np.linalg.norm(loc)
		shell = radp.shells_2d([r], powder.shape, center)[0]
		mean_intens = np.mean(dataset[:, shell[:,0], shell[:,1]], axis=1)
		dataset[:, loc[0], loc[1]] = mean_intens

def adu2photon(dataset, photon_percent=0.9, nproc=2, transfer=True, force_poisson=False):
	if type(dataset)==str and dataset=="help":
		print("This function is used to evaluate adu value per photon and transfer adu to photon")
		print("    -> Input: dataset ( patterns whith adu values, numpy.ndarray, shape=(Nd,Nx,Ny) )")
		print("     *option: photon_percent ( estimated percent of pixels that has photons, default=0.1)")
		print("     *option: nproc ( number of processes running in parallel, default=2)")
		print("     *option: transfer ( bool, Ture -> evaluate adu unit and transfer to photon, False -> just evlaute, default=True)")
		print("     *option: force_poisson ( bool, whether to determine photon numbers at each pixel according to poisson distribution, default=False, ignored if transfer=False )")
		print("    -> Return: adu (float) or [adu, data_photonCount] ( [float, int numpy.ndarray(Nd,Nx,Ny)] )")
		print("[Notice] This function is implemented with multi-processes. Nd is recommened to be >1k")
		print("Help exit.")
		return
	import sys
	import numpy as np
	sys.path.append(__file__.split("/image/preprocess.py")[0] + "/analyse")
	import saxs

	print("\nEvaluating adu units ...")
	dataset[np.where(dataset<0)] = 0
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	powder = saxs.cal_saxs(dataset)
	countp = np.bincount(np.round(powder.ravel()).astype(int))
	sumc = np.cumsum(countp)
	percentc = sumc/sumc[-1].astype(float)
	adu = np.where(np.abs(percentc-photon_percent)<0.01)[0][0]
	print("Estimated adu value is " + str(adu) + ". Done.\n")

	if transfer:
		import multiprocessing as mp
		print("Transferring adu patterns to photon count patterns ...")
		result = []
		partition = range(0, len(dataset), np.ceil(len(dataset)/float(nproc)).astype(int))
		if len(partition)==nproc:
			partition.append(len(dataset))
		pool = mp.Pool(processes = nproc)
		for i in np.arange(nproc):
			data_part = dataset[partition[i]:partition[i+1]]
			result.append(pool.apply_async(_transfer, args=(data_part,photon_percent,adu,force_poisson,)))
			print("Start process " + str(i) + " .")
		pool.close()
		pool.join()
		out = np.zeros(dataset.shape, dtype='i4')
		for ind,p in enumerate(result):
			out[partition[ind]:partition[ind+1]] = p.get()
		print("Done.\n")
		return adu, out
	else:
		return adu

def _transfer(data, photon_percent, adu, force_poisson):
	import numpy as np

	def poisson(lamb):
		return np.random.poisson(lamb,1)[0]

	if data == []:
		return np.array([])
	re = np.zeros(data.shape, dtype='i4')
	for ind,pat in enumerate(data):
		countp = np.bincount(np.round(pat.ravel()).astype(int))
		sumc = np.cumsum(countp)
		percentc = sumc/sumc[-1].astype(float)
		adu_mine = np.where(np.abs(percentc-photon_percent)<0.01)[0][0]
		real_adu = 0.6*adu_mine + 0.4*adu
		if force_poisson:
			newp = np.frompyfunc(poisson,1,1)
			re[ind] = newp(pat/real_adu)
		else:
			newp = np.round(pat/real_adu).astype(int)
			re[ind] = newp
	return re
