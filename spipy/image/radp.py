import numpy as np

def help(module):
	if module=="radial_profile_2d":
		print("This function is used to do averaged radial integration of an image")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny))")
		print("              center (the zero point of your radial profile)")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Return: radial_p ( three columns array, first column is radius (pixel), the second one is radial profile, third one is std-error)")
		return
	elif module=="radial_profile_3d":
		print("This function is used to do averaged radial integration of a volume")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny, Nz))")
		print("              center (the zero point of your radial profile)")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny, Nz), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Return: radial_p ( three columns array, first column is radius (pixel), the second one is radial profile, third one is std-error)")
		return
	elif module=="radial_profile":
		print("This function is used to do averaged radial integration of either an image or a volume")
		print("    -> Input: data (input image, numpy.ndarray, shape = (Nx, Ny) or (Nx, Ny, Nz))")
		print("              center (the zero point of your radial profile)")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny, Nz), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Return: radial_p ( three columns array, first column is radius (pixel), the second one is radial profile, third one is std-error)")
		return
	elif module=="shells_2d":
		print("This function returns indices of pixles in a pattern which forms a shell/circle at radius=rads")
		print("    -> Input: rads (int/float list[], a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey))")
		print("              center (data center, int/float turple/list, [cx, cy])")
		print("    -> Return: re (list, [shell1(numpy.ndarray,shape=(N1,2)), shell2(numpy.ndarray,shape=(N2,2)), ...])")
		return
	elif module=="shells_3d":
		print("This function returns indices of pixels in a pattern which forms a shell at radius=rads")
		print("    -> Input: rads (int/float list, a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey, sizez))")
		print("              center (data center, int/float turple/list, [cx, cy, cz])")
		print("    -> Return: re (list, [ shell1(numpy.ndarray), shell2(numpy.ndarray), ...])")
		return
	elif module=="shells":
		print("This function returns indices of pixels in a pattern/volume which forms a shell at radius=rads")
		print("    -> Input: rads (int/float list, a set of radius in pixels)")
		print("              data_shape (int turple, (sizex, sizey) or (sizex, sizey, sizez))")
		print("              center (data center, int/float turple/list, [cx, cy] or [cx, cy, cz])")
		print("    -> Return: re (list, [ shell1(numpy.ndarray), shell2(numpy.ndarray), ...])")
		return
	elif module=="radp_norm_2d":
		print("This function normalize pattern intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern 2, numpy.ndarray, shape=(Nx,Ny)")
		print("              center (center of data, shape=[Cx,Cy])")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("[Notice] zeros point of ref_Iq should locate on the center of input data")
		return
	elif module=="radp_norm_3d":
		print("This function normalize volume intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern 2, numpy.ndarray, shape=(Nx,Ny,Nz)")
		print("              center (center of data, shape=[Cx,Cy,Cz])")
		print("     #option: mask ( 0/1 binary pattern, shape=(Nx, Ny, Nz), 1 means masked area, 0 means useful area, default=None)")
		print("[Notice] While in normalization, zeros point of ref_Iq are forced to locate on the center of input data")
		return
	elif module=="radp_norm":
		print("This function normalize pattern/volume intensities (averaged inside r shells) using a given radial profile")
		print("    -> Input: ref_Iq (Reference radial intensity profile, numpy.ndarray, shape=(Nr,))")
		print("              data (pattern/volume, numpy.ndarray, shape=(Nx,Nz) or (Nx,Ny,Nz)")
		print("              center (center of data, shape=[Cx,Cy] or [Cx,Cy,Cz])")
		print("     #option: mask ( 0/1 binary array, same shape with data, 1 means masked area, 0 means useful area, default=None)")
		print("[Notice] While in normalization, zeros point of ref_Iq are forced to locate on the center of input data")
		return
	elif module=="circle":
		print("This function generates a circle/sphere area with given radius, centered by origin of coordinates")
		print("    -> Input: data_shape (int, 2 or 3, output dimension)")
		print("              rad (float/int, radius)")
		print("    -> Return: index of points inside given radius ( numpy.array, shape=(Np,data_shape))")
		print("[NOTICE] The diameter of output circle is rad*2+1, set rad<0 if you want None output")
	else:
		raise ValueError("No module names "+str(module))


def radial_profile(data, center, mask=None):
	if mask is not None:
		maskdata = data * (1-mask)
	else:
		maskdata = data
	#center_0 = np.round(center)
	meshgrids = np.indices(data.shape)  # return (x, y) or (x, y, z)
	if len(meshgrids) != len(center):
		raise ValueError('Data shape and center do not match each other.')

	r = np.sqrt(sum( ((grid - c)**2 for grid, c in zip(meshgrids, center)) ))
	r = np.round(r).astype(np.int)

	tbin = np.bincount(r.ravel(), maskdata.ravel())
	nr = np.bincount(r.ravel())
	if mask is not None:
		r_mask = np.zeros(r.shape)
		r_mask[np.where(mask==1)] = 1
		nr_mask = np.bincount(r.ravel(), r_mask.ravel())
		nr = nr - nr_mask
	radialprofile = np.zeros_like(nr, dtype=float)
	r_pixel = np.unique(r.ravel())  # sorted
	nomaskr = np.where(nr>0)
	radialprofile[nomaskr] = tbin[nomaskr] / nr[nomaskr]
	if mask is not None:
		residual = maskdata - radialprofile[r] * (1.0 - mask)
	else:
		residual = maskdata - radialprofile[r]
	resid_bin = np.bincount(r.ravel(), residual.ravel()**2)
	std_error = np.zeros_like(radialprofile, dtype=float)
	valid_samples = np.where(nr > 1)
	std_error[valid_samples] = np.sqrt(resid_bin[valid_samples] / (nr[valid_samples] - 1))
	return np.vstack((r_pixel, radialprofile, std_error)).T

def radial_profile_2d(data, center, mask=None):
	print("Warning: radial_profile_2d will be deprecated in next version, use radial_profile instead !")
	return radial_profile(data, center, mask=mask)

def radial_profile_3d(data, center, mask=None):
	print("Warning: radial_profile_3d will be deprecated in next version, use radial_profile instead !")
	return radial_profile(data, center, mask=mask)

def shells(rads, data_shape, center):
	meshgrids = np.indices(data_shape)  # return (x, y) or (x, y, z)
	r = np.sqrt(sum( ((grid - c)**2 for grid, c in zip(meshgrids, center)) ))
	re = []
	for rad in rads:
		sel = np.where(np.abs(r-rad)<0.5)
		re.append(np.vstack(sel).T)
	return re

def shells_2d(rads, data_shape, center):
	print("Warning: shells_2d will be deprecated in next version, use shells instead !")
	return shells(rads, data_shape, center)

def shells_3d(rads, data_shape, center):
	print("Warning: shells_3d will be deprecated in next version, use shells instead !")
	return shells(rads, data_shape, center)

def radp_norm(ref_Iq, data, center, mask=None):
	sum_ref_Iq = np.where(np.cumsum(ref_Iq)>1e-2)[0]
	if len(sum_ref_Iq) == 0:
		raise ValueError("reference Iq curve should not be 0")

	#center_0 = np.round(center)
	meshgrids = np.indices(data.shape)  # return (x, y) or (x, y, z)
	r = np.sqrt(sum( ((grid - c)**2 for grid, c in zip(meshgrids, center)) ))
	r = np.round(r).astype(np.int)
	if mask is not None:
		maskdata = data * (1-mask)
	else:
		maskdata = data

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	if mask is not None:
		r_mask = np.zeros(r.shape)
		r_mask[np.where(mask==1)] = 1
		nr_mask = np.bincount(r.ravel(), r_mask.ravel())
		nr = nr - nr_mask
	radialprofile = np.zeros(len(nr))
	r_pixel = np.sort(list(set(r.ravel())))
	nomaskr = np.where(nr>0)
	radialprofile[nomaskr] = tbin[nomaskr] / nr[nomaskr]

	norm_factor = np.zeros(radialprofile.shape)
	normed_len = min(len(ref_Iq), len(radialprofile))
	
	sum_radp = np.where(np.cumsum(radialprofile)>1e-2)[0]
	if len(sum_radp) == 0:
		return data

	stop_rad = max(sum_ref_Iq[0], sum_radp[0])
	norm_factor[stop_rad:normed_len] = (ref_Iq[stop_rad:normed_len]+1e-9)/(radialprofile[stop_rad:normed_len]+1e-9)

	newdata = np.zeros(data.shape)
	for ind,rad in enumerate(np.arange(r.min(), r.max()+1)):
		newdata[np.where(r==rad)] = maskdata[np.where(r==rad)] * norm_factor[ind]
	return newdata

def radp_norm_2d(ref_Iq, data, center, mask=None):
	print("Warning: radp_norm_2d will be deprecated in next version, use radp_norm instead !")
	return radp_norm(ref_Iq, data, center, mask)

def radp_norm_3d(ref_Iq, data, center, mask=None):
	print("Warning: radp_norm_3d will be deprecated in next version, use radp_norm instead !")
	return radp_norm(ref_Iq, data, center, mask)

def circle(data_shape, rad):
	if rad<0:
		return None
	dsize = data_shape
	if dsize==3:
		cube = [2*int(np.ceil(rad)) + 1]*3
		center = np.array([int(np.ceil(rad))]*3)
		x, y, z = np.indices(cube)
		r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
		selx, sely, selz = np.where(r < rad)
		return np.vstack((selx,sely,selz)).T - center
	elif dsize==2:
		cube = [2*int(np.ceil(rad)) + 1]*2
		center = np.array([int(np.ceil(rad))]*2)
		x, y = np.indices(cube)
		r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
		selx, sely = np.where(r < rad)
		return np.vstack((selx,sely)).T - center
	else:
		raise RuntimeError("Only support 2d or 3d input!")
