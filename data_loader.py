import numpy as np
import h5py

def load_fc7(fc7_loc,img_info_loc):
	with h5py.File( fc7_loc , 'r') as fl:
		fc7 = np.array(fl.get('fc7_features'))
	with h5py.File( img_info_loc , 'r') as fl:
		img_id = np.array(fl.get('image_ids'))
		img_shape = np.array(fl.get('image_shape'))
	img_id_map = {}
	for i in range(img_id.shape[0]):
		img_id_map[img_id[i]] = i
	print ' Image fc7 features loaded successfully!'
	return fc7, img_shape, img_id_map

def load_visual_genome(captions_loc,img_id_loc,region_loc):
	with h5py.File( captions_loc , 'r') as fl:
		captions = np.array(fl.get('vg_captions'))
	with h5py.File( img_id_loc , 'r') as fl:
		img_id = np.array(fl.get('vg_img_id'))
	with h5py.File( region_loc , 'r') as fl:
		region = np.array(fl.get('vg_region'))
	print ' Visual Genome data loaded successfully!'
	return captions, img_id, region