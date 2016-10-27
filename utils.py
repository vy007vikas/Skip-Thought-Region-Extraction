import numpy as np
import json
import h5py
import ConfigParser
import skipthoughts

def load_img_ids(img_info_loc):
	with h5py.File( img_info_loc , 'r') as fl:
		img_id = np.array(fl.get('image_ids'))
	img_id_map = {}
	for i in range(img_id.shape[0]):
		img_id_map[img_id[i]] = i
	return img_id_map

def build_visual_genome():
	parser = ConfigParser.ConfigParser()
	parser.read('./settings.conf')

	data_loc = parser.get('data','visual_genome_data')
	img_info_loc = parser.get('data','image_info')
	vec_loc = parser.get('data','word_vectors')
	captions_loc = parser.get('data','vg_captions')
	img_id_loc = parser.get('data','vg_img_id')
	region_loc = parser.get('data','vg_region')

	model = skipthoughts.load_model()
	img_dict = load_img_ids(img_info_loc)

	with open(data_loc,'r') as configFile:
		data = json.load(configFile)

	vg_dict = {}
	vg_vec = []
	cnt = 1
	total_count = 0
	valid_count = 0
	id_missing = 0
	img_id = []
	region = []
	brk = False

	for val in data:
		for val1 in val['regions']:
			total_count += 1

			if not val1['image_id'] in img_dict:
				id_missing += 1
				continue

			vg_vec.append(skipthoughts.encode(model,val1['phrase']))

			vec = np.zeros((lstm_steps,))

			valid_count = valid_count + 1
			captions.append(vec)
			img_id.append(val1['image_id'])
			region.append(np.array([val1['x'], val1['y'], val1['height'], val1['width']]))

			print val1['phrase']
			print val1['image_id']
			print region

			brk = True
			break
		if brk:
			break

	del data
	captions = np.array(captions)
	img_id = np.array(img_id)
	region = np.array(region)

	vg_vec = np.array(vg_vec)
	print ' Visual Genome custom dictionary built successfully!'

	print ' Missing ids :- ', id_missing
	print ' valid count :- ', valid_count
	print ' total count :- ', total_count

	print ' Saving word vectors to :- ' , vec_loc
	vec_h5 = h5py.File(vec_loc , 'w')
	vec_h5.create_dataset('vg_vec',data=vg_vec)
	vec_h5.close()

	print ' Saving VG encoded captions to :- ' , captions_loc
	captions_h5 = h5py.File(captions_loc , 'w')
	captions_h5.create_dataset('vg_captions',data=captions)
	captions_h5.close()

	print ' Saving VG image ids to :- ' , img_id_loc
	img_id_h5 = h5py.File(img_id_loc , 'w')
	img_id_h5.create_dataset('vg_img_id',data=img_id)
	img_id_h5.close()

	print ' Saving VG regions to :- ' , region_loc
	region_h5 = h5py.File(region_loc , 'w')
	region_h5.create_dataset('vg_region',data=region)
	region_h5.close()

	print ' Visual Genome data built!'
	print ' Count of valid data :- ', valid_count , total_count , float(valid_count)/float(total_count) * 100 , '%'


if __name__ == '__main__':
	build_visual_genome()
