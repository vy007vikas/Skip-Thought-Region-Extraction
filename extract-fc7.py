import numpy as np
import tensorflow as tf
import os
from os import listdir
import ConfigParser
from scipy import misc
import h5py
import time

def main():
	parser = ConfigParser.ConfigParser()
	parser.read('./settings.conf')
	vgg_loc = parser.get('model','conv_net')
	data_loc = parser.get('data','image_dataset')
	fc7_loc = parser.get('data','fc7')
	img_info_loc = parser.get('data','image_info')

	with open(vgg_loc) as vgg:
		vgg_net = vgg.read()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg_net)

	images = tf.placeholder(tf.float32,[None,224,224,3])

	tf.import_graph_def(graph_def, input_map={ "images":images })
	graph = tf.get_default_graph()

	for opn in graph.get_operations():
		print opn.name , opn.values()

	sess = tf.Session()
	batch_size = 64
	count = 0
	count_bad = 0
	start = time.clock()
	image_batch = np.ndarray((batch_size,224,224,3))
	fc7 = np.ndarray((0,4096))
	image_id = []
	image_shape = []
	flag = False
	for file in listdir(data_loc):
		if not file.endswith('.jpg'):
			continue
		print os.path.splitext(file)[0]
		try:
			im = misc.imread(data_loc+file)
		except:
			count_bad = count_bad + 1
			continue
		image_id.append( int(os.path.splitext(file)[0]) )
		if len(im.shape) == 2:
			img_final = np.ndarray( (im.shape[0], im.shape[1], 3), dtype = 'uint8')
			img_final[:,:,0] = im
			img_final[:,:,1] = im
			img_final[:,:,2] = im
		else :
			img_final = im
		image_shape.append(np.array(img_final).shape)
		img_final = misc.imresize(img_final,(224,224,3))
		image_batch[count,:,:,:] = np.array(img_final)
		count = count + 1
		if count == batch_size:
			count = 0
			feed_dict = { images : image_batch[0:batch_size,:,:,:] }
			fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
			fc7_batch = sess.run(fc7_tensor,feed_dict=feed_dict)
			print fc7.shape , fc7_batch.shape
			fc7 = np.append(fc7,fc7_batch,axis=0)
	if count > 0:
		feed_dict = { images : image_batch[0:count,:,:,:] }
		fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
		fc7_batch = sess.run(fc7_tensor,feed_dict=feed_dict)
		fc7 = np.append(fc7,fc7_batch,axis=0)
	image_id = np.asarray(image_id,int)
	image_shape = np.asarray(image_shape,int)

	print fc7.shape
	print image_shape.shape
	print image_id.shape

	end = time.clock()
	print ' Time taken to extract fc7 features is ' , (end-start)/(60.0*60.0) , ' hrs'
	print count_bad , ' images could not be opened'

	print ' Saving fc7 features to ' , fc7_loc
	fc7_h5 = h5py.File(fc7_loc , 'w')
	fc7_h5.create_dataset('fc7_features', data=fc7)
	fc7_h5.close()

	print ' Saving image data to ' , img_info_loc
	img_info_h5 = h5py.File(img_info_loc, 'w')
	img_info_h5.create_dataset('image_shape',data=image_shape)
	img_info_h5.create_dataset('image_ids',data=image_id)
	img_info_h5.close()

	print ' Data preprocessing complete! '


if __name__ == '__main__':
	main()
