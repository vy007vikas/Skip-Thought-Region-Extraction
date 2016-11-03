import numpy as np
import tensorflow as tf
import model
import data_loader
import ConfigParser
import time

def main():
	parser = ConfigParser.ConfigParser()
	parser.read('./settings.conf')

	fc7_loc = parser.get('data','fc7')
	img_info_loc = parser.get('data','image_info')
	captions_loc = parser.get('data','vg_captions')
	img_id_loc = parser.get('data','vg_img_id')
	region_loc = parser.get('data','vg_region')

	learning_rate = float(parser.get('lstm','learning_rate'))
	num_epochs = int(parser.get('lstm','num_epochs'))
	batch_size = int(parser.get('lstm','batch_size'))

	fc7, img_shape, img_id_mapping = data_loader.load_fc7(fc7_loc,img_info_loc)
	captions, img_id, regions = data_loader.load_visual_genome(captions_loc,img_id_loc,region_loc)

	skip_size = captions.shape[1]
	img_dim = fc7.shape[1]

	print ' Total images :- ', fc7.shape
	print ' Region Descriptions shape :- ', captions.shape

	my_net = model.nnet(skip_size,img_dim)
	input_tensor , loss_final , out = my_net.build_model()
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_final)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()
	# saver.restore(sess, './Model/nnet/0.ckpt' )

	totalTime = 0.0
	for i in range(num_epochs):
		batch = 0
		start = time.clock()
		while captions.shape[0] - batch*batch_size > 0:
			len = min(batch_size,captions.shape[0]-batch*batch_size)

			caption_batch = captions[batch*batch_size:batch*batch_size + len:1]
			img_id_batch = img_id[batch*batch_size:batch*batch_size + len:1]
			ans = regions[batch*batch_size:batch*batch_size + len:1]

			img_fc7 = fc7[img_id_mapping[img_id_batch]]
			img_shape = fc7[img_id_mapping[img_id_batch]]
			ans = ans[:,0] / img_shape[:,0]
			ans = ans[:,1] / img_shape[:,1]
			ans = ans[:,2] / img_shape[:,0]
			ans = ans[:,3] / img_shape[:,1]
			ans *= 224

			output, loss, _ = sess.run([out,loss_final,train_op] , feed_dict={
				input_tensor['img_fc7']:img_fc7,
				input_tensor['caption']:caption_batch,
				input_tensor['answer']:ans
			})

			print ' Batch :- ', batch, ' Loss :- ', loss
			batch = batch + 1
		end = time.clock()
		save_path = saver.save(sess, './Model/nnet/' + str(i) + '.ckpt')
		print ' Model saved successfully after ', i, ' epochs'
		print ' Time taken for ', i, 'th epoch :- ', (end-start)/(60.0*60.0) , ' hrs'
		totalTime += (end-start)/(60.0*60.0)
	print ' Model trained after ', totalTime , ' hrs'

if __name__=='__main__':
	main()