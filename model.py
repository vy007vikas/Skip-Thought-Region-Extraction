import numpy as np
import tensorflow as tf
import math

class nnet:

	def area(self,A):
		x,y,w,h = tf.split(1,4,A)
		return w*h

	def area_intersection(self,A,B):
		x1,y1,h1,w1 = tf.split(1,4,A)
		x2,y2,h2,w2 = tf.split(1,4,B)

		x1 = tf.reshape(x1,[tf.shape(x1)[0],])
		y1 = tf.reshape(y1,[tf.shape(y1)[0],])
		h1 = tf.reshape(h1,[tf.shape(h1)[0],])
		w1 = tf.reshape(w1,[tf.shape(w1)[0],])

		x2 = tf.reshape(x2,[tf.shape(x2)[0],])
		y2 = tf.reshape(y2,[tf.shape(y2)[0],])
		h2 = tf.reshape(h2,[tf.shape(h2)[0],])
		w2 = tf.reshape(w2,[tf.shape(w2)[0],])

		mask_hi_1 = tf.to_float(tf.greater_equal(x1,x2))
		mask_hi_2 = tf.to_float(tf.greater(x2,x1))
		hi = mask_hi_1*tf.maximum(0.0,h2-(x1-x2))
		hi = hi + mask_hi_2*tf.maximum(0.0,h1-(x2-x1))

		mask_wi_1 = tf.to_float(tf.greater_equal(y1,y2))
		mask_wi_2 = tf.to_float(tf.greater(y2,y1))
		wi = mask_wi_1*tf.maximum(0.0,w2-(y1-y2))
		wi = wi + mask_wi_2*tf.maximum(0.0,w1-(y2-y1))

		return wi*hi

	def area_union(self,A,B):
		return self.area(A) + self.area(B) - self.area_intersection(A,B)

	def __init__(self,img_dim,skip_size):
		self.sess = tf.InteractiveSession()

		self.img_dim = img_dim
		self.skip_size = skip_size

		self.W1 = tf.Variable(tf.truncated_normal([img_dim+skip_size,512],stddev = 0.1))
		self.B1 = tf.Variable(tf.zeros([512]))

		self.W2 = tf.Variable(tf.truncated_normal([512,256],stddev = 0.1))
		self.B2 = tf.Variable(tf.zeros([256]))

		self.W3 = tf.Variable(tf.truncated_normal([246,4],stddev = 0.1))
		self.B3 = tf.Variable(tf.zeros([4]))

	def build_model(self):
		Ximg = tf.placeholder(tf.float32,[None,self.img_dim])
		Xcap = tf.placeholder(tf.float32,[None,self.skip_size])
		answer = tf.placeholder(tf.float32,[None,4],name='bounding_box')

		X = tf.concat(1,[Ximg,Xcap])

		O1 = tf.matmul(X,self.W1) + self.B1
		O1 = tf.nn.relu(O1)

		O2 = tf.matmul(O1,self.W2) + self.B2
		O2 = tf.nn.relu(O2)

		O3 = tf.matmul(O2,self.W3) + self.B3
		O3 = tf.nn.relu(O3)

		loss = 1 - (self.area_intersection(answer,O3)/self.area_union(answer,O3))
		loss_final = tf.reduce_mean(loss,0)

		return loss_final , O3
