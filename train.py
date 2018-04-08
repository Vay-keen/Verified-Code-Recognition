#-*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, random, cv2

class Discuz():
	def __init__(self):
		# 数据集路径
		self.data_path = './pic/'
		# 写到指定的磁盘路径中 
		self.log_dir = './logs/'
		# 数据集图片大小
		self.width = 30
		self.heigth = 100
		# 最大迭代次数
		self.max_steps = 100
		# 读取数据集
		self.test_imgs, self.test_labels, self.train_imgs, self.train_labels = self.get_imgs()
		# 训练集大小
		self.train_size = len(self.train_imgs)
		# 测试集大小
		self.test_size = len(self.test_imgs)
		# 每次获得batch_size大小的当前训练集指针
		self.train_ptr = 0
		# 每次获取batch_size大小的当前测试集指针
		self.test_ptr = 0
		# 字符字典大小:0-9 a-z A-Z _(验证码如果小于4，用_补齐) 一共63个字符
		self.char_set_len = 63
		# 验证码最长的长度为4
		self.max_captcha = 4
		# 输入数据X占位符
		self.X = tf.placeholder(tf.float32, [None, self.heigth*self.width])
		# 输入数据Y占位符
		self.Y = tf.placeholder(tf.float32, [None, self.char_set_len*self.max_captcha])
		# keepout占位符
		self.keep_prob = tf.placeholder(tf.float32)


	def get_imgs(self, rate = 0.2):
		# 读取图片
		imgs = os.listdir(self.data_path)
		# 打乱图片顺序
		random.shuffle(imgs)
		# 数据集总共个数
		imgs_num = len(imgs)
		# 按照比例求出测试集个数
		test_num = int(imgs_num * rate / (1 + rate))
		# 测试集
		test_imgs = imgs[:test_num]
		# 根据文件名获取测试集标签
		test_labels = list(map(lambda x: x.split('.')[0], test_imgs))
		# 训练集
		train_imgs = imgs[test_num:]
		# 根据文件名获取训练集标签
		train_labels = list(map(lambda x: x.split('.')[0], train_imgs))
		return test_imgs, test_labels, train_imgs, train_labels


	def get_next_batch(self, train_flag=True, batch_size=100):
		
		# 从训练集获取数据
		if train_flag == True:
			if (batch_size + self.train_ptr) < self.train_size:
				trains = self.train_imgs[self.train_ptr:(self.train_ptr + batch_size)]
				labels = self.train_labels[self.train_ptr:(self.train_ptr + batch_size)]
				self.train_ptr += batch_size
			else:
				new_ptr = (self.train_ptr + batch_size) % self.train_size
				trains = self.train_imgs[self.train_ptr:] + self.train_imgs[:new_ptr]
				labels = self.train_labels[self.train_ptr:] + self.train_labels[:new_ptr]
				self.train_ptr = new_ptr

			batch_x = np.zeros([batch_size, self.heigth*self.width])
			batch_y = np.zeros([batch_size, self.max_captcha*self.char_set_len])

			for index, train in enumerate(trains):
				img = np.mean(cv2.imread(self.data_path + train), -1)
				# 将多维降维1维
				batch_x[index,:] = img.flatten() / 255
			for index, label in enumerate(labels):
				batch_y[index,:] = self.text2vec(label)

		# 从测试集获取数据
		else:
			if (batch_size + self.test_ptr) < self.test_size:
				tests = self.test_imgs[self.test_ptr:(self.test_ptr + batch_size)]
				labels = self.test_labels[self.test_ptr:(self.test_ptr + batch_size)]
				self.test_ptr += batch_size
			else:
				new_ptr = (self.test_ptr + batch_size) % self.test_size
				tests = self.test_imgs[self.test_ptr:] + self.test_imgs[:new_ptr]
				labels = self.test_labels[self.test_ptr:] + self.test_labels[:new_ptr]
				self.test_ptr = new_ptr

			batch_x = np.zeros([batch_size, self.heigth*self.width])
			batch_y = np.zeros([batch_size, self.max_captcha*self.char_set_len])

			for index, test in enumerate(tests):
				img = np.mean(cv2.imread(self.data_path + test), -1)
				# 将多维降维1维
				batch_x[index,:] = img.flatten() / 255
			for index, label in enumerate(labels):
				batch_y[index,:] = self.text2vec(label)			

		return batch_x, batch_y

	def text2vec(self, text):
		"""
		文本转向量
		Parameters:
			text:文本
		Returns:
			vector:向量
		"""
		if len(text) > 4:
			raise ValueError('验证码最长4个字符')

		vector = np.zeros(4 * self.char_set_len)
		def char2pos(c):
			if c =='_':
				k = 62
				return k
			k = ord(c) - 48
			if k > 9:
				k = ord(c) - 55
				if k > 35:
					k = ord(c) - 61
					if k > 61:
						raise ValueError('No Map') 
			return k
		for i, c in enumerate(text):
			idx = i * self.char_set_len + char2pos(c)
			vector[idx] = 1
		return vector

	def vec2text(self, vec):
		"""
		向量转文本
		Parameters:
			vec:向量
		Returns:
			文本
		"""
		char_pos = vec.nonzero()[0]
		text = []
		for i, c in enumerate(char_pos):
			char_at_pos = i #c/63
			char_idx = c % self.char_set_len
			if char_idx < 10:
				char_code = char_idx + ord('0')
			elif char_idx < 36:
				char_code = char_idx - 10 + ord('A')
			elif char_idx < 62:
				char_code = char_idx - 36 + ord('a')
			elif char_idx == 62:
				char_code = ord('_')
			else:
				raise ValueError('error')
			text.append(chr(char_code))
		return "".join(text)

	def crack_captcha_cnn(self, w_alpha=0.01, b_alpha=0.1):
		
		x = tf.reshape(self.X, shape=[-1, self.heigth, self.width, 1])
		# 卷积的filter:一个Tensor。数据维度是四维[filter_height, filter_width, in_channels, out_channels]
		# 具体含义是[卷积核的高度, 卷积核的宽度, 图像通道数, 卷积核个数]
		w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
		b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))	
		conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))

		conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
		b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
		conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
		
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
		b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
		
		conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
		conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		w_d = tf.Variable(w_alpha*tf.random_normal([4*13*64, 1024]))
		b_d = tf.Variable(b_alpha*tf.random_normal([1024]))

		dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
		dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
		dense = tf.nn.dropout(dense, self.keep_prob)
		
		w_out = tf.Variable(w_alpha*tf.random_normal([1024, self.max_captcha*self.char_set_len]))
		b_out = tf.Variable(b_alpha*tf.random_normal([self.max_captcha*self.char_set_len]))
		out = tf.add(tf.matmul(dense, w_out), b_out)
		return out

	def train_crack_captcha_cnn(self):
		
		output = self.crack_captcha_cnn()
		# 创建损失函数
		diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y)
		loss = tf.reduce_mean(diff)
		tf.summary.scalar('loss', loss)
		
		# 使用AdamOptimizer优化器训练模型，最小化交叉熵损失
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

		# 计算准确率
		y = tf.reshape(output, [-1, self.max_captcha, self.char_set_len])
		y_ = tf.reshape(self.Y, [-1, self.max_captcha, self.char_set_len])
		correct_pred = tf.equal(tf.argmax(y, 2), tf.argmax(y_, 2))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		merged = tf.summary.merge_all()

		with tf.Session() as sess:
			# 写到指定的磁盘路径中
			train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
			test_writer = tf.summary.FileWriter(self.log_dir + '/test')
			sess.run(tf.global_variables_initializer())

			# 遍历self.max_steps次
			for i in range(self.max_steps):
				# 迭代500次，打乱一下数据集
				if i % 20 == 0:
					self.test_imgs, self.test_labels, self.train_imgs, self.train_labels = self.get_imgs()
				# 每10次，使用测试集，测试一下准确率
				if i % 10 == 0:
					batch_x_test, batch_y_test = self.get_next_batch(False, 100)
					summary, acc = sess.run([merged, accuracy], feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1})
					print('迭代第%d次 accuracy:%f' % (i+1, acc))
					test_writer.add_summary(summary, i)

					# 如果准确率大于85%，则保存模型并退出。
					if acc > 0.85:
						train_writer.close()
						test_writer.close()
						break
				# 一直训练
				else:
					batch_x, batch_y = self.get_next_batch(True, 100)
					loss_value, _ = sess.run([loss, optimizer], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1})
					print('迭代第%d次 loss:%f' % (i+1, loss_value))
					curve = sess.run(merged, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1})
					train_writer.add_summary(curve, i)

			train_writer.close()
			test_writer.close()


if __name__ == '__main__':
	dz = Discuz()
	dz.train_crack_captcha_cnn()


