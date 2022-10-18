import os
import numpy as np
from boston_singapore import boston_train, boston_valid, singapore

class Parameters():
        def __init__(self):
                self.n_processors = 8
                # Path
                self.data_dir =  './nuscenes/'
                self.image_dir = self.data_dir + '/images_front/'
                self.pose_dir = self.data_dir + '/poses_front/'
                
                # self.train_video = ['{:04d}'.format(i) for i in range(700)]
                # self.valid_video = ['{:04d}'.format(i + 700) for i in range(150)]
                self.train_video = ['{:04d}'.format(i) for i in boston_train]

                # self.valid_video = ['{:04d}'.format(i)
                #         for i in np.random.choice(singapore, size=50, replace=False)]
                self.valid_video = ['{:04d}'.format(i) for i in boston_valid]
                self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8

                # Data Preprocessing
                self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
                self.img_w = 608   # original size is about 1226
                self.img_h = 184   # original size is about 370
                self.img_means = (-0.08768741516852717, -0.08404858641177723, -0.09257668686045128)
                self.img_stds = (0.21555054670891666, 0.21401082711095745, 0.22159439982015797)
                self.minus_point_5 = True

                # Size of train_df is: 1680 = 560 * 3. Because if we check pickle files, we will find that we have three columns.
                # When seq_len = (36,36), number of samples in training dataset is 560.
                # Number of samples in validation dataset is 232.
                self.seq_len = (5, 7)
                self.sample_times = 3

                # Data info path
                # self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
                # self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
                self.train_data_info_path = 'datainfo/boston_train_df_p{}_seq{}x{}_sample{}.pickle'.format(self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
                self.valid_data_info_path = 'datainfo/boston_valid_df_p{}_seq{}x{}_sample{}.pickle'.format(self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)

                # Model
                self.rnn_hidden_size = 1000
                self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
                self.rnn_dropout_out = 0.5
                self.rnn_dropout_between = 0   # 0: no dropout
                self.clip = None
                self.batch_norm = True
                self.loss_weight = 100
                self.split_loss = True
                # Training
                self.epochs = 200
                self.batch_size = 8
                self.pin_mem = True
                self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
                                        # Choice:
                                        # {'opt': 'Adagrad', 'lr': 0.001}
                                        # {'opt': 'Adam'}
                                        # {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
                
                # Pretrain, Resume training
                self.pretrained_flownet = './pretrained/flownets_bn_EPE2.459.pth.tar'
                                        # Choice:
                                        # None
                                        # './pretrained/flownets_bn_EPE2.459.pth.tar'
                                        # './pretrained/flownets_EPE1.951.pth.tar'
                                        # './pretrained/flownetc_EPE1.766.tar'

                self.resume = True  # resume training
                self.resume_t_or_v = '.train'
                # self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
                # self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
                #
                # self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
                # self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
                # self.save_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
                #
                # self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
                # self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

                self.load_model_path = 'models/boston_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
                self.load_optimizer_path = 'models/boston_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
                self.record_path = 'records/boston_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
                self.save_model_path = 'models/boston_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
                self.save_optimzer_path = 'models/boston_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
                
                if not os.path.isdir(os.path.dirname(self.record_path)):
                        os.makedirs(os.path.dirname(self.record_path))
                if not os.path.isdir(os.path.dirname(self.save_model_path)):
                        os.makedirs(os.path.dirname(self.save_model_path))
                if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
                        os.makedirs(os.path.dirname(self.save_optimzer_path))
                if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
                        os.makedirs(os.path.dirname(self.train_data_info_path))

par = Parameters()

