import pt_gan.utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62
            self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        pt_gan.utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # It must be Auto-Encoder style architecture
    # Architecture : (64)4c2s-FC32-FC64*14*14_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (self.input_height // 2) * (self.input_width // 2), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64 * (self.input_height // 2) * (self.input_width // 2)),
            nn.BatchNorm1d(64 * (self.input_height // 2) * (self.input_width // 2)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        pt_gan.utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = x.view(-1, 64, (self.input_height // 2), (self.input_width // 2))
        x = self.deconv(x)

        return x

class BEGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type

        # BEGAN parameters
        self.gamma = 0.75
        self.lambda_ = 0.001
        self.k = 0.

        # networks init
        self.G = generator(self.dataset)
        self.D = discriminator(self.dataset)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            # self.L1_loss = torch.nn.L1loss().cuda()   # BEGAN does not work well when using L1loss().
        # else:
        #     self.L1_loss = torch.nn.L1loss()

        print('---------- Networks architecture -------------')
        pt_gan.utils.print_network(self.G)
        pt_gan.utils.print_network(self.D)
        print('-----------------------------------------------')

        # load mnist
        self.data_X, self.data_Y = pt_gan.utils.load_mnist(args.dataset)
        self.z_dim = 62

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

    def train(self, save_dir, result_dir, log_dir):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        # set the logger for tensorboard
        logger = Logger(log_dir)
        step = 0

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter in range(len(self.data_X) // self.batch_size):
                x_ = self.data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_err = torch.mean(torch.abs(D_real - x_))

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_err = torch.mean(torch.abs(D_fake - G_))

                D_loss = D_real_err - self.k * D_fake_err
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_err = torch.mean(torch.abs(D_fake - G_))

                G_loss = D_fake_err
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                # convergence metric
                temp_M = D_real_err + torch.abs(self.gamma * D_real_err - D_fake_err)

                # operation for updating k
                temp_k = self.k + self.lambda_ * (self.gamma * D_real_err - D_fake_err)
                temp_k = temp_k.data[0]

                # self.k = temp_k.data[0]
                self.k = min(max(temp_k, 0), 1)
                self.M = temp_M.data[0]

                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                      ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.data[0], G_loss.data[0], self.M, self.k))

                # ============ TensorBoard logging ============#
                logger.scalar_summary('d_error_real', D_real_err.data[0], step + 1)
                logger.scalar_summary('d_error_fake', D_fake_err.data[0], step + 1)
                logger.scalar_summary('d_loss', D_loss.data[0], step + 1)
                logger.scalar_summary('g_loss', G_loss.data[0], step + 1)
                logger.scalar_summary('M', self.M, step + 1)
                logger.scalar_summary('K', self.k, step + 1)
                step += 1

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1), result_dir)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save(save_dir)
        pt_gan.utils.generate_animation(result_dir + '/' + self.model_name, self.epoch)
        pt_gan.utils.loss_plot(self.train_hist, save_dir, self.model_name)

    def visualize_results(self, epoch, result_dir, fix=True):
        self.G.eval()

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        pt_gan.utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          result_dir + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self, save_dir):
        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))