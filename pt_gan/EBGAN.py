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
        self.code = nn.Sequential(
            nn.Linear(64 * (self.input_height // 2) * (self.input_width // 2), 32), # bn and relu are excluded since code is used in pullaway_loss
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64 * (self.input_height // 2) * (self.input_width // 2)),
            nn.BatchNorm1d(64 * (self.input_height // 2) * (self.input_width // 2)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            # nn.Sigmoid(),         # EBGAN does not work well when using Sigmoid().
        )
        pt_gan.utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        code = self.code(x)
        x = self.fc(code)
        x = x.view(-1, 64, (self.input_height // 2), (self.input_width // 2))
        x = self.deconv(x)

        return x, code

class EBGAN(object):
    def __init__(self, params):
        # parameters
        self.sample_num = 64
        self.batch_size = params.batch_size
        self.dataset = params.dataset
        self.gpu_mode = params.gpu_mode
        self.model_name = params.gan_type

        # EBGAN parameters
        self.pt_loss_weight = 0.1
        self.margin = max(1, self.batch_size / 64.)  # margin for loss function
        # usually margin of 1 is enough, but for large batch size it must be larger than 1

        # networks init
        self.G = generator(self.dataset)
        self.D = discriminator(self.dataset)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        pt_gan.utils.print_network(self.G)
        pt_gan.utils.print_network(self.D)
        print('-----------------------------------------------')

        # load mnist
        self.data_X, self.data_Y = pt_gan.utils.load_mnist(params.dataset)
        self.z_dim = 62

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

    def train(self, params):
        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

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
        logger = Logger(params.log_dir)
        step = 0

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(params.epoch):
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

                D_real, D_real_code = self.D(x_)
                D_real_err = self.MSE_loss(D_real, x_)

                G_ = self.G(z_)
                D_fake, D_fake_code = self.D(G_)
                G_ = Variable(G_.data, requires_grad=False)
                D_fake_err = self.MSE_loss(D_fake, G_)
                if list(self.margin-D_fake_err.data)[0] > 0:
                    D_loss = D_real_err + (self.margin - D_fake_err)
                else:
                    D_loss = D_real_err
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake, D_fake_code = self.D(G_)
                G_ = Variable(G_.data, requires_grad=False)
                D_fake_err = self.MSE_loss(D_fake, G_)
                G_loss = D_fake_err + self.pt_loss_weight * self.pullaway_loss(D_fake_code)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                      ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.data[0], G_loss.data[0]))

                # ============ TensorBoard logging ============#
                logger.scalar_summary('d_error_real', D_real_err.data[0], step + 1)
                logger.scalar_summary('d_error_fake', D_fake_err.data[0], step + 1)
                logger.scalar_summary('d_loss', D_loss.data[0], step + 1)
                logger.scalar_summary('g_loss', G_loss.data[0], step + 1)
                step += 1

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1), params.result_dir)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        params.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save(params.save_dir)
        pt_gan.utils.generate_animation(params.result_dir + '/' + self.model_name, params.epoch)
        pt_gan.utils.loss_plot(self.train_hist, params.save_dir, self.model_name)

    def pullaway_loss(self, embeddings):
        """ pullaway_loss tensorflow version code

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            similarity = tf.matmul(
                normalized_embeddings, normalized_embeddings, transpose_b=True)
            batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
            pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
            return pt_loss

        """
        norm = torch.sqrt(torch.sum(embeddings ** 2, 1, keepdim=True))
        normalized_embeddings = embeddings / norm
        similarity = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(1, 0))
        batch_size = embeddings.size()[0]
        pt_loss = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss


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