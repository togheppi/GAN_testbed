import tensorflow as tf
from tf_gan.GAN import GAN
from tf_gan.CGAN import CGAN
from tf_gan.infoGAN import infoGAN
from tf_gan.ACGAN import ACGAN
from tf_gan.EBGAN import EBGAN
from tf_gan.WGAN import WGAN
from tf_gan.DRAGAN import DRAGAN
from tf_gan.LSGAN import LSGAN
from tf_gan.BEGAN import BEGAN
from tf_gan.utils import show_all_variables


class TF_model():
    def __init__(self, params):
        super(TF_model, self).__init__()
        self.params = params
        self.model_name = params.gan_type

        # open session
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

        # declare instance for GAN
        if params.gan_type == 'GAN':
            self.gan = GAN(self.sess, params)
        elif params.gan_type == 'CGAN':
            self.gan = CGAN(self.sess, params)
        elif params.gan_type == 'ACGAN':
            self.gan = ACGAN(self.sess, params)
        elif params.gan_type == 'infoGAN':
            self.gan = infoGAN(self.sess, params)
        elif params.gan_type == 'EBGAN':
            self.gan = EBGAN(self.sess, params)
        elif params.gan_type == 'WGAN':
            self.gan = WGAN(self.sess, params)
        elif params.gan_type == 'DRAGAN':
            self.gan = DRAGAN(self.sess, params)
        elif params.gan_type == 'LSGAN':
            self.gan = LSGAN(self.sess, params)
        elif params.gan_type == 'BEGAN':
            self.gan = BEGAN(self.sess, params)
        else:
            raise Exception("[!] There is no option for " + params.gan_type)

    def build_model(self):
        # build graph
        self.gan.build_model()

        # show network architecture
        show_all_variables()
        
    def train_model(self, params):
        self.gan.train(params)

    def test_model(self, result_dir):
        pass
