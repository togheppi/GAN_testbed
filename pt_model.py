from pt_gan.GAN import GAN
from pt_gan.CGAN import CGAN
from pt_gan.LSGAN import LSGAN
from pt_gan.DRAGAN import DRAGAN
from pt_gan.ACGAN import ACGAN
from pt_gan.WGAN import WGAN
from pt_gan.infoGAN import infoGAN
from pt_gan.EBGAN import EBGAN
from pt_gan.BEGAN import BEGAN


class PT_model():
    def __init__(self, params):
        super(PT_model, self).__init__()
        self.params = params
        self.model_name = params.gan_type
        
        # declare instance for GAN
        if params.gan_type == 'GAN':
            self.gan = GAN(params)
        elif params.gan_type == 'CGAN':
            self.gan = CGAN(params)
        elif params.gan_type == 'ACGAN':
            self.gan = ACGAN(params)
        elif params.gan_type == 'infoGAN':
            self.gan = infoGAN(params, SUPERVISED=True)
        elif params.gan_type == 'EBGAN':
            self.gan = EBGAN(params)
        elif params.gan_type == 'WGAN':
            self.gan = WGAN(params)
        elif params.gan_type == 'DRAGAN':
            self.gan = DRAGAN(params)
        elif params.gan_type == 'LSGAN':
            self.gan = LSGAN(params)
        elif params.gan_type == 'BEGAN':
            self.gan = BEGAN(params)
        else:
            raise Exception("[!] There is no option for " + params.gan_type)

    def build_model(self):
        pass

    def train_model(self, checkpoint_dir, result_dir, log_dir):
        self.gan.train()

    def test_model(self, result_dir):
        # visualize learned generator
        self.gan.visualize_results(self.params.epoch - 1)