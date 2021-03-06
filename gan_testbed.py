import os
import sys
import pip
from datetime import datetime
from enum import Enum
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)

from tf_model import TF_model
from pt_model import PT_model
import ui_gan_testbed

# enum parameters
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')


class ModelParams:
    def __init__(self):
        # initial parameters
        self.gan_type = 'GAN'
        self.dataset = 'mnist'
        self.batch_size = 100
        self.gpu_mode = False
        self.check_params()

    """checking arguments"""
    def check_params(self):
        # --batch_size
        try:
            assert self.batch_size >= 1
        except:
            print('batch size must be larger than or equal to one')


class TrainParams:
    def __init__(self):
        # initial parameters
        self.epoch = 20
        self.checkpoint_dir = 'checkpoint'
        self.save_dir = 'models'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.lrG = 0.0002
        self.lrD = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
    
    """checking arguments"""
    def check_params(self):
        # --checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # --result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # --result_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # --save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --epoch
        try:
            assert self.epoch >= 1
        except:
            print('number of epochs must be larger than or equal to one')


class MyWindow(QMainWindow, ui_gan_testbed.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ui control
        # combo boxes
        self.comboBox_dataset.addItems(['mnist', 'fashion_mnist'])
        self.comboBox_model.addItems(['GAN', 'CGAN', 'ACGAN', 'infoGAN',
                                     'EBGAN', 'WGAN', 'DRAGAN', 'LSGAN', 'BEGAN'])
        self.comboBox_actFnGen.addItems(['ReLU', 'Leaky ReLU', 'tanh'])
        self.comboBox_actFnDisc.addItems(['ReLU', 'Leaky ReLU', 'tanh'])
        self.comboBox_initFn.addItems(['No_init', 'Normal', 'Xavier'])
        self.comboBox_optimizer.addItems(['SGD', 'Adam', 'RMSProp'])

        self.pushButton_buildModel.clicked.connect(self.btn_BuildModel_clicked)
        self.pushButton_clearModel.clicked.connect(self.btn_ClearModel_clicked)
        self.pushButton_trainModel.clicked.connect(self.btn_TrainModel_clicked)
        self.pushButton_evaluation.clicked.connect(self.btn_Evaluation_clicked)
        self.pushButton_tensorBoard.clicked.connect(self.btn_TensorBoard_clicked)
        self.pushButton_Exit.clicked.connect(app.quit)

        self.pushButton_buildModel.setEnabled(True)
        self.pushButton_trainModel.setDisabled(True)
        self.pushButton_clearModel.setDisabled(True)

        self.radioButton_tensorflow.clicked.connect(self.tensorflow_selected)
        self.radioButton_pytorch.clicked.connect(self.pytorch_selected)

        # load library module
        if self.radioButton_tensorflow.isChecked():
            self.tensorflow_selected()
        elif self.radioButton_pytorch.isChecked():
            self.pytorch_selected()

        # initial parameters
        self.model_params = ModelParams()
        self.train_params = TrainParams()
        self.comboBox_model.setCurrentText(self.model_params.gan_type)  # GAN
        self.comboBox_dataset.setCurrentText(self.model_params.dataset)  # mnist
        self.comboBox_actFnGen.setCurrentIndex(0)  # ReLU
        self.comboBox_actFnDisc.setCurrentIndex(1)  # ReLU
        self.comboBox_initFn.setCurrentIndex(0)  # No initializer
        self.comboBox_optimizer.setCurrentIndex(1)  # Adam optimizer

        # show score
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.horizontalLayout_plotPrediction.addWidget(self.canvas)
        self.subplot = self.fig.add_subplot(1, 1, 1)
        self.subplot.set_xticks(range(10))
        self.subplot.set_xlim(-0.5, 9.5)
        self.subplot.set_ylim(0, 1)
        self.subplot.tick_params(axis='both', labelsize=8)

    def tensorflow_selected(self):
        pip_list = pip.get_installed_distributions()
        package_list = [package.key for package in pip_list]
        if 'tensorflow' not in package_list:
            print("tensorflow library is not installed!")
        else:
            print("TensorFlow library is selected.")
        # try:
        #     import tensorflow as tf
        #     print("TensorFlow library is selected.")
        # except ImportError:
        #     raise ImportError("tensorflow library is not installed!")

    def pytorch_selected(self):
        pip_list = pip.get_installed_distributions()
        package_list = [package.key for package in pip_list]
        if 'torch' not in package_list:
            print("pytorch library is not installed!")
        else:
            print("pytorch library is selected.")
        # try:
        #     import torch
        #     import torch.nn as nn
        #     import torch.optim as optim
        #     from torch.autograd import Variable
        #     print("PyTorch library is selected.")
        # except ImportError:
        #     raise ImportError("tensorflow library is not installed!")

    def btn_ClearModel_clicked(self):
        # reset parameters
        del self.model_params
        del self.train_params
        self.model_params = ModelParams()
        self.train_params = TrainParams()
        self.comboBox_model.setCurrentText(self.model_params.gan_type)

        self.label_modelViewer.clear()
        print('Clear model.')
        self.textEdit_log.append('Clear model.')

        self.pushButton_trainModel.setDisabled(True)


    def btn_BuildModel_clicked(self):
        # model parameters
        self.model_params.gan_type = self.comboBox_model.currentText()
        self.model_params.gpu_mode = self.checkBox_gpuMode.isChecked()
        self.model_params.batch_size = self.spinBox_batchSize.value()

        # initialize a model with respective library
        print("\nBuilding a model...")
        self.textEdit_log.append("Building a model...")
        if self.radioButton_tensorflow.isChecked():
            self.gan = TF_model(self.model_params)

        elif self.radioButton_pytorch.isChecked():
            self.gan = PT_model(self.model_params)

        self.gan.build_model()
        print("\nModel is built.")
        self.textEdit_log.append("Model is built.")

        self.pushButton_trainModel.setEnabled(True)
        self.pushButton_evaluation.setEnabled(True)
        self.pushButton_trainModel.setEnabled(True)
        return True

        # img_fn = self.model_params.checkpoint_dir + self.model_params.name + '.png'
        # pixmap = QPixmap(img_fn)
        # self.label_modelViewer = QLabel()
        #
        # width = self.scrollArea_modelViewer.width() - self.scrollArea_modelViewer.verticalScrollBar().sizeHint().width()
        # self.label_modelViewer.setPixmap(pixmap.scaledToWidth(width, mode=Qt.SmoothTransformation))
        # self.scrollArea_modelViewer.setWidget(self.label_modelViewer)

    def btn_TrainModel_clicked(self):
        # update dirs
        temp_log_dir = os.path.join(self.train_params.log_dir, self.model_params.dataset, self.gan.model_name)
        temp_result_dir = os.path.join(self.train_params.result_dir, self.model_params.dataset, self.gan.model_name)
        temp_checkpoint_dir = os.path.join(self.train_params.checkpoint_dir, self.model_params.dataset, self.gan.model_name)
        # timestamp
        now = datetime.now()
        run_stamp = now.strftime('%Y-%m-%d_%H:%M')

        if self.radioButton_tensorflow.isChecked():
            self.train_params.log_dir = temp_log_dir + '/TF_%s' % run_stamp
            self.train_params.result_dir = temp_result_dir + '/TF_%s' % run_stamp
            self.train_params.checkpoint_dir = temp_checkpoint_dir + '/TF_%s' % run_stamp
        elif self.radioButton_pytorch.isChecked():
            self.train_params.log_dir = temp_log_dir + '/PT_%s' % run_stamp
            self.train_params.result_dir = temp_result_dir + '/PT_%s' % run_stamp
            self.train_params.checkpoint_dir = temp_checkpoint_dir + '/PT_%s' % run_stamp

        # training parameters
        self.train_params.epoch = self.spinBox_numEpochs.value()
        self.train_params.lrG = (float)(self.lineEdit_learningRateG.text())
        self.train_params.lrD = (float)(self.lineEdit_learningRateD.text())

        # train model
        self.gan.train_model(self.train_params)
        print(" [*] Training finished!")
        self.textEdit_log.append(" [*] Training finished.")

        self.pushButton_evaluation.setEnabled(True)

    def btn_Evaluation_clicked(self):

        print(" [*] Testing finished!")
        self.textEdit_log.append(" [*] Testing finished!")

    def btn_TensorBoard_clicked(self):
        import subprocess

        subprocess.Popen("/home/mjkim/tensorflow/GAN_testbed/tensorboard --logdir=%s" % self.train_params.log_dir, shell=True)

        print('\nTensorBoard is running: logdir =', self.train_params.log_dir)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()