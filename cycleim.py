#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/15


import utils
import ops
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import util.logit as log
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FaceDataset
from util.exception import NeuralException
from tensorboardX import SummaryWriter
from vggPerceptualLoss import VGGPerceptualLoss

"""
CycleIm
用来模拟游戏引擎：由params生成图片/灰度图
network: 8 layer
input: params (batch, 95)
output: tensor (batch, 3, 512, 512)
"""


class CycleIm(nn.Module):
    def __init__(self, name, args, clean=True):
        """
        CycleIm
        :param name: CycleIm name
        :param args: argparse options
        """
        super(CycleIm, self).__init__()
        self.name = name
        self.args = args
        self.initial_step = 0
        self.prev_path = "./output/CycleIm_preview"
        self.model_path = "./output/CycleIm"
        if clean:
            self.clean()
        self.writer = SummaryWriter(comment='CycleIm', log_dir=args.path_tensor_log)

        """
        model_imitator
        """
        self.model_imitator = nn.Sequential(
            utils.deconv_layer(args.params_cnt, 512, kernel_size=4),  # 1. (batch, 512, 4, 4)
            utils.deconv_layer(512, 512, kernel_size=4, stride=2, pad=1),  # 2. (batch, 512, 8, 8)
            utils.deconv_layer(512, 512, kernel_size=4, stride=2, pad=1),  # 3. (batch, 512, 16, 16)
            utils.deconv_layer(512, 256, kernel_size=4, stride=2, pad=1),  # 4. (batch, 256, 32, 32)
            utils.deconv_layer(256, 128, kernel_size=4, stride=2, pad=1),  # 5. (batch, 128, 64, 64)
            utils.deconv_layer(128, 64, kernel_size=4, stride=2, pad=1),  # 6. (batch, 64, 128, 128)
            utils.deconv_layer(64, 64, kernel_size=4, stride=2, pad=1),  # 7. (batch, 64, 256, 256)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 8. (batch, 3, 512, 512)
            nn.Sigmoid(),
        )
        self.optimizer_imitator = optim.Adam(self.model_imitator.parameters(), lr=args.learning_rate)
        """
        model_inverter 
        """
        self.model_inverter = nn.Sequential(
            utils.conv_layer(3, 64, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(64, 64, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(64, 128, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(128, 256, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(256, 256, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(256, 512, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(512, 512, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(512, 512, kernel_size=4, stride=2, pad=1),
            utils.conv_layer(512, args.params_cnt, kernel_size=4, stride=2, pad=1),
        )
        self.model_inverter.apply(utils.init_weights)
        self.optimizer_inverter = optim.Adam(self.model_inverter.parameters(), lr=args.learning_rate)
        self.l2_c = (torch.ones((512, 512)), torch.ones((512, 512)))
        self.pereptual_lossNet = VGGPerceptualLoss()

    def forward_inverter(self, params):
        """
        forward_inverter module
        :param params: [batch, params_cnt]
        :return: (batch, 3, 512, 512)
        """
        return self.model_inverter(params)

    def forward_imitator(self, params):
        """
        forward_imitator module
        :param params: [batch, params_cnt]
        :return: (batch, 3, 512, 512)
        """
        batch = params.size(0)
        length = params.size(1)
        _params = params.reshape((batch, length, 1, 1))
        return self.model_imitator(_params)


    def itr_train(self, params, reference):
        """
        iterator training
        :param params:  [batch, params_cnt]
        :param reference: reference photo [batch, 3, 512, 512]
        :return loss: [batch], y_: generated picture
        """
        params_fake = self.forward_inverter(reference).squeeze() # CycleIm use image as input
        reference_recon = self.forward_imitator(params_fake)  # Reconstruct
        reference_fake = self.forward_imitator(params)
        params_recon = self.forward_inverter(reference_fake).squeeze()

        self.optimizer_inverter.zero_grad()        # Training inverter
        self.optimizer_imitator.zero_grad()        # Training imitator


        # Inverter Past F.l1_loss
        loss_l1 = self.pereptual_lossNet(params, params_fake)              # Compare to the Ground truth label
        loss_recon = self.pereptual_lossNet(reference, reference_recon)    # Reconsstruction loss
        loss_inverter = 0.9 * loss_l1 + 0.1 * loss_recon


        # Imitator
        loss_l1 = self.pereptual_lossNet(reference, reference_fake)   # Compare to the Ground truth image
        loss_recon = self.pereptual_lossNet(params,params_recon)       # Reconsstruction loss
        loss_imitator = 0.9 * loss_l1 + 0.1 * loss_recon

        loss_inverter.backward()  # 求导  loss: [1] scalar
        loss_imitator.backward()  # 求导  loss: [1] scalar
        self.optimizer_inverter.step()  # update inverter
        self.optimizer_imitator.step()  # update imitator

        return loss_inverter, loss_imitator, params_fake, reference_fake, reference_recon

    def batch_train(self, cuda=False):
        """
        batch training
        :param cuda: 是否开启gpu加速运算
        """
        rnd_input = torch.randn(self.args.batch_size, self.args.params_cnt)
        test_dataset = FaceDataset(self.args, mode="test")
        names, params, images = test_dataset.get_batch(batch_size=self.args.batch_size, edge=False)
        if cuda:
            rnd_input = rnd_input.cuda()
            images    = images.cuda()
        # self.writer.add_graph(self.model_imitator, input_to_model=rnd_input)
        # self.writer.add_graph(self.model_inverter, input_to_model=images)

        self.model_inverter.train()
        dataset = FaceDataset(self.args, mode="train")
        initial_step = self.initial_step
        total_steps = self.args.total_steps
        progress = tqdm(range(initial_step, total_steps + 1), initial=initial_step, total=total_steps)
        for step in progress:
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size, edge=False)
            if cuda:
                params = params.cuda()
                images = images.cuda()

            loss_inverter, loss_imitator, params_fake, reference_fake, reference_recon = self.itr_train(params, images)
            loss_inverter_ = loss_inverter.cpu().detach().numpy()
            loss_imitator_ = loss_imitator.cpu().detach().numpy()
            progress.set_description("loss_inverter: {:.3f} loss_imitator_: {:.3f}".format(loss_inverter_,loss_imitator_))
            self.writer.add_scalar('CycleIm/loss_inverter', loss_inverter_, step)
            self.writer.add_scalar('CycleIm/loss_imitator', loss_imitator_, step)

            if (step + 1) % self.args.prev_freq == 0:
                path = "{1}/imit_{0}.jpg".format(step + 1, self.prev_path)
                self.capture(path, images, reference_fake, reference_recon, self.args.parsing_checkpoint, cuda)  
                x = step / float(total_steps)
                lr = self.args.learning_rate * (x ** 2 - 2 * x + 1) + 2e-3
                utils.update_optimizer_lr(self.optimizer_inverter, lr)
                utils.update_optimizer_lr(self.optimizer_imitator, lr)
                self.writer.add_scalar('CycleIm/learning rate', lr, step)
                self.upload_weights(step)
            if (step + 1) % self.args.save_freq == 0:
                self.save(step)
        self.writer.close()

    def upload_weights(self, step):
        """
        把neural net的权重以图片的方式上传到tensorboard
        :param step: train step
        :return weights picture
        """
        for module in self.model_inverter._modules.values():
            if isinstance(module, nn.Sequential):
                for it in module._modules.values():
                    if isinstance(it, nn.ConvTranspose2d):
                        if it.in_channels == 64 and it.out_channels == 64:
                            name = "weight_{0}_{1}".format(it.in_channels, it.out_channels)
                            weights = it.weight.reshape(4, 64, -1)
                            self.writer.add_image(name, weights, step)
                            return weights

    def load_checkpoint(self, path, training=False, cuda=False):
        """
        从checkpoint 中恢复net
        :param training: 恢复之后 是否接着train
        :param path: checkpoint's path
        :param cuda: gpu speedup
        """
        path_ = self.args.path_to_inference + "/" + path
        if not os.path.exists(path_):
            raise NeuralException("not exist checkpoint of CycleIm with path " + path_)
        if cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location='cpu')
        self.model_inverter.load_state_dict(checkpoint['net'])
        self.optimizer_inverter.load_state_dict(checkpoint['optimizer_inverter'])
        self.optimizer_imitator.load_state_dict(checkpoint['optimizer_imitator'])
        self.initial_step = checkpoint['epoch']
        log.info("recovery CycleIm from %s", path)
        if training:
            self.batch_train(cuda)

    def evaluate(self):
        """
        评估准确率
        方法是loss取反，只是一个相对值
        :return: accuracy rate
        """
        self.model_inverter.eval()
        dataset = FaceDataset(self.args, mode="test")
        steps = 100
        accuracy = 0.0
        losses = []
        for step in range(steps):
            log.info("step: %d", step)
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size, edge=False)
            loss, _ = self.itr_train(params, images)
            accuracy += 1.0 - loss
            losses.append(loss.item())
        self.plot(losses)
        accuracy = accuracy / steps
        log.info("accuracy rate is %f", accuracy)
        return accuracy

    def evaluate_model(self, cuda=False):
        # path_ = self.model_path + "/" + "imitator_30000_cuda.pth"
        """
        Imitator Load
        """
        path_ = "./output/imitator_base" + "/" + "imitator_30000_cuda.pth"
        if not os.path.exists(path_):
            raise NeuralException("not exist checkpoint of imitator with path " + path_)
        if cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location='cpu')
        self.model_imitator.load_state_dict(checkpoint['net'])
        """
        CycleIm Load
        """
        path_ = "./output/CycleIm_base" + "/" + "imitator_30000_cuda.pth"
        if not os.path.exists(path_):
            raise NeuralException("not exist checkpoint of imitator with path " + path_)
        if cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location='cpu')
        self.model_inverter.load_state_dict(checkpoint['net'])


        """
        Start eval
        """
        fake_buffer = None
        self.model_inverter.eval()
        self.model_imitator.eval()
        dataset = FaceDataset(self.args, mode="test")
        for i in range(self.args.eval_CycleIm):
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size, edge=False)
            if cuda:
                params = params.cuda()
                images = images.cuda()

            fake_params = self.model_inverter(images) # Input image to model_inverter
            loss_parm = F.l1_loss(params, fake_params) # Compare to ground truth 
            loss_parm = loss_parm.cpu().detach().numpy()   

            fake_image = self.model_imitator(fake_params)
            loss_image = F.l1_loss(images, fake_image)
            loss_image = loss_image.cpu().detach().numpy()  

            batch = params.size(0)
            length = params.size(1)
            _params = params.reshape((batch, length, 1, 1))
            imitator_image = self.model_imitator.forward_inverter(_params)

            self.output(fake_image,images,imitator_image,sample_id=i) 
            if fake_buffer == None:
                fake_buffer = fake_image
            else:
                print(torch.sum(fake_buffer-fake_image))
                fake_buffer = fake_image

    def output(self, fake_image,images,imitator_image,sample_id=0):
        """
        capture for result
        :param x: generated image with grad, torch tensor [b,params]
        :param refer: reference picture
        :sample_id: sample id of the output sample image
        """
        
        fake_image = self.tensor2image(fake_image)
        images     = self.tensor2image(images)
        imitator_image = self.tensor2image(imitator_image)

        # Blank Image
        im1 = self.l2_c[0]
        np_im1 = im1.cpu().detach().numpy()
        f_im1 = ops.fill_gray(np_im1) 

        image_ = ops.merge_4image(images, fake_image, imitator_image, f_im1, transpose=False)
        path = os.path.join(self.prev_path, "eval_{0}.jpg".format(sample_id))
        cv2.imwrite(path, image_)

    def tensor2image (self, tensorImage):
        image = tensorImage.cpu().detach().numpy()
        image = np.squeeze(image, axis=0)
        image = np.swapaxes(image, 0, 2) * 255
        image = image.astype(np.uint8)
        return image

    def plot(self, losses):
        plt.style.use('seaborn-whitegrid')
        steps = len(losses)
        x = range(steps)
        plt.plot(x, losses)
        plt.xlabel('step')
        plt.ylabel('loss')
        path = os.path.join(self.prev_path, "CycleIm.png")
        plt.savefig(path)

    def clean(self):
        """
        清空前记得手动备份
        """
        ops.clear_files(self.args.path_tensor_log)
        ops.clear_files(self.prev_path)
        ops.clear_files(self.model_path)

    def save(self, step):
        """
       save checkpoint
       :param step: train step
       """
        state = {'net': self.model_inverter.state_dict(), 'optimizer_inverter': self.optimizer_inverter.state_dict(), 
                'optimizer_imitator':self.optimizer_imitator,'epoch': step}
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        ext = "cuda" if self.cuda() else "cpu"
        torch.save(state, '{1}/imitator_{0}_{2}.pth'.format(step + 1, self.model_path, ext))

    @staticmethod
    def capture(path, tensor1, tensor2, tensor3, parse, cuda):
        """
        CycleIm 快照
        :param cuda: use gpu
        :param path: save path
        :param tensor1: input photo
        :param tensor2: generated image (fake)
        :param tensor3: generated image (reconstruction)
        :param parse: parse checkpoint's path
        """
        img1 = ops.tensor_2_image(tensor1)[0].swapaxes(0, 1).astype(np.uint8)
        img2 = ops.tensor_2_image(tensor2)[0].swapaxes(0, 1).astype(np.uint8)
        imgR = ops.tensor_2_image(tensor3)[0].swapaxes(0, 1).astype(np.uint8)
        img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        img3 = utils.faceparsing_ndarray(img1, parse, cuda)
        img4 = utils.img_edge(img3)
        img4 = 255 - ops.fill_gray(img4)
        image = ops.merge_4image(img1, img2, imgR, img4, transpose=False)
        cv2.imwrite(path, image)
