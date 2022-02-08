"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.numpy_tools import save_origin_npy, plot_spectrogram
from util.eeg_tools import save_origin_mat
from gansynth.normalizer import DataNormalizer
import matplotlib.pyplot as plt
from models import networks
import numpy as np


# def visualize_mid_layer_output(model, n_block, f_intv, save_dir):
#     '''
#     保存模型中间层的输出的可视化结果
#     :param model: 需要研究的模型
#     :param n_block: 模型中有几个要输出的模块
#     :param f_intv: 模块最后一层cnn的filter的输出间隔
#     :param save_dir: 图像保存目录
#     :return: None
#     '''
#     fig = plt.figure(figsize=(10, 10))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
#
#     for j in range(n_block):
#
#         conv_out = networks.LayerActivations(getattr(model.netD.module, 'up_' + str(j)))  # 这里根据需要输出的模块名去修改
#         model.test()
#         conv_out.remove()  #
#         act = conv_out.features
#
#         for k in range(0, len(act[0]), f_intv):  # 每间隔f_intv个filter输出结果
#
#             tmp = act[0][k].detach().numpy()
#             h, w = tmp.shape
#             name = 'up' + str(i) + '_' + str(j) + '_' + str(k)  # 保存图像的名字
#             plot_spectrogram(np.arange(w), np.arange(h), tmp, name, show=False, save=True, save_dir=save_dir)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    normalizer = DataNormalizer(None, opt.normalizer_path, False)
    print("DataNormalizer prepared.")
    model.set_normalizer(normalizer)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    npy_dir = os.path.join(web_dir, "npys")
    # mat_dir = os.path.join(web_dir, "mat")
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        if opt.save_mid_layer_visuals:  # 如果要保存中间层结果
            model.visualize_mid_layer_output(i, 'G', 4, 4, '......')
        else:
            model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results 模仿一下这一段把原始的numpy数据给保存下来
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        save_origin_npy(visuals, npy_dir, img_path)
        #save_origin_mat(visuals, mat_dir, img_path, normalizer)
        #save_resized_npy(visuals, npy_dir, img_path, (22, 256))
    webpage.save()  # save the HTML
