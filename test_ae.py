import os
from options.test_options import TestOptions
from data import create_dataset
import torch
from models.ae_model import AEModel
from gansynth.normalizer import DataNormalizer
from util.numpy_tools import save_origin_npy


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    save_dir = os.path.join(opt.results_dir, opt.name, opt.phase, 'npys')
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = AEModel(opt)
    model.setup(opt)
    normalizer = DataNormalizer(None, opt.normalizer_path, False)
    print("DataNormalizer prepared.")
    model.set_normalizer(normalizer)
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()
        visuals = model.get_current_visuals()
        save_origin_npy(visuals, save_dir, model.image_paths)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th sample... %s' % (i, model.image_paths))
