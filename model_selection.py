# from util.numpy_tools import make_dataset
from util.distance_metrics import calculate_distance
from gansynth.normalizer import DataNormalizer
import os
import shutil
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def find_best_results(checkpoint_dirs, test_file, home_dir, method='GAN', min_epoch=0, save_dir=False, **opt_args):
    """
    挑选在测试集上表型最好的模型
    :param checkpoint_dir: str|list， checkpoints所在的目录
    :param test_file:  str，test.py的完整名字
    :param home_dir: str，home路径
    :param method: str，GAN|EEGGAN|asae, etc
    :param cpts: list，指定要测试的checkponits或者遍历所有
    :param opt_args: opt的flags
    :return: 最好的模型的epcoh数和对应的数值
    """

    if type(checkpoint_dirs) != list:
        checkpoint_dirs = [checkpoint_dirs]

    datasetName = 'cv_0704_60'
    normalizer_dir = os.path.join(home_dir, 'Infos/norm_args/')
    real_path = os.path.join(home_dir, 'Datasets', datasetName, 'A', 'test')
    best_results = {}

    for dir in checkpoint_dirs:

        if dir.endswith('/'):
            dir = dir[:-1]

        experName = dir.split('/')[-1]
        patient = experName.split('_')[-1]
        normalizer_name = datasetName + '_without_' + patient
        domain = 'freq'
        is_IF = False
        if 'is_IF' in opt_args.keys():
            normalizer_name += '_IF'
            is_IF = True
        if 'domain' in opt_args:
            if opt_args['domain'] == 'temporal':
                normalizer_name += '_temporal'
                domain = 'temporal'
        normalizer_name += '.npy'
        normalizer = DataNormalizer(None, os.path.join(normalizer_dir, normalizer_name), False, use_phase=True, domain=domain)

        best_dtw_epoch = {'epoch': -1, 'temporal_mean': float('inf'), 'mag_mean': float('inf'), 'psd_mean': float('inf')}
        best_mag_epoch = dict(best_dtw_epoch)
        best_psd_epoch = dict(best_dtw_epoch)

        # if len(cpts) == 0:  # 若没有指定checkpoints
        cpts = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                cpts.append(fname)
        cpts = [cpt.split('_')[0] for cpt in list(filter(lambda x: 'net_D' in x, cpts))]

        for cpt in cpts:

            if cpt != 'latest' and int(cpt) < min_epoch:
                continue
            cmd = 'python ' + test_file
            cmd += ' --epoch ' + cpt
            cmd += ' --name ' + '_'.join(experName.split('_')[:-1])
            cmd += ' --leave_out ' + patient
            for flag, v in opt_args.items():
                cmd += ' --' + flag + ' ' + str(v)
            if 'xlwang' in home_dir:
                cmd = 'export LD_LIBRARY_PATH=/public/home/xlwang/jyy/anaconda/lib:$LD_LIBRARY_PATH; ' \
                      'source /public/home/xlwang/jyy/anaconda/bin/activate swin; ' + cmd
            os.system(cmd)

            fake_path = os.path.join(home_dir, 'Projects/experiments/results', experName, '_'.join(['test', cpt]), 'npys')
            results = calculate_distance(real_path, fake_path, normalizer=normalizer, method=method, aggregate=True, is_IF=is_IF, aux_normalizer=normalizer)

            if results['temporal_mean'] < best_dtw_epoch['temporal_mean']:
                best_dtw_epoch.update({'epoch': cpt, 'temporal_mean': results['temporal_mean'], 'mag_mean': results['mag_mean'][-1], 'psd_mean': results['psd_mean'][-1]})
            if results['mag_mean'][-1] < best_mag_epoch['mag_mean']:
                best_mag_epoch.update({'epoch': cpt, 'temporal_mean': results['temporal_mean'], 'mag_mean': results['mag_mean'][-1], 'psd_mean': results['psd_mean'][-1]})
            if results['psd_mean'][-1] < best_psd_epoch['psd_mean']:
                best_psd_epoch.update({'epoch': cpt, 'temporal_mean': results['temporal_mean'], 'mag_mean': results['mag_mean'][-1], 'psd_mean': results['psd_mean'][-1]})

        for cpt in cpts:  # 删除不好的结果

            if cpt not in (best_dtw_epoch['epoch'], best_mag_epoch['epoch'], best_psd_epoch['epoch']):
                rm_dir = os.path.join(home_dir, 'Projects/experiments/results', experName, '_'.join(['test', cpt]))
                if os.path.exists(rm_dir):
                    shutil.rmtree(rm_dir)

        best_results[experName] = {'best_temporal': best_dtw_epoch, 'best_mag': best_mag_epoch, 'best_psd': best_psd_epoch}
        if save_dir is not None:
            np.save(os.path.join(save_dir, experName, 'best_test_results'), best_results[experName])

    return best_results


if __name__ == '__main__':
    # home_dir = '/home/hmq'
    home_dir = '/public/home/xlwang/hmq'
    checkpoint_dir = 'Projects/experiments/checkpoints/'
    statDir = os.path.join(home_dir, "Projects/experiments/statistics/")
    exprName = 'cv_pix2pix_attention_1029_lxh'
    test_file = 'test_cv.py'
    method = 'GAN'
    dataroot = os.path.join(home_dir, 'Datasets/cv_0704_60')
    # p_eval = np.load("/public/home/xlwang/hmq/Infos/pix2pix_global_patient_best_results_mapping.npy", allow_pickle=True).item()
    # cpts = [os.path.join(home_dir, checkpoint_dir, v.split('/')[0]) for v in p_eval.values()]
    cpts = os.path.join(home_dir, checkpoint_dir, exprName)

    best_results = find_best_results(cpts, test_file, home_dir, method, 0,
                                     os.path.join(home_dir, statDir),
                                     dataroot=dataroot, direction='BtoA', dataset_mode='eeg', norm='instance',
                                     input_nc=2, output_nc=2, preprocess='none', no_flip='', gpu_ids='-1', phase='test',
                                     model='pix2pix_ae', n_blocks=2, ndf=16, ngf=16, is_IF='')
    print(best_results)
    # np.save(os.path.join(home_dir, statDir, expr, 'best_test_results'), best_results[expr])
