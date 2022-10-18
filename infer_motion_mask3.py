import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from lib.data_iterators import SeqReader
from lib.img_processing import ImageProcessor, concat_depth_img
from lib.utils import get_model_opt

from evaluators import EVALUATORS
from options import InferOptions

import pdb

ROWS = 6
COLUMNS = 2
D = 1e-2
TH_X = 0.2
TH_Y = 0.05
TH_Z = 0.005


def save_3dim(path, ego, res_trans, img, depth):
    res_x = res_trans[0,0].detach()             # (128, 416)
    ego_x = ego[0,0].detach()

    res_y = res_trans[0,1].detach()
    ego_y = ego[0,1].detach()

    res_z = res_trans[0,2].detach()
    ego_z = ego[0,2].detach()

    # d=0.1
    # x[x>=0] = x[x>=0]/d
    # x[x<0] = x[x<0]/d
    # pdb.set_trace()
    # res_x = res_x / ego_x.abs() # [-10, 10]
    # res_y = res_y / ego_y.abs() # [-10, 10]
    # res_z = res_z / ego_z.abs() # [-0.1, 0.1]

    res_x = res_x / D
    res_y = res_y / D
    res_z = res_z / D

    res_x = res_x.squeeze().detach().cpu().numpy()
    res_y = res_y.squeeze().detach().cpu().numpy()
    res_z = res_z.squeeze().detach().cpu().numpy()

    mask1 = (res_x > TH_X) + (res_x < -1 * TH_X)
    mask2 = (res_y > TH_Y) + (res_y < -1 * TH_Y)
    mask3 = (res_z > TH_Z) + (res_z < -1 * TH_Z)
    # mask = mask1 + mask2 + mask3
    mask = mask1 + mask2 + mask3
    mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

    res_trans2 = res_trans[0,0].squeeze().detach().cpu().numpy()

    # fig = plt.figure(0)
    fig = plt.figure(figsize=(8,8))

    fig.add_subplot(ROWS, COLUMNS, 1)
    plt.title('image')
    plt.imshow(img)
    plt.colorbar()
    # pdb.set_trace()
    '''
    plt.close("all"); plt.figure(9); plt.imshow(img); plt.colorbar(); plt.ion(); 
    plt.close("all"); plt.figure(9); plt.imshow(depth); plt.colorbar(); plt.ion(); plt.show()
    plt.close("all"); plt.figure(1); plt.imshow(res_x); plt.colorbar(); plt.show()
    plt.close("all"); plt.figure(2); plt.imshow(res_y); plt.colorbar(); plt.show()
    plt.close("all"); plt.figure(figsize=(8,8)); plt.figure(1); plt.imshow(res_x); plt.colorbar(); plt.show()
    plt.close("all"); plt.subplot(2, 2, 1); plt.imshow(res_x); plt.colorbar(); plt.subplot(2, 2, 2); plt.imshow(res_y); plt.colorbar(); plt.subplot(2, 2, 3); plt.imshow(res_z); plt.colorbar()
    '''


    fig.add_subplot(ROWS, COLUMNS, 2)
    plt.title('depth')
    plt.imshow(depth)
    plt.colorbar()
    
    fig.add_subplot(ROWS, COLUMNS, 3)
    plt.title('res x')
    plt.imshow(res_x, vmax=-1, vmin=1, cmap='bwr')
    plt.colorbar()
    fig.add_subplot(ROWS, COLUMNS, 4)
    plt.imshow(mask1, vmax=0, vmin=1, cmap='plasma')
    plt.colorbar()

    fig.add_subplot(ROWS, COLUMNS, 5)
    plt.title('res y')
    plt.imshow(res_y, vmax=-0.5, vmin=0.5, cmap='bwr')
    plt.colorbar()  
    fig.add_subplot(ROWS, COLUMNS, 6)
    plt.imshow(mask2, vmax=0, vmin=1, cmap='plasma')
    plt.colorbar()

    fig.add_subplot(ROWS, COLUMNS, 7)
    plt.title('res z')
    plt.imshow(res_z, vmax=-0.01, vmin=0.01, cmap='bwr')
    plt.colorbar()
    fig.add_subplot(ROWS, COLUMNS, 8)
    plt.imshow(mask3, vmax=0, vmin=1, cmap='plasma')
    plt.colorbar()

    fig.add_subplot(ROWS, COLUMNS, 9)
    plt.title('img * mask')
    plt.imshow(img * mask)
    plt.colorbar()
    fig.add_subplot(ROWS, COLUMNS, 10)
    plt.imshow(mask[:,:,0], vmax=0, vmin=1, cmap='plasma')
    plt.colorbar()    

    fig.add_subplot(ROWS, COLUMNS, 11)
    plt.title('img * (1-mask)')
    plt.imshow(img * (1-mask))
    plt.colorbar()


    # pdb.set_trace()
    plt.savefig(path)
    plt.close(fig)

'''
def save_y(path, res_trans):
    # pdb.set_trace()
    x = res_trans[0,DIM].detach()
    d=0.01

    x[x>=0] = x[x>=0]/d
    x[x<0] = x[x<0]/d
    
    x = x.squeeze().detach().cpu().numpy()
    fig = plt.figure(0)
    plt.imshow(x, vmax=-0.1, vmin=0.1, cmap='bwr')
    plt.colorbar()
    plt.savefig(path)
    plt.close(fig)


def save_z(path, res_trans):
    # pdb.set_trace()
    x = res_trans[0,DIM].detach()
    d=0.01

    x[x>=0] = x[x>=0]/d
    x[x<0] = x[x<0]/d
    
    x = x.squeeze().detach().cpu().numpy()
    fig = plt.figure(0)
    plt.imshow(x, vmax=-0.1, vmin=0.1, cmap='bwr')
    plt.colorbar()
    plt.savefig(path)
    plt.close(fig)
'''
class DepthInference:
    def __init__(self, args):
        
        self.args = args
        self._init_evaluator()
        self._init_img_processor()
        self._init_data_reader()

    def _init_evaluator(self):
        model_opt = get_model_opt(self.args.model_path)
        self.evaluator = EVALUATORS[model_opt.method](model_opt)

    def _init_img_processor(self):
        # pdb.set_trace()
        model_h, model_w = self.evaluator.get_training_res()
        # if running on Waymo Dataset
        # model_h = 320
        # model_w = 480
        self.img_processor = ImageProcessor(
                self.args.trim, self.args.crop, model_h, model_w)

    def _init_data_reader(self):
        """Make an image iterator
        """
        self.data_reader = []
        fp = self.args.input_path
        fp_subs = glob(fp + '/*/', recursive=True)
        # fp_subs = glob(fp + '/', recursive=True)

        # pdb.set_trace()

        for sub in fp_subs:
            dr = SeqReader(sub)
            self.data_reader.append(dr)

    def infer(self):
        for idx in tqdm(range(len(self.data_reader))):
            dr = self.data_reader[idx]
            # pdb.set_trace()

            # file_path = dr.file_path
            file_path = dr.file_path[27:]   # /aachen_1/

            # if 'aachen_1/' not in file_path:
            #    continue
            
            # out_dir = self.args.output_dir + '/' + file_path
            # out_dir = self.args.output_dir + '/CityScapes/' + str(idx)
            out_dir = self.args.output_dir + '/test_xyz' + file_path

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            frame_results = {}
            fps = dr.get_fps()
            # pdb.set_trace()
            # pdb.set_trace()
            for idx, imgs in enumerate(dr):
                tgt_img, tgt_img_with_raw_ar = self.img_processor.process(imgs[0])
                src_img, src_img_with_raw_ar = self.img_processor.process(imgs[1])
                disp_colormap, depth = self.evaluator.estimate_depth(tgt_img)
                t2s_trans, t2s_res_trans = self.evaluator.estimate_motion(tgt_img, src_img)
                s2t_trans, s2t_res_trans = self.evaluator.estimate_motion(src_img, tgt_img)
                '''
                print('----------------------------------------------------\n')
                print('t=>s ego trans\n', t2s_trans)
                print('t=>s res trans, max %.5f, min %.5f' 
                    % (t2s_res_trans[:,DIM,:,:].max(), t2s_res_trans[:,DIM,:,:].min()))
                print('s=>t ego trans\n', s2t_trans)
                print('s=>t res trans, max %.5f, min %.5f' 
                    % (s2t_res_trans[:,DIM,:,:].max(), s2t_res_trans[:,DIM,:,:].min()))
                '''
                # pdb.set_trace()
                #  plt.close("all"); plt.figure(9); plt.imshow(disp_colormap); plt.colorbar(); plt.show()

                # disp_img = concat_depth_img(disp_colormap, tgt_img_with_raw_ar,
                #                             self.args.crop[2])
                # disp_img_bgr = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
                # cv2.imshow('disp_img', disp_img_bgr)
                # if fps is not None:
                #     cv2.waitKey(int(1000/fps))
                # else:
                #     # use fps 10 if not available
                #     fps = 10
                #     cv2.waitKey(int(1000/fps))
                # cv2.imwrite(
                #         os.path.join(self.args.output_dir, f'{idx:010d}.png'),
                #         disp_img_bgr
                #         )
                save_3dim(
                    os.path.join(out_dir,  f'{idx:010d}_plots.png'),
                    t2s_trans,
                    t2s_res_trans,
                    tgt_img_with_raw_ar,
                    disp_colormap
                )

if __name__ == '__main__':
    depth_estimator = DepthInference(InferOptions().parse()[0])
    depth_estimator.infer()
