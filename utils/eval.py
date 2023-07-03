import numpy as np
import random
import cv2
import glob
import os
import time
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import class_metrics

class EvalCocoDataset(Dataset):
    def __init__(self, dataset_dir, subset, year="2014",
                 patch_size=128, rho=32, img_h=240, img_w=320,random_rho=True,input_channels=3):
        super(EvalCocoDataset, self).__init__()
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        self.fnames = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.patch_size = patch_size
        self.rho = rho
        self.img_h = img_h
        self.img_w = img_w
        self.input_channels=input_channels
        self.random_rho=random_rho
    def __len__(self):
        return len(self.fnames)

    def cal_mask(self, H, image, top_left_point, bottom_right_point):
        mask_ori = np.zeros([image.shape[0], image.shape[1]])
        mask_ori[top_left_point[1]:bottom_right_point[1],
        top_left_point[0]:bottom_right_point[0]] = 1
        mask_pert = cv2.warpPerspective(np.float32(mask_ori), H, (image.shape[1], image.shape[0]))

        mask = mask_pert.astype(np.bool_) & mask_ori.astype(np.bool_)
        mask_trans = cv2.warpPerspective(np.float32(mask), np.linalg.inv(H), (image.shape[1], image.shape[0]))
        mask = mask[top_left_point[1]:bottom_right_point[1],
               top_left_point[0]:bottom_right_point[0]].astype(np.float32)
        mask_trans = mask_trans[top_left_point[1]:bottom_right_point[1],
                     top_left_point[0]:bottom_right_point[0]]
        mask_stack = np.stack((mask, mask_trans), axis=0)

        return mask_stack
    def gaussian_noise(self,image_pair,mask):
        noise_mask=1-mask
        noise = np.random.normal(0.3, 0.1, noise_mask.shape)
        gaussian_out = image_pair + noise * noise_mask
        gaussian_out = np.clip(gaussian_out, 0, 1)
        return gaussian_out
    def __getitem__(self, index):

        if self.input_channels == 3:
            image = cv2.imread(self.fnames[index], 1)
            image = cv2.resize(image, (self.img_w, self.img_h))
            height, width, _ = image.shape
        else:

            image = cv2.imread(self.fnames[index], 0)
            image = cv2.resize(image, (self.img_w, self.img_h))
            height, width = image.shape

        # create random point P within appropriate bounds
        y = random.randint(self.rho, height - self.rho - self.patch_size)
        x = random.randint(self.rho, width - self.rho - self.patch_size)
        # define corners of image patch
        top_left_point = (x, y)
        bottom_left_point = (x, self.patch_size + y)
        bottom_right_point = (self.patch_size + x, self.patch_size + y)
        top_right_point = (x + self.patch_size, y)

        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        perturbed_four_points = []
        if self.random_rho:
            for point in four_points:
                perturbed_four_points.append((point[0] + random.randint(-self.rho, self.rho),
                                              point[1] + random.randint(-self.rho, self.rho)))
        else:
            rho=np.random.randint(-1,2,[4,2])*self.rho
            for point,perturb in zip(four_points,rho):
                perturbed_four_points.append((point[0] + perturb[0],
                                              point[1] + perturb[1]))
        y_grid, x_grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

        # Two branches. The CNN try to learn the H and inv(H) at the same time. So in the first branch, we just compute the
        #  homography H from the original image to a perturbed image. In the second branch, we just compute the inv(H)
        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        try:
            H_inverse = np.linalg.inv(H)
        except:
            # either matrix could not be solved or inverted
            # this will show up as None, so use safe_collate in train.py
            return

        warped_image = cv2.warpPerspective(image, H_inverse, (image.shape[1], image.shape[0]))

        img_patch_ori = image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
        img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1],
                         top_left_point[0]:bottom_right_point[0]]

        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float32), H).squeeze()
        diff_branch1 = point_transformed_branch1 - point
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((image.shape[0], image.shape[1]))
        diff_y_branch1 = diff_y_branch1.reshape((image.shape[0], image.shape[1]))

        pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch = np.zeros((2, self.patch_size, self.patch_size))
        pf_patch[0, :, :] = pf_patch_x_branch1
        pf_patch[1, :, :] = pf_patch_y_branch1

        img_patch_ori = img_patch_ori / 255
        img_patch_pert = img_patch_pert / 255
        image_patch_pair = np.zeros((self.input_channels * 2, self.patch_size, self.patch_size))
        image_patch_pair[0:self.input_channels, :, :] = img_patch_ori.transpose(2, 0, 1)
        image_patch_pair[self.input_channels:, :, :] = img_patch_pert.transpose(2, 0, 1)

        base_four_points = np.asarray([x, y,
                                       x, self.patch_size + y,
                                       self.patch_size + x, self.patch_size + y,
                                       x + self.patch_size, y])

        perturbed_four_points = np.asarray(perturbed_four_points)
        perturbed_base_four_points = np.asarray([perturbed_four_points[0, 0], perturbed_four_points[0, 1],
                                                 perturbed_four_points[1, 0], perturbed_four_points[1, 1],
                                                 perturbed_four_points[2, 0], perturbed_four_points[2, 1],
                                                 perturbed_four_points[3, 0], perturbed_four_points[3, 1]])
        mask=self.cal_mask(H,image,top_left_point,bottom_right_point)
        #image_patch_pair = self.gaussian_noise(image_patch_pair, mask)
        return image_patch_pair, pf_patch, base_four_points, perturbed_base_four_points, image, warped_image, H,mask


def plotMace(maceList):
    fig, ax = plt.subplots()
    ax.plot(np.sort(maceList), np.linspace(0.0, 1.0, len(maceList)))
    plt.xlim(left=1e-2)
    plt.xlim(right=1e2)
    plt.ylim([0, 1])
    plt.xscale("log")
    plt.xlabel(r'Corner Error')
    plt.ylabel(r'Fraction of the number of images')
    plt.grid(True, which='both', ls='--')
    plt.show()
def evaluate_PFNet(model, val_loader, limit,
                   device,visual=False,timing=False,
                   macePath="Eval-mace.npy",normalize_factor=1,
                   mask_metrics=False,mask_mode=0,macePlot=False):
    t_start = time.time()
    print("rho:{%d}",{val_loader.dataset.rho})
    with tqdm(total=limit,desc="Evaluate") as pdar:
        with torch.no_grad():
            total_mace = []
            total_PA =[]
            for i, batch_value in enumerate(val_loader):
                X = batch_value[0].float().to(device)
                Y_true = batch_value[1].numpy()
                base_four_points = batch_value[2].numpy()
                perturbed_base_four_points = batch_value[3].numpy()
                mask_gt=batch_value[7][:,0,:,:].numpy()
                Y_pred = model(X)
                #mask_pred = F.softmax(Y_pred[1], dim=1).argmax(dim=1).cpu()
                mask_pred = F.softmax(Y_pred[1][:,0:2,:,:], dim=1).argmax(dim=1).cpu()
                # mask_pred = (torch.sigmoid(Y_pred[1])>=0.5).cpu().float()
                if mask_mode==0:
                    PFmask=np.ones([Y_true.shape[0],Y_true.shape[-2],Y_true.shape[-1]])
                elif mask_mode==1:
                    PFmask=mask_gt
                elif mask_mode==2:
                    PFmask=mask_pred.numpy()
                mace_, H_predicted = metric_paf(Y_true, Y_pred[0].cpu().numpy()*normalize_factor, base_four_points, perturbed_base_four_points,PFmask)
                PA=class_metrics(mask_pred,mask_gt)["Recall"][0]
                total_mace.append(mace_)
                total_PA.append(PA)
                pdar.update(1)
                pdar.set_postfix({"mace":mace_,"PA":PA,"mean_mace":np.mean(total_mace)})
                #print('i = {} / {}, mace = {:.4f}'.format(i + 1, limit, mace_))

                if visual and i%100==0:    # Save visualization images every 100 steps
                    result_dir = './results'
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    image = batch_value[4].numpy()[0]
                    warped_image = batch_value[5].numpy()[0]
                    gt_h = batch_value[6].numpy()[0]
                    pts1 = base_four_points[0].reshape(4,2)
                    pred_h = H_predicted[0]

                    gt_h_inv = np.linalg.inv(gt_h)
                    pts1_ = cv2.perspectiveTransform(np.float32([pts1]), gt_h_inv).squeeze()

                    pred_h_inv = np.linalg.inv(pred_h)
                    pred_pts1_ = cv2.perspectiveTransform(np.float32([pts1]), pred_h_inv).squeeze()

                    visual_file_name = ('%s' % i).zfill(4) + '.jpg'
                    utils.save_correspondences_img(image, warped_image, pts1, pts1_, pred_pts1_, mace=mace_,
                                                   result_name=os.path.join(result_dir, visual_file_name))

                if (i + 1) == limit:
                    break
    np.save(macePath, total_mace)
    if macePlot:
        plotMace(total_mace)
    final_mace = np.mean(total_mace)
    final_PA = np.mean(total_PA)
    if timing:
        t_prediction = (time.time() - t_start)
        print("Prediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / limit))
        print("Total time: ", time.time() - t_start)
    print("Evaluate MACE Metric: ", final_mace)
    print("Evaluate PA Metric: ", final_PA)


def metric_paf(Y_true, Y_pred, base_four_points, perturbed_base_four_points,
               mask_gt=None):
    # Compute the True H using Y_true
    assert (Y_true.shape == Y_pred.shape), "the shape of gt and pred should be the same"
    if mask_gt is None:
        mask_gt=np.ones([Y_true.shape[0],Y_true.shape[-2],Y_true.shape[-1]])
    batch_size = Y_true.shape[0]

    mace_b = []
    batch_H_predicted = np.zeros((batch_size, 3, 3))

    for i in range(batch_size):
        mask_true_in_loop = mask_gt[i, :, :]
        Y_true_in_loop = Y_true[i, :, :, :]*mask_true_in_loop
        Y_pred_in_loop = Y_pred[i, :, :, :]*mask_true_in_loop
        base_four_points_in_loop = base_four_points[i, :]
        perturbed_base_four_points_in_loop = perturbed_base_four_points[i, :]

        # define corners of image patch
        top_left_point = (base_four_points_in_loop[0], base_four_points_in_loop[1])
        bottom_left_point = (base_four_points_in_loop[2], base_four_points_in_loop[3])
        bottom_right_point = (base_four_points_in_loop[4], base_four_points_in_loop[5])
        top_right_point = (base_four_points_in_loop[6], base_four_points_in_loop[7])
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        perturbed_top_left_point = (perturbed_base_four_points_in_loop[0], perturbed_base_four_points_in_loop[1])
        perturbed_bottom_left_point = (perturbed_base_four_points_in_loop[2], perturbed_base_four_points_in_loop[3])
        perturbed_bottom_right_point = (perturbed_base_four_points_in_loop[4], perturbed_base_four_points_in_loop[5])
        perturbed_top_right_point = (perturbed_base_four_points_in_loop[6], perturbed_base_four_points_in_loop[7])
        perturbed_four_points = [perturbed_top_left_point, perturbed_bottom_left_point, perturbed_bottom_right_point,
                                 perturbed_top_right_point]


        predicted_pf_x1 = Y_pred_in_loop[0, :, :]
        predicted_pf_y1 = Y_pred_in_loop[1, :, :]

        pf_x1_img_coord = predicted_pf_x1
        pf_y1_img_coord = predicted_pf_y1
        y_patch_grid, x_patch_grid = np.mgrid[0:Y_true.shape[-2], 0:Y_true.shape[-1]]

        patch_coord_x = (x_patch_grid + top_left_point[0])*mask_true_in_loop
        patch_coord_y = (y_patch_grid + top_left_point[1])*mask_true_in_loop

        points_branch1 = np.vstack((patch_coord_x.flatten(), patch_coord_y.flatten())).transpose()
        mapped_points_branch1 = points_branch1 + np.vstack(
            (pf_x1_img_coord.flatten(), pf_y1_img_coord.flatten())).transpose()

        original_points = np.vstack((points_branch1))
        mapped_points = np.vstack((mapped_points_branch1))

        original_points=np.delete(original_points, original_points.sum(axis=1) == 0, axis=0)
        mapped_points=np.delete(mapped_points, mapped_points.sum(axis=1) == 0, axis=0)
        assert mapped_points.shape == original_points.shape
        # H_predicted = cv2.findHomography(np.float32(original_points), np.float32(mapped_points), cv2.RANSAC, 10)[0]
        H_predicted = cv2.findHomography(np.float32(original_points), np.float32(mapped_points), cv2.RANSAC, 10)[0]

        predicted_delta_four_point = cv2.perspectiveTransform(np.asarray([four_points], dtype=np.float32),
                                                              H_predicted).squeeze() - np.asarray(perturbed_four_points)

        result = np.mean(np.linalg.norm(predicted_delta_four_point, axis=1))
        mace_b.append(result)
        batch_H_predicted[i, :, :] = H_predicted

    __mace = np.mean(mace_b)
    return __mace, batch_H_predicted