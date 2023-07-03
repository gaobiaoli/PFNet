import random
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
def safe_collate(batch):
    """Return batch without any None values"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class CocoDataset(Dataset):
    def __init__(self, dataset_dir, subset, year="2014",
                 patch_size=128, rho=32, img_h=240, img_w=320,input_channels=3
                 ,diffusion=None,noise_factor=0):
        super(CocoDataset, self).__init__()
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        self.fnames = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.patch_size = patch_size
        self.rho = rho
        self.img_h = img_h
        self.img_w = img_w
        self.input_channels=input_channels
        self.diffusion = diffusion
        self.noisefactor=noise_factor
        if self.diffusion is not None:
            self.rho=self.diffusion[0]
            self.diffusion.pop(0)
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
        mask_stack=np.stack((mask,mask_trans),axis=0)

        return mask_stack
    def gaussian_noise(self,image_pair,mask,loc=0.1,scale=0.1):
        noise_mask=1-mask
        noise = np.random.normal(loc=loc, scale=scale, size=noise_mask.shape)
        gaussian_out = image_pair + noise * noise_mask * self.noise_factor
        gaussian_out = np.clip(gaussian_out, 0, 1)
        return gaussian_out
    def __getitem__(self, index):
        if self.input_channels==3:
            image = cv2.imread(self.fnames[index], 1)
            image = cv2.resize(image, (self.img_w, self.img_h))
            height, width,_ = image.shape
        else:

            image = cv2.imread(self.fnames[index], 0)
            image = cv2.resize(image, (self.img_w, self.img_h))
            height, width = image.shape

        # create random point P within appropriate bounds
        y = random.randint(self.rho, height - self.rho - self.patch_size)
        x = random.randint(self.rho, width - self.rho - self.patch_size)
        # define corners of image patch
        top_left_point = (x, y)
        bottom_left_point = (x, self.patch_size + y-1)
        bottom_right_point = (self.patch_size + x -1, self.patch_size + y-1)
        top_right_point = (x + self.patch_size-1, y)

        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-self.rho, self.rho),
                                          point[1] + random.randint(-self.rho, self.rho)))

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

        img_patch_ori = image[top_left_point[1]:bottom_right_point[1],
                              top_left_point[0]:bottom_right_point[0]]
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
        image_patch_pair = np.zeros((self.input_channels*2, self.patch_size, self.patch_size))
        image_patch_pair[0:self.input_channels, :, :] = img_patch_ori.transpose(2,0,1)
        image_patch_pair[self.input_channels:, :, :] = img_patch_pert.transpose(2,0,1)
        mask=self.cal_mask(H,image,top_left_point,bottom_right_point)
        #image_patch_pair=self.gaussian_noise(image_patch_pair,mask)
        return image_patch_pair, pf_patch,mask

    def step(self):
        if self.diffusion is not None and len(self.diffusion)!=0:
            self.rho=self.diffusion[0]
            self.diffusion.pop(0)


def datasetPreprocess(dataset,savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for i in tqdm(range(len(dataset))):
        filename=os.path.basename(dataset.fnames[i]).split(".")[0]
        np.save(os.path.join(savedir,filename+".npy"),dataset[i])
    pass
if __name__=="__main__":
    train_dataset = CocoDataset("/CV/3T/dataset-public/COCO/coco2014", "train", year=2014)
    datasetPreprocess(train_dataset,"/CV/3T/dataset-public/COCO/coco2014/preprocessed")