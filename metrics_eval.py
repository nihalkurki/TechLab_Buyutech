import os, cv2, tqdm
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt

def RMSE(im1, im2):
    im1, im2 = im1/1.0, im2/1.0
    squared_diff = (im1 - im2) ** 2
    # print("im1:",im1)
    # print("im2:",im2)
    # print("im1-im2:",np.abs(im1 - im2))
    # print("squared_diff:",squared_diff)
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    # print("mean_squared_diff", mean_squared_diff)
    # Calculate the root mean squared error
    rmse = np.sqrt(mean_squared_diff)

    return rmse

def evaluate(disparity, gt, psm_threshold=192, max_disparity=1e6):
  """Computes metrics for predicted disparity against GT.

  Computes:
    PSM EPE: average disparity error for pixels with less than psm_threshold GT
    disparity value.
    bad_X: percent of pixels with disparity error larger than X. The divisor is
    the number of pixels with valid GT in the image.

  Args:
    disparity: Predicted disparity.
    gt: GT disparity.
    psm_threshold: Disparity threshold to compute PSM EPE.
    max_disparity: Maximum valid GT disparity.

  Returns:
    An np array with example metrics.
    [psm_epe, bad_0.1, bad_0.5, b ad_1.0, bad_2.0, bad_3.0].
  """
  disparity, gt = disparity/1.0, gt/1.0
  gt_mask = np.where((gt > 0) & (gt < max_disparity), np.ones_like(gt),
                     np.zeros_like(gt))
  gt_diff = np.where(gt_mask > 0, np.abs(gt - disparity), np.zeros_like(gt))
  psm_mask = np.where(gt < psm_threshold, gt_mask, np.zeros_like(gt))
  gt_mask_count = np.sum(gt_mask) + 1e-5
  psm_mask_count = np.sum(psm_mask) + 1e-5
  bad01 = np.where(gt_diff > 0.1, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad05 = np.where(gt_diff > 0.5, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad1 = np.where(gt_diff > 1.0, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad2 = np.where(gt_diff > 2.0, np.ones_like(gt_diff), np.zeros_like(gt_diff))
  bad3 = np.where(gt_diff > 3.0, np.ones_like(gt_diff), np.zeros_like(gt_diff))

  bad01 = 100.0 * np.sum(bad01 * gt_mask) / gt_mask_count
  bad05 = 100.0 * np.sum(bad05 * gt_mask) / gt_mask_count
  bad1 = 100.0 * np.sum(bad1 * gt_mask) / gt_mask_count
  bad2 = 100.0 * np.sum(bad2 * gt_mask) / gt_mask_count
  bad3 = 100.0 * np.sum(bad3 * gt_mask) / gt_mask_count
  psm_epe = np.sum(gt_diff * psm_mask) / psm_mask_count
  return np.array([psm_epe, bad01, bad05, bad1, bad2, bad3])






if __name__ == "__main__":

    # dis_new = "/Users/evanwyf/Desktop/techlab/data/mini_test/eval_test.jpeg"
    # im_new = cv2.imread(dis_new, cv2.IMREAD_GRAYSCALE)
    # im_new = im_new/1.0
    # im_new[:im_new.shape[0]//2,:] += 2.0
    # print("im_new", im_new)
    # cv2.imwrite('/Users/evanwyf/Desktop/techlab/data/mini_test/eval_test_new.jpeg', im_new)

    dis_pred = "/Users/evanwyf/Desktop/techlab/data/mini_test/eval_test_new.jpeg"
    dis_gt = "/Users/evanwyf/Desktop/techlab/data/mini_test/eval_test.jpeg"


    im_bm = cv2.imread(dis_pred, cv2.IMREAD_GRAYSCALE)
    im_bm = im_bm / 1.0
    im_gt = cv2.imread(dis_gt, cv2.IMREAD_GRAYSCALE)
    im_gt = im_gt / 1.0
    

    print("im_pred",im_bm)
    print("im_gt",im_gt)
    # print("MAX",np.max(im_bm), np.max(im_gt))
    print(im_bm.shape, im_gt.shape)

    print("RMSE: ",RMSE(im_bm, im_gt))
    print("[psm_epe,    bad_0.1,    bad_0.5,    bad_1.0,    bad_2.0,    bad_3.0]:")
    print(evaluate(im_bm, im_gt))
    # root_dir = "/Users/evanwyf/Desktop/carla_data/non_learning_output"

    # all_f = [f[:-8] for f in os.listdir(osp.join(root_dir,"gt"))]

    # RMSE_bm = []
    # RMSE_sgbm = []
    # MAE_bm = []
    # MAE_sgbm = []
    # log_squre_bm = []
    # log_squre_sgbm = []
    # gradient_loss_bm = []
    # gradient_loss_sgbm = []
    # _3px_error_bm = []
    # _3px_error_sgbm = []
    # EPE_bm = []
    # EPE_sgbm = []
    # for f in tqdm.tqdm(all_f):
    #     bm_f = osp.join(root_dir,"bm",f+"_bm.jpg")
    #     sgbm_f = osp.join(root_dir,"sgbm",f+"_sgbm.jpg")
    #     gt_f = osp.join(root_dir,"gt",f+"_dep.jpg")

    #     im_bm = cv2.imread(bm_f, cv2.IMREAD_GRAYSCALE)
    #     im_sgbm = cv2.imread(sgbm_f, cv2.IMREAD_GRAYSCALE)
    #     im_gt = cv2.imread(gt_f, cv2.IMREAD_GRAYSCALE)

    #     assert im_bm.shape == im_sgbm.shape == im_gt.shape, "Images must have the same dimensions"

    #     # RMSE
    #     RMSE_bm.append(RMSE(im_bm, im_gt))
    #     RMSE_sgbm.append(RMSE(im_sgbm, im_gt))
    #     # MAE
    #     MAE_bm.append(MAE(im_bm, im_gt))
    #     MAE_sgbm.append(MAE(im_sgbm, im_gt))
    #     # log square error
    #     log_squre_bm.append(log_square_error(im_bm, im_gt))
    #     log_squre_sgbm.append(log_square_error(im_sgbm, im_gt))
    #     # gradient loss
    #     gradient_loss_bm.append(gradient_loss(im_bm, im_gt))
    #     gradient_loss_sgbm.append(gradient_loss(im_sgbm, im_gt))
    #     # 3px error
    #     _3px_error_bm.append(compute_3px_error(im_bm, im_gt)) 
    #     _3px_error_sgbm.append(compute_3px_error(im_sgbm, im_gt)) 
    #     # EPE
    #     EPE_bm.append(compute_epe(im_bm, im_gt))
    #     EPE_sgbm.append(compute_epe(im_sgbm, im_gt))
    #     # 

    
    # print(f"RMSE_bm = {np.sum(RMSE_bm)/len(RMSE_bm)}")
    # print(f"RMSE_sgbm = {np.sum(RMSE_sgbm)/len(RMSE_sgbm)}")
    # print(f"MAE_bm = {np.sum(MAE_bm)/len(MAE_bm)}")
    # print(f"MAE_sgbm = {np.sum(MAE_sgbm)/len(MAE_sgbm)}")
    # print(f"log_squre_bm = {np.sum(log_squre_bm)/len(log_squre_bm)}")
    # print(f"log_squre_sgbm = {np.sum(log_squre_sgbm)/len(log_squre_sgbm)}")
    # print(f"gradient_loss_bm = {np.sum(gradient_loss_bm)/len(gradient_loss_bm)}")
    # print(f"gradient_loss_sgbm = {np.sum(gradient_loss_sgbm)/len(gradient_loss_sgbm)}")
    # print(f"_3px_error_bm = {np.sum(_3px_error_bm)/len(_3px_error_bm)}")
    # print(f"_3px_error_sgbm = {np.sum(_3px_error_sgbm)/len(_3px_error_sgbm)}")
    # print(f"EPE_bm = {np.sum(EPE_bm)/len(EPE_bm)}")
    # print(f"EPE_sgbm = {np.sum(EPE_sgbm)/len(EPE_sgbm)}")

    # # plt.plot(np.arange(len(RMSE_bm)), RMSE_bm, label="bm")
    # # plt.plot(np.arange(len(RMSE_sgbm)), RMSE_sgbm, label="sgbm")
    # # plt.xlabel("data #")
    # # plt.ylabel("RMSE loss")
    # # plt.legend()
    # # plt.show()

    # # plt.plot(np.arange(len(MAE_bm)), MAE_bm, label="bm")
    # # plt.plot(np.arange(len(MAE_sgbm)), MAE_sgbm, label="sgbm")
    # # plt.xlabel("data #")
    # # plt.ylabel("MAE loss")
    # # plt.legend()
    # # plt.show()

    # # plt.plot(np.arange(len(log_squre_bm)), log_squre_bm, label="bm")
    # # plt.plot(np.arange(len(log_squre_sgbm)), log_squre_sgbm, label="sgbm")
    # # plt.xlabel("data #")
    # # plt.ylabel("Log Square Loss")
    # # plt.legend()
    # # plt.show()

    # # plt.plot(np.arange(len(gradient_loss_bm)), gradient_loss_bm, label="bm")
    # # plt.plot(np.arange(len(gradient_loss_sgbm)), gradient_loss_sgbm, label="sgbm")
    # # plt.xlabel("data #")
    # # plt.ylabel("Gradient loss")
    # # plt.legend()
    # # plt.show()
