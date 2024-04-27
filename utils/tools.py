import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import Tensor


def find_contours(mask: Tensor):
    h,w = mask.shape
    if isinstance(mask, Tensor): 
        mask = mask.numpy().astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    assert len(contours) == 1
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return edge


def find_contour_points(mask: Tensor):
    '''
        mask: (h,w), 0 or 1
        return: contours (n,2)
                the x,y of the points 
    '''
    h,w = mask.shape
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy().astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # assert len(contours) == 1
    if len(contours) != 1:
        return np.array([])
    
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return contours


def hausdorff_distance(mask1: Tensor, mask2: Tensor, percentile: int = 95):
    if isinstance(mask1, torch.Tensor) and mask1.device == 'cuda':
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()

    contours1 = find_contour_points(mask1)
    contours2 = find_contour_points(mask2)
    if contours1.size == 0 or contours2.size == 0:
        return 0

    dist = cdist(contours1, contours2)
    dist = np.concatenate((np.min(dist, axis=0), np.min(dist, axis=1)))
    assert percentile >= 0 and percentile <= 100, 'percentile invaild'
    hausdorff_dist = np.percentile(dist, percentile)

    return hausdorff_dist

def draw_sem_seg_by_cv2_sum(mask: Tensor, color_map: np.ndarray):
    '''
    根据给定的掩码和颜色映射，使用OpenCV绘制语义分割图像。

    参数:
        mask: 一个Tensor，表示掩码图像，尺寸为(h,w)，其中h和w分别为高度和宽度。
        color_map: 一个numpy数组，表示颜色映射，尺寸为(n,3)，其中n为类别数量，每个类别对应一个RGB颜色。

    返回值:
        一个numpy数组，表示绘制完成的语义分割图像，尺寸为(h,w,3)，使用RGB格式。

    主要步骤:
        1. 获取掩码的尺寸。
        2. 通过find_contours函数寻找掩码中的轮廓。
        3. 通过find_contour_points函数为这些轮廓找到点集。
        4. 将点集转换为适合cv2.fillPoly函数的格式。
        5. 初始化一个全黑的图像。
        6. 使用cv2.fillPoly函数根据轮廓和对应的颜色映射填充图像。
    '''
    h,w = mask.shape  # 获取掩码的尺寸
    mask = find_contours(mask)  # 寻找掩码中的轮廓
    mask = find_contour_points(mask)  # 为这些轮廓找到点集
    mask = mask.astype(np.int32)  # 确保点集的数据类型为整型
    mask = mask.reshape(-1,1,2)  # 调整点集的形状以适合cv2.fillPoly的输入格式
    img = np.zeros((h,w,3), dtype=np.uint8)  # 初始化一个全黑的RGB图像
    cv2.fillPoly(img, [mask], color_map[mask[:,0]])  # 使用轮廓和颜色映射填充图像
    return img  # 返回绘制完成的语义分割图像


def corr(mask1: Tensor, mask2: Tensor) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Args:
    - mask1 (Tensor): A binary tensor mask.
    - mask2 (Tensor): Another binary tensor mask of the same size as mask1.

    Returns:
    - iou_score (float): The IoU score between the two masks, ranging from 0 to 1.
    """
    # Ensure both masks are binary and of the same shape
    assert mask1.shape == mask2.shape, "Input masks must have the same dimensions."
    assert torch.all(torch.logical_or(mask1 == 0, mask1 == 1)), "Mask1 should only contain binary values."
    assert torch.all(torch.logical_or(mask2 == 0, mask2 == 1)), "Mask2 should only contain binary values."

    # Compute intersection and union
    intersection = torch.logical_and(mask1, mask2).sum().item()
    union = torch.logical_or(mask1, mask2).sum().item()

    # Handle the case where both masks are entirely zero (no true positives or false positives)
    if union == 0:
        return 1 if intersection == 0 else 0  # If both are empty, they match perfectly (IoU=1); otherwise, it's 0.

    # Compute IoU
    iou_score = intersection / union

    return iou_score

