'''

一个尝试解决脏污误判的方法，已弃用

'''


import cv2

def detect_dirty(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用Canny边缘检测算法
    edges = cv2.Canny(gray, 100, 200)  # 根据实际情况调整阈值
    
    # 计算边缘像素点的数量
    edge_pixels = cv2.countNonZero(edges)
    
    # 根据阈值确定是否为脏污图像
    dirty_threshold = 1000  # 根据实际情况调整阈值
    
    if edge_pixels >= dirty_threshold:
        return True
    else:
        return False
