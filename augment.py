'''

图像的预增强，并非训练时的数据增强，仅仅只是为了补齐数量较少的分类，解决数据集的不均衡的问题

'''
import os
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import cv2

# 把需要增强的类的文件夹放到这个目录底下，进行增强后移回原位
img_dir = r"e:\users\XieZF02\Desktop\augment\aug"  # 输入目录
# 设置增强方法
aug = iaa.Sequential([iaa.Crop(percent=(0, 0.1)),
                      # iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
                      # iaa.Add((-20, 20)),
                      # iaa.Affine(rotate=(-20, 20))
                    ],
                      random_order=True
                    )


# 遍历子目录
for subdir in os.listdir(img_dir):
    print('开始增强'+str(subdir))
    subdir_path = os.path.join(img_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue
    
    for img_name in os.listdir(subdir_path):
        img_path = os.path.join(subdir_path, img_name)
        img = cv2.imread(img_path)
        #print(img)
        for i in range(1, 10):
            img_aug = aug.augment_image(img)  # 进行图像增强

            save_name = f"{img_name.split('.')[0]}_aug{i}.png"  # 新文件名，添加增强次数的后缀
            save_path = os.path.join(subdir_path, save_name)
            cv2.imwrite(save_path, img_aug)
print("图像增强完成！")

