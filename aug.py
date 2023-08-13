import imageio
from imgaug import augmenters as iaa
import os

def augment_image(image_path, output_path):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist!")
        return

    # 读取图像
    image = imageio.imread(image_path)

    # 定义一个图像增强序列
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻转图像 (50%的概率)
        iaa.Crop(percent=(0, 0.1)),  # 随机裁剪图像 (0-10%)
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),  # 随机施加高斯模糊
        iaa.ContrastNormalization((0.75, 1.5)),  # 改变对比度
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),  # 添加噪声
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # 改变亮度
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 缩放图像 (80-120%)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移图像
            rotate=(-25, 25),  # 旋转图像 (-25至25度)
            shear=(-8, 8)  # 剪切图像 (-8至8度)
        )
    ], random_order=True)  # 随机应用增强

    # 对图像进行增强
    aug_image = seq.augment_image(image)
    
    # 保存增强后的图像
    try:
        imageio.imwrite(output_path, aug_image)
        print(f"Augmented image saved to {output_path}")
    except Exception as e:
        print(f"Error while saving the image: {e}")
if __name__ == "__main__":
    # 对 './A/1.png' 图像进行增强，并保存到 './A/1_aug.png'
    augment_image(r"D:\桌面\data_car\img_new\images_original_1.png_1ed4e631-7222-432d-aefa-fbb6a34b468f.png", r"D:\桌面\data_car\img_new\1_aug.png")
