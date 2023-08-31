import os

def rename_images(directory):
    print("开始重命名...")
    for root, dirs, files in os.walk(directory):
        count = 1
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                old_name = os.path.join(root, file)
                new_name = os.path.join(root, str(count) + ".png")
                os.rename(old_name, new_name)
                count += 1
                

# 指定目录路径
directory_path = r'e:\users\XieZF02\Desktop\data_all\weiying'

# 调用函数进行文件重命名
rename_images(directory_path)
