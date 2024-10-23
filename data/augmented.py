import os
import cv2
import numpy as np
import random
from shutil import copyfile
from os import remove

# 输入文件夹路径和输出文件夹路径
input_folder_path = ''
output_folder_path = ''

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder_path, exist_ok=True)

# 获取文件夹下所有文件
for filename in os.listdir(input_folder_path):
        if filename.endswith('.JPG') or filename.endswith('.jpg'):
            # 读取图像
            image_path = os.path.join(input_folder_path, filename)
            image = cv2.imread(image_path)
            
            # 复制原始图像并重命名
            base_name, ext = os.path.splitext(filename)
            original_output_path = os.path.join(output_folder_path, f"{base_name}_0{ext}")
            copyfile(image_path, original_output_path)

             # 删除原始图像
            remove(image_path)
            
            for i in range(3):
                
                if random.random() < 0.5:
                    # 添加噪点（这里是添加高斯噪点，可以根据需求选择其他噪点类型）
                    noise = np.random.normal(0, np.random.randint(10, 100), image.shape)
                    noisy_image = image + noise

                    # 将像素值限制在 0 到 255 之间
                    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                    
                else:
                    noisy_image = image
                    
                # 随机选择是否进行模糊增强
                if random.random() < 0.8:
                    
                    # 随机选择模糊类型
                    blur_types = ['gaussian', 'median', 'average']
                    selected_blur = np.random.choice(blur_types)

                    # 根据选择的模糊类型应用相应的模糊操作
                    if selected_blur == 'gaussian':
                        kernel_size = (np.random.randint(1, 3) * 2 + 1, np.random.randint(1, 3) * 2 + 1)  # 随机选择高斯核大小
                        blurred_image = cv2.GaussianBlur(noisy_image, kernel_size, 0)
                    elif selected_blur == 'median':
                        ksize = np.random.randint(1, 3) * 2 + 1  # 随机选择中值滤波核大小
                        blurred_image = cv2.medianBlur(noisy_image, ksize)
                    elif selected_blur == 'average':
                        kernel_size = (np.random.randint(1, 3) * 2 + 1, np.random.randint(1, 3) * 2 + 1)  # 随机选择均值滤波核大小
                        blurred_image = cv2.blur(noisy_image, kernel_size)
                else:
                    blurred_image = noisy_image
                    
                # 随机调整亮度
                brightness_factor = np.random.uniform(0.5, 1.5)  # 随机生成亮度因子
                brightened_image = np.clip(blurred_image * brightness_factor, 0, 255).astype(np.uint8)

                # 构建新的文件名，在原文件名后面加上"_darkened"
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{base_name}_{i+1}{ext}"

                # 保存处理后的图像到输出文件夹
                output_path = os.path.join(output_folder_path, new_filename)
                cv2.imwrite(output_path, brightened_image)
        

print("图像处理完成，保存到", output_folder_path)
