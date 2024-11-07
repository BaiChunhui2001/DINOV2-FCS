from PIL import Image
import os
import glob

# 设置图片和标签所在的文件夹路径
image_folder = 'path_to_image_folder'
label_folder = 'path_to_label_folder'

# 设置目标尺寸
target_size = (512, 512)

# 读取所有图片文件
image_files = glob.glob(os.path.join(image_folder, '*.png'))
label_files = glob.glob(os.path.join(label_folder, '*.png'))

# 确保图片和标签数量匹配
assert len(image_files) == len(label_files), "图片和标签数量不匹配。"

# 遍历所有图片和标签
for image_path, label_path in zip(image_files, label_files):
    # 打开图片和标签
    image = Image.open(image_path)
    label = Image.open(label_path)

    # Resize图片和标签
    image_resized = image.resize(target_size, Image.ANTIALIAS)
    label_resized = label.resize(target_size, Image.NEAREST)

    # 将处理后的图片和标签保存回原来的路径，这将覆盖原始文件
    image_resized.save(image_path)
    label_resized.save(label_path)

    # 打印处理结果（可选）
    print(f'Processed and saved image: {image_path}')
    print(f'Processed and saved label: {label_path}')