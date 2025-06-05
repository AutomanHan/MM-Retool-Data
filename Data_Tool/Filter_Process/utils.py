from PIL import Image
import imagehash

# 计算pHash
def calculate_phash(image_path):
    try:
        img = Image.open(image_path)
        phash = imagehash.phash(img)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    return str(phash)

# # 使用示例
# phash_value = calculate_phash("example.jpg")

