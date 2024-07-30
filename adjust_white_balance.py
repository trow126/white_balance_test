import numpy as np
from matplotlib import pyplot as plt

def adjust_white_balance(image, target_color_temp=6500):
    """
    画像データのホワイトバランスを指定された色温度に調整する関数
    
    Parameters:
    - image: numpy.ndarray型の画像データ
    - target_color_temp: 目標の色温度（デフォルトは6500K）
    
    Returns:
    - 修正後の画像データ(numpy.ndarray)
    """
    # RGBカラースペースへの変換
    rgb = image / 255.0
    
    # CIE XYZカラースペースへの変換
    xyz = rgb_to_xyz(rgb)
    
    # 目標色温度に基づく白色点のXYZ値を計算
    target_xyz = color_temp_to_xyz(target_color_temp)
    
    # XYZからRGBへの逆変換
    adjusted_rgb = xyz_to_rgb(xyz * (target_xyz / xyz.mean(axis=(0, 1))))
    
    return adjusted_rgb * 255

def rgb_to_xyz(rgb):
    """
    RGBカラースペースをCIE XYZカラースペースへ変換する関数
    """
    # RGBからXYZへの変換行列
    matrix = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])
    
    # RGB -> XYZ
    xyz = np.dot(matrix, rgb)
    xyz[:, 0] += 16 / 116
    xyz[:, 1] += 16 / 116
    xyz[:, 2] += 16 / 116
    
    return xyz

def xyz_to_rgb(xyz):
    """
    CIE XYZカラースペースをRGBカラースペースへ変換する関数
    """
    # XYZからRGBへの逆変換行列
    matrix = np.linalg.inv(np.array([[3.2406, -0.9692, 0.0588],
                                     [-0.9689, 1.8765, -0.0415],
                                     [0.0557, -0.2040, 1.0570]]))
    
    # XYZ -> RGB
    rgb = np.dot(matrix, xyz)
    rgb[:, 0] /= 12.92
    rgb[:, 1] /= 12.92
    rgb[:, 2] /= 12.92
    
    return rgb.clip(0, 1)

def color_temp_to_xyz(color_temp):
    """
    色温度をCIE XYZカラースペースの白色点に変換する関数
    """
    # 色温度に基づく白色点のXYZ値を計算
    if color_temp < 5000:
        x = 0.4124564 * (color_temp / 2500) ** (-2/3)
        y = 0.3575761 * (color_temp / 2500) ** (-1/3)
        z = 0.1804375 * (color_temp / 2500) ** (-2/3)
    else:
        x = 0.30391225 * (color_temp / 100) ** (-1/3)
        y = 1.08313156 * (color_temp / 100) ** (-1/3)
        z = 0.44091839 * (color_temp / 100) ** (-1/3)
    
    return np.array([x, y, z])

# 例として、ランダムなRGB画像データを作成
image = np.random.rand(100, 100, 3).astype(np.float32)

# ホワイトバランスを6500Kに調整
adjusted_image = adjust_white_balance(image, target_color_temp=6500)

# 修正後の画像データを表示
plt.imshow(adjusted_image.astype(np.uint8))
plt.show()
