import cv2
import numpy as np

def adjust_white_balance_opencv(image_path, target_color_temp=6500):
    """
    指定されたパスの画像ファイルのホワイトバランスを色温度に調整し、
    OpenCVのイメージデータオブジェクトとして返す関数
    
    Parameters:
    - image_path: 調整対象の画像ファイルのパス
    - target_color_temp: 目標の色温度（デフォルトは6500K）
    
    Returns:
    - OpenCVのイメージデータオブジェクト(cv2.Image)
    """
    # 画像を読み込む
    original_image = cv2.imread(image_path)
    
    # BGRからRGBに変換
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # ホワイトバランス調整
    adjusted_rgb = adjust_white_balance(rgb_image, target_color_temp=target_color_temp)
    
    # RGBからBGRに変換してOpenCVのイメージデータオブジェクトとして返す
    adjusted_bgr = cv2.cvtColor(adjusted_rgb, cv2.COLOR_RGB2BGR)
    
    return adjusted_bgr

# 例として、画像ファイルのパスを指定
image_path = 'path/to/your/image.jpg'

# ホワイトバランスを6500Kに調整
adjusted_image = adjust_white_balance_opencv(image_path, target_color_temp=6500)

# 修正後の画像データを表示
cv2.imshow('Adjusted Image', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
