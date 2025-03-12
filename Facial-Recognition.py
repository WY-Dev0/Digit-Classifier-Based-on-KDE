# 安装 pymetalog 包
!pip install pymetalog

# 导入所需模块
from pymetalog import metalog


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neighbors import KernelDensity


# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化像素值到0到1之间
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 输出数据集的形状
print("训练集形状:", x_train.shape, y_train.shape)
print("测试集形状:", x_test.shape, y_test.shape)

# 可视化前10个训练图像
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()




# 定义过滤无意义像素的函数
def filter_pixels(x, threshold=0.1):
    # 根据所有样本中某像素的平均值过滤无意义像素
    mask = np.mean(x, axis=0) > threshold
    return mask

# 获取过滤后的像素位置
pixel_mask = filter_pixels(x_train)

# 将训练和测试数据展平为二维数组，仅保留有效像素
x_train_flat = x_train[:, pixel_mask]
x_test_flat = x_test[:, pixel_mask]

# 创建 Naive Bayes 模型并训练
model = GaussianNB()
model.fit(x_train_flat, y_train)

# 对测试集进行预测
y_pred = model.predict(x_test_flat)

# 创建混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()

# 高斯分布拟合（仅保留有效像素）
mean_values = np.zeros((10, np.sum(pixel_mask)))  # 每个类别的每个有效像素的均值
std_values = np.zeros((10, np.sum(pixel_mask)))   # 每个类别的每个有效像素的标准差

for i in range(10):
    # 获取当前类别的所有图像并应用过滤器
    class_images = x_train[y_train == i][:, pixel_mask]

    # 计算当前类别的每个有效像素的均值和标准差
    mean_values[i] = np.mean(class_images, axis=0)
    std_values[i] = np.std(class_images, axis=0)

print("高斯分布拟合完成：")
print("均值形状:", mean_values.shape)
print("标准差形状:", std_values.shape)

# 核密度估计（KDE）拟合（仅保留有效像素）
kde_models = {}

for i in range(10):
    # 获取当前类别的所有图像并展平为二维数据
    class_images = x_train[y_train == i][:, pixel_mask]

    # 为当前类别创建KDE模型，调整带宽参数（可根据数据分布优化）
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05)
    kde.fit(class_images)

    # 存储模型
    kde_models[i] = kde

print("核密度估计（KDE）拟合完成")

# 初始化存储MetaLog模型的字典
metalog_models = {}

for i in range(10):
    # 获取当前类别的所有图像并展平为一维数据（用于MetaLog拟合）
    class_images = x_train[y_train == i][:, pixel_mask].flatten()
    
    # 清理输入数据
    class_images = class_images[np.isfinite(class_images)]  # 去除 NaN 和 inf 值
    class_images = np.clip(class_images, 0, 1)             # 确保数据在 [0, 1] 范围内
    
    # 检查异常值
    print(f"类别 {i} 的数据统计:")
    print("最小值:", np.min(class_images))
    print("最大值:", np.max(class_images))
    
    # 创建 MetaLog 模型
    ml_model = metalog(
        x=class_images.tolist(),  # 转换为列表格式
        bounds=[0, 1],            # 设置边界范围为 [0, 1]
        boundedness='b',          # 指定为双边界
        term_limit=10             # 设置最大项数为10
    )
    
    # 存储模型
    metalog_models[i] = ml_model

print("MetaLog分布拟合完成")

