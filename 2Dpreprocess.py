import cv2 as cv
import numpy as np
from PIL import Image,ImageEnhance
import os

def gaussian_blur_demo(a,image):
    dst = cv.GaussianBlur(image, (15,15), 0)
    cv.imwrite('./image/blur/'+str(a)+'_gaussian_blur.png', dst)
    return dst
def contrast(a, image):
    ImageObject = Image.fromarray(np.uint8(image))
    en = ImageEnhance.Contrast(ImageObject)
    en_end = en.enhance(3.5)
    en_end.save('./image/duibi/'+str(a)+'_duibi.png')
    imge = np.array(en_end)
    return imge

def three_color_split(a, binaryimage):
    contours, hierarchy = cv.findContours(binaryimage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mask1 = np.zeros((binaryimage.shape[0],binaryimage.shape[1], 3), np.uint8)
    max_areaa = 0
    max_contour = []
    for contour in contours:  # 遍历所有轮廓列表，其中每个元素是一个数组
        area = cv.contourArea(contour)  # 找到轮廓图面积最大的轮廓数组
        if(area > max_areaa):
            max_areaa = area
            max_contour = contour
    # print("max_area:",max_areaa)
    mask1[:, :, 0] = 5      #B
    mask1[:, :, 1] = 39     #G
    mask1[:, :, 2] = 175    #R
    #thickness，轮廓粗细越大越粗，-1代表cv.FILLED,填充轮廓内部
    cv.drawContours(mask1, [max_contour], -1, (0, 0, 0), -1)
    #mask1背景为彩色,填充内部为黑色，与二值图取余加，得到种子轮廓三色图
    #二值图转为三通道
    binary_three = cv.cvtColor(binaryimage, cv.COLOR_GRAY2RGB)
    # 得到种子、垩白、背景三色图
    ROI_threecolor = mask1 + binary_three
    cv.imwrite('./image/roi/'+str(a)+'_roi_threecolor.png', ROI_threecolor)
    return ROI_threecolor

def cir(img):   #输入为单通道
    c_t = [1006, 1060, 904]     #大致位置，不够准确
    c_tt = [1002, 1066, 813]
    mask3 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask3 = np.bitwise_not(mask3)
    mask4 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask4 = np.bitwise_not(mask4)
    cv.circle(mask3, (c_t[0], c_t[1]), c_t[2]+15, (0, 0, 0), -1)
    cv.circle(mask4, (c_tt[0], c_tt[1]), c_tt[2]-10, (0, 0, 0), -1)
    ring = cv.subtract(mask4, mask3)
    ring = cv.bitwise_not(ring)
    return ring
if __name__ == '__main__':
    file_pathname = './dataset'
    i = 0
    white_pixel = 0
    black_pixel = 0
    for filename in os.listdir(file_pathname):
        wh_pixel = 0
        bla_pixel = 0
        i = i + 1
        src = cv.imread(file_pathname+'/'+filename, 0)
        #step1:去掉圆形边框
        ring_mask = cir(src)
        img = cv.bitwise_and(ring_mask, src)
        cv.imwrite('./image/dataset/'+str(i)+'.png', img)
        #step2:对图像降噪滤波
        gaussian_blur_img = gaussian_blur_demo(i, img)
        #step3:对图像调整对比度
        imgg = contrast(i, gaussian_blur_img)
        #step4:二值化,自定义阈值ret，超过阈值显示为白色，低于该阈值显示为黑色
        ret, binary_image = cv.threshold(imgg, 205, 255, cv.THRESH_BINARY)
        cv.imwrite('./image/binary/'+str(i)+'_binary.png', binary_image)
        #step5:三色分割
        ROI_threecolor = three_color_split(i, binary_image)
        #step6:分别统计三色ROI图中黑色像素、白色像素的数量
        wh_pixel = np.count_nonzero(np.all(ROI_threecolor==[255,255,255],axis = 2))
        white_pixel = white_pixel + wh_pixel
        bla_pixel = np.count_nonzero(np.all(ROI_threecolor==[0,0,0],axis = 2))
        black_pixel = black_pixel + bla_pixel
    white_pixel = float(white_pixel)
    black_pixel = float(black_pixel)
    chalkiness_rate = black_pixel / white_pixel
    # print('chalkiness rate:{:.2f}%'.format(black_pixel/white_pixel*100))
    np.savetxt('./ChalkinessRate.txt', [chalkiness_rate])