import re
import matplotlib.pyplot as plt

# 文件路径
# log_file_path = '/home/haida/d_j/Semantic-Aware-Low-Light-Image-Enhancement-main/DRBN_SKF/src/result_l_g/train_RRDB_P+SeD/train.log'
log_file_path = '/home/haida/d_j/Semantic-Aware-Low-Light-Image-Enhancement-main/DRBN_SKF/src/0.5_0.5_k_10_0.001/train_RRDB_P+SeD/train.log'


# 读取log文件
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# 定义正则表达式来匹配PSNR, SSIM, LPIPS-VGG, LPIPS-Alex的行
psnr_ssim_lpips_pattern = re.compile(r'.*?Step: (\d+).*?psnr: ([\d\.]+).*?ssim: ([\d\.]+).*?LPIPS-VGG: ([\d\.]+).*?LPIPS-Alex: ([\d\.]+)')
# psnr_ssim_pattern = re.compile(r'.*?Step: (\d+).*?psnr: ([\d\.]+).*?ssim: ([\d\.]+).*')

# 存储数据
steps = []
psnr_values = []

ssim_values = []
lpips_vgg_values = []
lpips_alex_values = []

# 解析log文件
for line in lines:
    match = psnr_ssim_lpips_pattern.match(line)
    if match:
        step = int(match.group(1))
        psnr_str = match.group(2).strip('.')

        ssim_str = match.group(3).strip('.')
        lpips_vgg_str = match.group(4).strip('.')
        lpips_alex_str = match.group(5).strip('.')

        try:
            psnr = float(psnr_str)
            ssim = float(ssim_str)
            lpips_vgg = float(lpips_vgg_str)
            lpips_alex = float(lpips_alex_str)
        except ValueError:
            continue

        steps.append(step)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

        lpips_vgg_values.append(lpips_vgg)
        lpips_alex_values.append(lpips_alex)

# 每2000个step取一个断点
filtered_steps = []
filtered_psnr_values = []
filtered_ssim_values = []

filtered_lpips_vgg_values = []
filtered_lpips_alex_values = []

# 添加调试信息，检查步骤和PSNR值
# print("Steps:", steps)
# print("PSNR values:", psnr_values)




for i in range(len(steps)):
    if steps[i] % 500 == 0:
        filtered_steps.append(steps[i])
        filtered_psnr_values.append(psnr_values[i])
        filtered_ssim_values.append(ssim_values[i])
        filtered_lpips_vgg_values.append(lpips_vgg_values[i])
        filtered_lpips_alex_values.append(lpips_alex_values[i])


min_lpips_alex = min(filtered_lpips_alex_values)
min_lpips_alex_step = filtered_steps[filtered_lpips_alex_values.index(min_lpips_alex)]
min_lpips_psnr = filtered_psnr_values[filtered_lpips_alex_values.index(min_lpips_alex)]
min_lpips_ssim = filtered_ssim_values[filtered_lpips_alex_values.index(min_lpips_alex)]
min_lpips_vgg = filtered_lpips_vgg_values[filtered_lpips_alex_values.index(min_lpips_alex)]




# 可视化
plt.figure(figsize=(10, 5))
# plt.plot(filtered_steps, filtered_psnr_values, 'b-', label='PSNR')
# plt.plot(filtered_steps, filtered_ssim_values, 'r-', label='SSIM')
# plt.plot(filtered_steps, filtered_lpips_vgg_values, 'g-', label='LPIPS-VGG')
plt.plot(filtered_steps, filtered_lpips_alex_values, 'm-', label='LPIPS-Alex')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('PSNR, LPIPS-VGG, LPIPS-Alex over Steps')
plt.legend()
plt.grid(True)

# 标记最高PSNR值
# plt.annotate(f'Max PSNR: {max_psnr:.2f}', xy=(max_psnr_step, max_psnr),
#              xytext=(max_psnr_step, max_psnr + 1),
#              arrowprops=dict(facecolor='blue', shrink=0.05),
#              horizontalalignment='right')

# plt.annotate(f'Max SSIM: {max_ssim:.2f}', xy=(max_ssim_step, max_ssim),
#              xytext=(max_ssim_step, max_ssim + 1),
#              arrowprops=dict(facecolor='red', shrink=0.05),
#              horizontalalignment='right')

# 标记最低LPIPS-VGG值
# plt.annotate(f'Min LPIPS-VGG: {min_lpips_vgg:.4f}', xy=(min_lpips_vgg_step, min_lpips_vgg),
#              xytext=(min_lpips_vgg_step, min_lpips_vgg + 0.005),
#              arrowprops=dict(facecolor='green', shrink=0.05),
#              horizontalalignment='right')

# 标记最低LPIPS-Alex值
plt.annotate(f'Min LPIPS-Alex: {min_lpips_alex:.4f}', xy=(min_lpips_alex_step, min_lpips_alex),
             xytext=(min_lpips_alex_step, min_lpips_alex + 0.005),
             arrowprops=dict(facecolor='magenta', shrink=0.05),
             horizontalalignment='right')

# 在图中用不同颜色的字体显示最高PSNR值、最低LPIPS-VGG值和最低LPIPS-Alex值
# plt.text(max_psnr_step, max_psnr, f'Step: {max_psnr_step}\nPSNR: {max_psnr:.2f}', SSIM: {max_psnr_ssim:.4f}',
#          fontsize=12, color='blue', ha='center')
# plt.text(max_psnr_step, max_psnr, f'Step: {max_psnr_step}\nPSNR: {max_psnr:.2f}, SSIM: {max_psnr_ssim:.4f}',
#          fontsize=12, color='blue', ha='center')
# plt.text(max_ssim_step, max_ssim, f'Step: {max_ssim_step}\nSSIM: {max_ssim:.2f}',
#          fontsize=12, color='red', ha='center')
# plt.text(min_lpips_vgg_step, min_lpips_vgg, f'Step: {min_lpips_vgg_step}\nLPIPS-VGG: {min_lpips_vgg:.4f}',
#          fontsize=12, color='green', ha='center')
plt.text(min_lpips_alex_step, min_lpips_alex, f'Step: {min_lpips_alex_step}\nLPIPS-Alex: {min_lpips_alex:.4f}, LPIPS-PSNR: {min_lpips_psnr:.4f}, LPIPS-SSIM: {min_lpips_ssim:.4f},  LPIPS-VGG: {min_lpips_vgg:.4f}',
         fontsize=12, color='magenta', ha='center')

plt.show()
