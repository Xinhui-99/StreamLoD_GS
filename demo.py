import cv2
import os
import numpy as np
from pathlib import Path
import argparse

def images_to_video(input_dir, output_path, fps=30, codec='mp4v'):
    """
    将图像序列转换为MP4视频
    
    Args:
        input_dir: 输入图像文件夹路径
        output_path: 输出视频文件路径
        fps: 视频帧率 (默认30)
        codec: 视频编码器 (默认'mp4v')
    """
    
    # 确保输入目录存在
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return False
    
    # 获取所有PNG图像并按名称排序
    image_files = sorted(list(input_dir.glob('*.png')))
    
    if not image_files:
        print(f"错误: 在 {input_dir} 中没有找到PNG图像")
        return False
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 读取第一张图像以获取尺寸
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"错误: 无法读取第一张图像 {image_files[0]}")
        return False
    
    height, width, layers = first_image.shape
    print(f"图像尺寸: {width}x{height}")
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # 创建视频写入器
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {output_path}")
        return False
    
    # 逐帧写入视频
    print("正在生成视频...")
    for i, image_file in enumerate(image_files):
        # 显示进度
        if i % 10 == 0:
            print(f"处理进度: {i}/{len(image_files)} ({100*i/len(image_files):.1f}%)")
        
        # 读取图像
        frame = cv2.imread(str(image_file))
        
        if frame is None:
            print(f"警告: 无法读取图像 {image_file}，跳过")
            continue
        
        # 确保图像尺寸一致
        if frame.shape[:2] != (height, width):
            print(f"警告: 图像 {image_file} 尺寸不匹配，进行缩放")
            frame = cv2.resize(frame, (width, height))
        
        # 写入帧
        video_writer.write(frame)
    
    # 释放资源
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n视频已成功保存到: {output_path}")
    print(f"视频参数: {width}x{height}, {fps}fps, {len(image_files)} 帧")
    print(f"视频时长: {len(image_files)/fps:.2f} 秒")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将PNG图像序列转换为MP4视频')
    
    parser.add_argument('--input_dir', '-i', type=str, 
                       default='/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/train/streamGS_Lod_step1/output_meeting_demo/output_sparse50/discussion3_r0.0/test/renders/cam',
                       help='输入图像文件夹路径')
    
    parser.add_argument('--output', '-o', type=str,
                       default='discussion3_test_video.mp4',
                       help='输出视频文件路径 (默认: output_video.mp4)')
    
    parser.add_argument('--fps', '-f', type=int, default=30,
                       help='视频帧率 (默认: 30)')
    
    parser.add_argument('--codec', '-c', type=str, default='mp4v',
                       choices=['mp4v', 'XVID', 'MJPG', 'X264', 'H264'],
                       help='视频编码器 (默认: mp4v)')
    
    args = parser.parse_args() 
    
    # 执行转换
    success = images_to_video(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
        codec=args.codec
    )
    
    if not success:
        print("\n视频生成失败!")
        return 1
    
    return 0

if __name__ == "__main__":
    # 如果直接运行脚本，使用默认参数
    try:
        # 使用默认路径（根据截图）
        input_directory = "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/train/streamGS_Lod_step1/output_meeting_demo/output_sparse50/discussion3_r0.0/test/renders/cam"
        output_file = "output_test_video.mp4"
        
        # 可以修改这些参数
        fps_setting = 30  # 帧率
        codec_setting = 'mp4v'  # 编码器
        
        print(f"输入目录: {input_directory}")
        print(f"输出文件: {output_file}")
        print(f"帧率: {fps_setting} FPS")
        print(f"编码器: {codec_setting}\n")
        
        success = images_to_video(
            input_dir=input_directory,
            output_path=output_file,
            fps=fps_setting,
            codec=codec_setting
        )
        
        if not success:
            print("\n请检查输入路径是否正确,以及是否安装了必要的依赖库。")
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")