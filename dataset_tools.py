import os
import sys

# 设置环境变量
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

# 导入必要的库
import warnings
warnings.filterwarnings("ignore")
import json
import platform
import re
import shutil
import signal
import psutil
import torch
import subprocess
from subprocess import Popen

# 导入配置
from config import (
    exp_root,
    infer_device,
    is_half,
    is_share,
    python_exec,
    webui_port_subfix,
    webui_port_uvr5,
)
from tools import my_utils
from tools.my_utils import check_details, check_for_existance, clean_path
from tools.i18n.i18n import I18nAuto, scan_language_list
from tools.asr.config import asr_dict

# 设置语言
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)

# 设置临时目录
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp

# 清理临时文件
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass

# 设置环境变量
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda: None
except:
    ...

import gradio as gr

# 全局进程变量
p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None

# 进程名称
process_name_subfix = i18n("音频标注WebUI")
process_name_uvr5 = i18n("人声分离WebUI")
process_name_asr = i18n("语音识别")
process_name_denoise = i18n("语音降噪")

# 系统信息
system = platform.system()

# 进程管理函数
def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("进程已终止"))

def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name

# 音频标注工具
def change_label(path_list):
    global p_label
    if p_label is None:
        check_for_existance([path_list])
        path_list = clean_path(path_list)
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
            python_exec,
            path_list,
            webui_port_subfix,
            is_share,
        )
        yield (
            process_info(process_name_subfix, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_label = Popen(cmd, shell=True)
    else:
        kill_process(p_label.pid, process_name_subfix)
        p_label = None
        yield (
            process_info(process_name_subfix, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )

# 人声分离工具
def change_uvr5():
    global p_uvr5
    if p_uvr5 is None:
        cmd = '"%s" tools/uvr5/webui.py "%s" %s %s %s' % (python_exec, infer_device, is_half, webui_port_uvr5, is_share)
        yield (
            process_info(process_name_uvr5, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    else:
        kill_process(p_uvr5.pid, process_name_uvr5)
        p_uvr5 = None
        yield (
            process_info(process_name_uvr5, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )

# 语音识别工具
def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if p_asr is None:
        asr_inp_dir = clean_path(asr_inp_dir)
        asr_opt_dir = clean_path(asr_opt_dir)
        check_for_existance([asr_inp_dir])
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f" -s {asr_model_size}"
        cmd += f" -l {asr_lang}"
        cmd += f" -p {asr_precision}"
        output_file_name = os.path.basename(asr_inp_dir)
        output_folder = asr_opt_dir or "output/asr_opt"
        output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")
        yield (
            process_info(process_name_asr, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        yield (
            process_info(process_name_asr, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": output_file_path},
            {"__type__": "update", "value": output_file_path},
            {"__type__": "update", "value": asr_inp_dir},
        )
    else:
        yield (
            process_info(process_name_asr, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )

def close_asr():
    global p_asr
    if p_asr is not None:
        kill_process(p_asr.pid, process_name_asr)
        p_asr = None
    return (
        process_info(process_name_asr, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )

# 语音降噪工具
def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if p_denoise == None:
        denoise_inp_dir = clean_path(denoise_inp_dir)
        denoise_opt_dir = clean_path(denoise_opt_dir)
        check_for_existance([denoise_inp_dir])
        cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
            python_exec,
            denoise_inp_dir,
            denoise_opt_dir,
            "float16" if is_half == True else "float32",
        )

        yield (
            process_info(process_name_denoise, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        yield (
            process_info(process_name_denoise, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": denoise_opt_dir},
            {"__type__": "update", "value": denoise_opt_dir},
        )
    else:
        yield (
            process_info(process_name_denoise, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )

def close_denoise():
    global p_denoise
    if p_denoise is not None:
        kill_process(p_denoise.pid, process_name_denoise)
        p_denoise = None
    return (
        process_info(process_name_denoise, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )

# 获取GPU信息
def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    mem = []
    if_gpu_ok = False
    
    # 判断是否有能用来训练和加速推理的N卡
    ok_gpu_keywords = {
        "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", 
        "70", "80", "90", "M4", "T4", "TITAN", "L4", "4060", "H", "600", "506", 
        "507", "508", "509",
    }
    set_gpu_numbers = set()
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            if any(value in gpu_name.upper() for value in ok_gpu_keywords):
                if_gpu_ok = True  # 至少有一张能用的N卡
                gpu_infos.append("%s\t%s" % (i, gpu_name))
                set_gpu_numbers.add(i)
                mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
    
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "%s\t%s" % ("0", "CPU")
        gpu_infos.append("%s\t%s" % ("0", "CPU"))
        set_gpu_numbers.add(0)
    
    return gpu_info, gpu_infos, set_gpu_numbers

# 创建Gradio界面
def create_ui():
    gpu_info, gpu_infos, set_gpu_numbers = get_gpu_info()
    
    with gr.Blocks(title="GPT-SoVITS 数据集处理工具") as app:
        gr.Markdown("# GPT-SoVITS 数据集处理工具")
        gr.Markdown(i18n("## 前置数据集获取工具"))
        
        with gr.Tabs():
            with gr.TabItem("0a-" + i18n("UVR5人声分离和语音降噪工具")):
                with gr.Accordion(i18n("人声分离工具")):
                    with gr.Row():
                        with gr.Column():
                            but_uvr5 = gr.Button(i18n("打开人声分离WebUI"), variant="primary")
                            info_uvr5 = gr.Markdown(value="")
                            but_uvr5_stop = gr.Button(i18n("关闭人声分离WebUI"))
                
                with gr.Accordion(i18n("语音降噪工具")):
                    with gr.Row():
                        with gr.Column():
                            denoise_inp_dir = gr.Textbox(
                                label=i18n("输入文件夹路径"),
                                value="",
                                placeholder=i18n("请输入文件夹路径"),
                            )
                            denoise_opt_dir = gr.Textbox(
                                label=i18n("输出文件夹路径"),
                                value="output/denoise_opt",
                                placeholder=i18n("请输入文件夹路径"),
                            )
                        with gr.Column():
                            but_denoise = gr.Button(i18n("开启语音降噪"), variant="primary")
                            info_denoise = gr.Markdown(value="")
                            but_denoise_stop = gr.Button(i18n("停止语音降噪"))


            with gr.TabItem("0b-" + i18n("语音切分工具")):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## " + i18n("语音切分工具"))
                        gr.Markdown(i18n("将长音频文件切分为多个短音频片段，便于后续处理"))
                        
                        input_audio_path = gr.Textbox(
                            label=i18n("音频目录或单个音频路径"),
                            placeholder=i18n("可文件夹批量，可文件夹拖放")
                        )
                        output_dir = gr.Textbox(
                            label=i18n("切分后的子音频输出目录"),
                            value="output/slicer_opt"
                        )
                    
                    with gr.Row():
                        threshold = gr.Slider(
                            minimum=-100, maximum=0, value=-34, step=1,
                            label=i18n("threshold:音量小于多少视作静音")
                        )
                    
                    with gr.Row():
                        min_length = gr.Slider(
                            minimum=100, maximum=10000, value=4000, step=100,
                            label=i18n("min_length:最短小片段，如果一段剪辑小于这个就忽略")
                        )
                        min_interval = gr.Slider(
                            minimum=10, maximum=1000, value=300, step=10,
                            label=i18n("min_interval:最短切割间隔")
                        )
                    
                    with gr.Row():
                        hop_size = gr.Slider(
                            minimum=1, maximum=50, value=10, step=1,
                            label=i18n("hop_size:多少毫秒检测一次，越小越精确，但越慢")
                        )
                        max_sil_kept = gr.Slider(
                            minimum=10, maximum=1000, value=500, step=10,
                            label=i18n("max_sil_kept:切割后静音保留多少毫秒")
                        )
                    
                    with gr.Row():
                        max_db = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                            label=i18n("max的一体化最大值多少")
                        )
                        alpha_mix = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                            label=i18n("alpha_mix:混合参数")
                        )
                    
                    with gr.Row():
                        chunk_num = gr.Slider(
                            minimum=0, maximum=10, value=4, step=1,
                            label=i18n("切分使用进程数")
                        )
                    
                    segment_btn = gr.Button(i18n("开始语音切分"), variant="primary")
                    segment_output = gr.Textbox(label=i18n("语音切分进程输出信息"))
            with gr.TabItem("0c-" + i18n("语音识别工具")):
                with gr.Row():
                    with gr.Column():
                        asr_inp_dir = gr.Textbox(
                            label=i18n("输入文件夹路径"),
                            value="",
                            placeholder=i18n("D:/GPT-SoVITS/raw/xxx"),
                        )
                        asr_opt_dir = gr.Textbox(
                            label=i18n("输出文件夹路径"),
                            value="output/asr_opt",
                            placeholder=i18n("D:/GPT-SoVITS/output/asr_opt"),
                        )
                    with gr.Column():
                        asr_model = gr.Dropdown(
                            label=i18n("ASR 模型"),
                            choices=list(asr_dict.keys()),
                            value="Faster Whisper (多语种)",
                        )
                        asr_model_size = gr.Dropdown(
                            label=i18n("ASR 模型尺寸"),
                            choices=["large-v3", "medium", "small", "base", "tiny"],
                            value="large-v3",
                        )
                        asr_lang = gr.Dropdown(
                            label=i18n("ASR 语言设置"),
                            choices=["auto", "zh", "en", "ja", "ko"],
                            value="auto",
                        )
                        asr_precision = gr.Dropdown(
                            label=i18n("数据类型精度"),
                            choices=["float16", "float32", "int8"],
                            value="float16",
                        )
                with gr.Row():
                    with gr.Column():
                        but_asr = gr.Button(i18n("开启语音识别"), variant="primary")
                        info_asr = gr.Markdown(value="")
                        but_asr_stop = gr.Button(i18n("停止语音识别"))
                        asr_output_info = gr.Markdown(value=i18n("语音识别进程输出信息"))
            
            with gr.TabItem("0d-" + i18n("音频标注工具")):
                with gr.Row():
                    with gr.Column():
                        path_list = gr.Textbox(
                            label=i18n("标注文件路径 (文件扩展名*.list)"),
                            value="D:/GPT-SoVITS/raw/xxx.list",
                        )
                    with gr.Column():
                        but_label = gr.Button(i18n("打开音频标注WebUI"), variant="primary")
                        info_label = gr.Markdown(value="")
                        but_label_stop = gr.Button(i18n("关闭音频标注WebUI"))
            
        # 事件绑定
        but_uvr5.click(change_uvr5, [], [info_uvr5, but_uvr5, but_uvr5_stop])
        but_uvr5_stop.click(change_uvr5, [], [info_uvr5, but_uvr5, but_uvr5_stop])
        
        but_denoise.click(
            open_denoise,
            [denoise_inp_dir, denoise_opt_dir],
            [info_denoise, but_denoise, but_denoise_stop, denoise_inp_dir, denoise_opt_dir],
        )
        but_denoise_stop.click(close_denoise, [], [info_denoise, but_denoise, but_denoise_stop])
        
        but_asr.click(
            open_asr,
            [asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision],
            [info_asr, but_asr, but_asr_stop, asr_inp_dir, asr_opt_dir, asr_output_info],
        )
        but_asr_stop.click(close_asr, [], [info_asr, but_asr, but_asr_stop])
        
        but_label.click(change_label, [path_list], [info_label, but_label, but_label_stop])
        but_label_stop.click(change_label, [path_list], [info_label, but_label, but_label_stop])
        
        # 语音切分功能
        def audio_segmentation(input_audio_path, output_dir, threshold, min_length, min_interval, hop_size, max_sil_kept, max_db, alpha_mix, chunk_num):
            try:
                from tools.slicer2 import Slicer
                import librosa
                import soundfile as sf
                import os
                import glob
                import concurrent.futures
                
                os.makedirs(output_dir, exist_ok=True)
                
                # 检查输入是文件还是目录
                input_files = []
                if os.path.isdir(input_audio_path):
                    # 如果是目录，获取所有音频文件
                    input_files.extend(glob.glob(os.path.join(input_audio_path, "*.wav")))
                    input_files.extend(glob.glob(os.path.join(input_audio_path, "*.mp3")))
                    input_files.extend(glob.glob(os.path.join(input_audio_path, "*.flac")))
                else:
                    # 如果是单个文件
                    input_files.append(input_audio_path)
                
                if not input_files:
                    return f"未找到音频文件，请检查输入路径"
                
                # 创建切分器
                slicer = Slicer(
                    sr=44100,
                    threshold=threshold,
                    min_silence_len=min_interval,
                    hop_len=hop_size,
                    min_length=min_length,
                    max_sil_kept=max_sil_kept,
                    max_db=max_db,
                    alpha_mix=alpha_mix
                )
                
                # 处理函数
                def process_file(file_path):
                    try:
                        # 加载音频
                        audio, sr = librosa.load(file_path, sr=None, mono=True)
                        
                        # 切分音频
                        chunks = slicer.slice(audio)
                        
                        # 保存切分后的音频片段
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        saved_files = []
                        
                        for i, chunk in enumerate(chunks):
                            output_file = os.path.join(output_dir, f"{base_name}_{i+1:04d}.wav")
                            sf.write(output_file, chunk, sr)
                            saved_files.append(output_file)
                        
                        return f"文件 {file_path} 成功切分为 {len(chunks)} 个片段"
                    except Exception as e:
                        return f"处理文件 {file_path} 失败: {str(e)}"
                
                # 使用多进程处理
                results = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, chunk_num)) as executor:
                    futures = [executor.submit(process_file, file) for file in input_files]
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                
                return "\n".join(results)
            except Exception as e:
                return f"音频切分失败: {str(e)}"
        
        # 更新事件绑定
        segment_btn.click(
            fn=audio_segmentation,
            inputs=[
                input_audio_path, output_dir, 
                threshold, min_length, min_interval, 
                hop_size, max_sil_kept, max_db, 
                alpha_mix, chunk_num
            ],
            outputs=[segment_output]
        )
        
    return app

# 主函数
if __name__ == "__main__":
    # 设置端口
    webui_port = 9870
    
    # 创建并启动Gradio界面
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=webui_port,
        share=is_share,
        inbrowser=True
    )