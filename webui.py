import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")
import platform
import signal
import psutil
import torch
torch.manual_seed(233333)
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
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
import traceback
import subprocess
from subprocess import Popen
from config import (
    infer_device,
    is_half,
    is_share,
    python_exec,
    webui_port_main,
    webui_port_subfix,
    webui_port_uvr5,
)
from tools import my_utils
from tools.i18n.i18n import I18nAuto, scan_language_list
i18n = I18nAuto(language= "zh_CN")
from multiprocessing import cpu_count
from tools.my_utils import check_for_existance
import gradio as gr
n_cpu = cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
    "L4",
    "4060",
    "H",
    "600",
    "506",
    "507",
    "508",
    "509",
}
set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))




p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None
p_tts_inference = None


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


system = platform.system()
def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
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


process_name_subfix = i18n("音频标注WebUI")
def change_label(path_list):
    global p_label
    if p_label is None:
        check_for_existance([path_list])
        path_list = my_utils.clean_path(path_list)
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


process_name_uvr5 = i18n("人声分离WebUI")
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





from tools.asr.config import asr_dict
process_name_asr = i18n("语音识别")
def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if p_asr is None:
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_opt_dir = my_utils.clean_path(asr_opt_dir)
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



process_name_denoise = i18n("语音降噪")


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if p_denoise == None:
        denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
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







ps_slice = []
process_name_slice = i18n("语音切分")
default_batch_size = 3

def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if os.path.exists(inp) == False:
        yield (
            i18n("输入路径不存在"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield (
            i18n("输入路径存在但不可用"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s' % (
                python_exec,
                inp,
                opt_root,
                threshold,
                min_length,
                min_interval,
                hop_size,
                max_sil_kept,
                _max,
                alpha,
                i_part,
                n_parts,
            )
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield (
            process_info(process_name_slice, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield (
            process_info(process_name_slice, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": opt_root},
            {"__type__": "update", "value": opt_root},
            {"__type__": "update", "value": opt_root},
        )
    else:
        yield (
            process_info(process_name_slice, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close_slice():
    global ps_slice
    if ps_slice != []:
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid, process_name_slice)
            except:
                traceback.print_exc()
        ps_slice = []
    return (
        process_info(process_name_slice, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )







with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    with gr.Tabs():
        with gr.TabItem("0-" + i18n("前置数据集获取工具")):  # 提前随机切片防止uvr5爆内存->uvr5->slicer->asr->打标
            gr.Markdown(value="0a-" + i18n("UVR5人声伴奏分离&去混响去延迟工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        uvr5_info = gr.Textbox(label=process_info(process_name_uvr5, "info"))
                open_uvr5 = gr.Button(value=process_info(process_name_uvr5, "open"), variant="primary", visible=True)
                close_uvr5 = gr.Button(value=process_info(process_name_uvr5, "close"), variant="primary", visible=False)

            gr.Markdown(value="0b-" + i18n("语音切分工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        slice_inp_path = gr.Textbox(label=i18n("音频自动切分输入路径，可文件可文件夹"), value="")
                        slice_opt_root = gr.Textbox(label=i18n("切分后的子音频的输出根目录"), value="output/slicer_opt")
                    with gr.Row():
                        threshold = gr.Textbox(label=i18n("threshold:音量小于这个值视作静音的备选切割点"), value="-34")
                        min_length = gr.Textbox(
                            label=i18n("min_length:每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值"),
                            value="4000",
                        )
                        min_interval = gr.Textbox(label=i18n("min_interval:最短切割间隔"), value="300")
                        hop_size = gr.Textbox(
                            label=i18n("hop_size:怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）"),
                            value="10",visible=False, interactive=False
                        )
                        max_sil_kept = gr.Textbox(label=i18n("max_sil_kept:切完后静音最多留多长"), value="500",visible=False, interactive=False)
                    with gr.Row(visible=False):
                        _max = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            label=i18n("max:归一化后最大值多少"),
                            value=0.9,
                            interactive=True,
                        )
                        alpha = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            label=i18n("alpha_mix:混多少比例归一化后音频进来"),
                            value=0.25,
                            interactive=True,
                        )
                    with gr.Row():
                        n_process = gr.Slider(
                            minimum=1, maximum=n_cpu, step=1, label=i18n("切割使用的进程数"), value=4, interactive=True
                        )
                        slicer_info = gr.Textbox(label=process_info(process_name_slice, "info"))
                open_slicer_button = gr.Button(
                    value=process_info(process_name_slice, "open"), variant="primary", visible=True
                )
                close_slicer_button = gr.Button(
                    value=process_info(process_name_slice, "close"), variant="primary", visible=False
                )

            gr.Markdown(value="0bb-" + i18n("语音降噪工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        denoise_input_dir = gr.Textbox(label=i18n("输入文件夹路径"), value="")
                        denoise_output_dir = gr.Textbox(label=i18n("输出文件夹路径"), value="output/denoise_opt")
                    with gr.Row():
                        denoise_info = gr.Textbox(label=process_info(process_name_denoise, "info"))
                open_denoise_button = gr.Button(
                    value=process_info(process_name_denoise, "open"), variant="primary", visible=True
                )
                close_denoise_button = gr.Button(
                    value=process_info(process_name_denoise, "close"), variant="primary", visible=False
                )

            gr.Markdown(value="0c-" + i18n("语音识别工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        asr_inp_dir = gr.Textbox(
                            label=i18n("输入文件夹路径"), value="D:\\GPT-SoVITS\\raw\\xxx", interactive=True
                        )
                        asr_opt_dir = gr.Textbox(label=i18n("输出文件夹路径"), value="output/asr_opt", interactive=True)
                    with gr.Row():
                        asr_model = gr.Dropdown(
                            label=i18n("ASR 模型"),
                            choices=list(asr_dict.keys()),
                            interactive=True,
                            value="达摩 ASR (中文)",
                        )
                        asr_size = gr.Dropdown(
                            label=i18n("ASR 模型尺寸"), choices=["large"], interactive=True, value="large"
                        )
                        asr_lang = gr.Dropdown(
                            label=i18n("ASR 语言设置"), choices=["zh", "yue"], interactive=True, value="zh"
                        )
                        asr_precision = gr.Dropdown(
                            label=i18n("数据类型精度"), choices=["float32"], interactive=True, value="float32"
                        )
                    with gr.Row():
                        asr_info = gr.Textbox(label=process_info(process_name_asr, "info"))
                open_asr_button = gr.Button(
                    value=process_info(process_name_asr, "open"), variant="primary", visible=True
                )
                close_asr_button = gr.Button(
                    value=process_info(process_name_asr, "close"), variant="primary", visible=False
                )

                def change_lang_choices(key):  # 根据选择的模型修改可选的语言
                    return {"__type__": "update", "choices": asr_dict[key]["lang"], "value": asr_dict[key]["lang"][0]}

                def change_size_choices(key):  # 根据选择的模型修改可选的模型尺寸
                    return {"__type__": "update", "choices": asr_dict[key]["size"], "value": asr_dict[key]["size"][-1]}

                def change_precision_choices(key):  # 根据选择的模型修改可选的语言
                    if key == "Faster Whisper (多语种)":
                        if default_batch_size <= 4:
                            precision = "int8"
                        elif is_half:
                            precision = "float16"
                        else:
                            precision = "float32"
                    else:
                        precision = "float32"
                    return {"__type__": "update", "choices": asr_dict[key]["precision"], "value": precision}

                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                asr_model.change(change_precision_choices, [asr_model], [asr_precision])

            gr.Markdown(value="0d-" + i18n("语音文本校对标注工具"))
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        path_list = gr.Textbox(
                            label=i18n("标注文件路径 (含文件后缀 *.list)"),
                            value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                            interactive=True,
                        )
                        label_info = gr.Textbox(label=process_info(process_name_subfix, "info"))
                open_label = gr.Button(value=process_info(process_name_subfix, "open"), variant="primary", visible=True)
                close_label = gr.Button(
                    value=process_info(process_name_subfix, "close"), variant="primary", visible=False
                )

            open_label.click(change_label, [path_list], [label_info, open_label, close_label])
            close_label.click(change_label, [path_list], [label_info, open_label, close_label])
            open_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])
            close_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])

            open_asr_button.click(
                open_asr,
                [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision],
                [asr_info, open_asr_button, close_asr_button, path_list],
            )
            close_asr_button.click(close_asr, [], [asr_info, open_asr_button, close_asr_button])
            open_slicer_button.click(
                open_slice,
                [
                    slice_inp_path,
                    slice_opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    n_process,
                ],
                [slicer_info, open_slicer_button, close_slicer_button, asr_inp_dir, denoise_input_dir],
            )
            close_slicer_button.click(close_slice, [], [slicer_info, open_slicer_button, close_slicer_button])
            open_denoise_button.click(
                open_denoise,
                [denoise_input_dir, denoise_output_dir],
                [denoise_info, open_denoise_button, close_denoise_button, asr_inp_dir],
            )
            close_denoise_button.click(close_denoise, [], [denoise_info, open_denoise_button, close_denoise_button])

    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        # quiet=True,
    )
