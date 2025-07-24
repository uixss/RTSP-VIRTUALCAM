import os
import cv2
import pyvirtualcam
import numpy as np
import mss
import customtkinter as ctk
from threading import Thread
import time
from PIL import Image, ImageTk
import subprocess
import socket
from customtkinter import CTkImage  
default_virtual_width, default_virtual_height = 1920, 1080
preview_width, preview_height = 640, 360
fps = 60
source_mode = None   
IMAGE_PATH = None
VIDEO_PATH = None
top_offset = left_offset = zoom_level = rotate_angle = None
flip_horizontal = flip_vertical = show_grid = None
brightness = contrast = saturation = sharpen = None
stretch_x = stretch_y = input_width = input_height = None
output_width = output_height = monitor_index = None
monitor = None
root = None
cam_restart_requested = False
virtual_cam_thread = None
preview_window = None
preview_label = None
USE_PREVIEW_WINDOW = True 
ffmpeg_process = None

FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"  
rtsp_profiles = {
    "HD 1080px": {
    "scale": "1920:1080",
    "fps": 30,
    "preset": "faster",        
    "crf": 22,
    "profile": "baseline"
},

    "HIGH 720px": {
        "scale": "1280:720",
        "fps": 30,
        "preset": "veryfast",
        "crf": 23,
        "profile": "baseline"
    },
    "MID 480px": {
        "scale": "854:480",
        "fps": 30,
        "preset": "veryfast",
        "crf": 25,
        "profile": "baseline"
    },
    "LOW 360px": {
        "scale": "640:360",
        "fps": 15,
        "preset": "ultrafast",
        "crf": 24,
        "profile": "baseline"
    }
}

def girar_90_grados():
    current_angle = rotate_angle.get()
    rotate_angle.set((current_angle + 90) % 360)

def stop_ffmpeg_stream():
    global ffmpeg_process
    if ffmpeg_process and ffmpeg_process.poll() is None:
        ffmpeg_process.terminate()
        ffmpeg_process = None

def iniciar_media_mtx_en_terminal():
    try:
 
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_exe = os.path.join(directorio_actual, "mediamtx.exe")
 
        subprocess.Popen(
            ['cmd.exe', '/k', ruta_exe],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    except Exception as e:
        print("❌ Error al iniciar MediaMTX:", e)

def start_ffmpeg_stream(camera_name, mic_name, stream_url, resolution, label_status, profile_name):
    def run():
        global ffmpeg_process
        try:
            stop_ffmpeg_stream()

            width, height = resolution.split("x")
            profile = rtsp_profiles.get(profile_name, rtsp_profiles["LOW 360px"])

            scale = profile["scale"]
            fps = str(profile["fps"])
            preset = profile["preset"]
            crf = str(profile["crf"])
            x264_profile = profile["profile"]

            cmd = [
                FFMPEG_PATH,
                "-f", "dshow",
                "-rtbufsize", "200M",
                "-i", f"video={camera_name}:audio={mic_name}",
                "-vf", f"scale={scale}",
                "-pix_fmt", "yuv420p",
                "-r", fps,
                "-vcodec", "libx264",
                "-preset", preset,
                "-tune", "zerolatency",
                "-crf", crf,
                "-x264-params", f"profile={x264_profile}",
                "-f", "rtsp",
                stream_url
            ]

            ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            label_status.configure(text=f"📡 {profile_name} transmitiendo a {scale}: {stream_url}", text_color="green")
        except Exception as e:
            label_status.configure(text=f"❌ Error: {str(e)}", text_color="red")

    Thread(target=run, daemon=True).start()
def dispositivo_audio_existe(nombre):
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        return nombre in result.stderr
    except:
        return False

def apply_crop(frame, target_width, target_height):
    h, w = frame.shape[:2]
    target_ratio = target_width / target_height
    frame_ratio = w / h

    if frame_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x_offset = (w - new_w) // 2
        frame = frame[:, x_offset:x_offset + new_w]
    else:
        new_h = int(w / target_ratio)
        y_offset = (h - new_h) // 2
        frame = frame[y_offset:y_offset + new_h, :]

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def rotate_image(frame, angle):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
    if brightness != 0:
        shadow = brightness if brightness > 0 else 0
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha = (highlight - shadow) / 255
        gamma = shadow
        frame = cv2.addWeighted(frame, alpha, frame, 0, gamma)
    if contrast != 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
    return frame

def adjust_saturation(frame, saturation=1.0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_sharpen(frame, amount=1.0):
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1 + amount, blur, -amount, 0)

def show_calibration_grid(frame):
    h, w = frame.shape[:2]
    for x in range(0, w, 50):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, 50):
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)
    cv2.line(frame, (w//2, 0), (w//2, h), (0, 0, 255), 2)
    cv2.line(frame, (0, h//2), (w, h//2), (0, 0, 255), 2)
    return frame

def apply_effects(frame):
    frame = adjust_brightness_contrast(
        frame,
        safe_get_int(brightness),
        safe_get_int(contrast, 50) / 50
    )
    frame = adjust_saturation(frame, safe_get_float(saturation, 1.0))
    frame = apply_sharpen(frame, safe_get_float(sharpen, 0.0))
    return frame

def reset_defaults():
    top_offset.set(0)
    left_offset.set(0)
    zoom_level.set(100)
    rotate_angle.set(0)
    flip_horizontal.set(False)
    flip_vertical.set(False)
    show_grid.set(False)
    brightness.set(0)
    contrast.set(50)
    saturation.set(1.0)
    sharpen.set(0.0)
    stretch_x.set(100)
    stretch_y.set(100)

    # Obtener monitor actual
    try:
        current_monitor = mss.mss().monitors[monitor_index.get()]
        input_width.set(current_monitor["width"])
        input_height.set(current_monitor["height"])
    except Exception as e:
        print("Error obteniendo el monitor:", e)
        input_width.set(default_virtual_width)
        input_height.set(default_virtual_height)

    output_width.set(default_virtual_width)
    output_height.set(default_virtual_height)

def safe_get_int(var, default=0):
    try:
        return int(var.get())
    except (ValueError, TypeError):
        return default

def safe_get_float(var, default=0.0):
    try:
        return float(var.get())
    except (ValueError, TypeError):
        return default


def show_preview(frame):
    if USE_PREVIEW_WINDOW:
        preview = cv2.resize(frame, (preview_width, preview_height))
        img = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        ctk_img = CTkImage(light_image=img, size=(preview_width, preview_height))

        if preview_label and isinstance(preview_label, ctk.CTkLabel):
            preview_label.configure(image=ctk_img)
            preview_label.image = ctk_img

        if preview_window and hasattr(preview_window, "update_modal_image"):
            preview_window.update_modal_image(ctk_img)

def start_preview_loop():
    global cam_restart_requested, monitor, IMAGE_PATH, VIDEO_PATH

    while any(v is None for v in [
        top_offset, left_offset, zoom_level, rotate_angle,
        flip_horizontal, flip_vertical, show_grid, brightness,
        contrast, saturation, sharpen, stretch_x, stretch_y,
        input_width, input_height, output_width, output_height, monitor_index
    ]):
        time.sleep(0.1)

    vw = output_width.get()
    vh = output_height.get()

    with pyvirtualcam.Camera(width=vw, height=vh, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        if source_mode.get() == "imagen":
            if IMAGE_PATH is None or not os.path.exists(IMAGE_PATH):
                print("Ruta de imagen inválida.")
                return
            img = cv2.imread(IMAGE_PATH)
            img = cv2.resize(img, (vw, vh))

            while not cam_restart_requested:
                frame = apply_effects(img.copy())
                if show_grid.get():
                    frame = show_calibration_grid(frame)
                cam.send(frame)
                show_preview(frame)
                cam.sleep_until_next_frame()

        elif source_mode.get() == "video":
            if VIDEO_PATH is None or not os.path.exists(VIDEO_PATH):
                print("Ruta de video inválida.")
                return
            cap = cv2.VideoCapture(VIDEO_PATH)

            while cap.isOpened() and not cam_restart_requested:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (vw, vh))
                frame = apply_effects(frame)
                if show_grid.get():
                    frame = show_calibration_grid(frame)
                cam.send(frame)
                show_preview(frame)
                cam.sleep_until_next_frame()

            cap.release()

        else:  # modo pantalla
            with mss.mss() as sct:
                while not cam_restart_requested:
                    selected_monitor = sct.monitors[monitor_index.get()]
                    zoom = max(safe_get_int(zoom_level, 100) / 100, 0.1)
                    region_w = int(safe_get_int(input_width, default_virtual_width) / zoom)
                    region_h = int(safe_get_int(input_height, default_virtual_height) / zoom)
                    x = min(safe_get_int(left_offset, 0), selected_monitor["width"] - region_w)
                    y = min(safe_get_int(top_offset, 0), selected_monitor["height"] - region_h)

                    region = {
                        "left": selected_monitor["left"] + x,
                        "top": selected_monitor["top"] + y,
                        "width": region_w,
                        "height": region_h
                    }

                    frame = np.array(sct.grab(region))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    fx = max(safe_get_int(stretch_x, 100) / 100.0, 0.01)
                    fy = max(safe_get_int(stretch_y, 100) / 100.0, 0.01)

                    if fx != 1.0 or fy != 1.0:
                        try:
                            frame = cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
                        except cv2.error as e:
                            print("Error al redimensionar:", e)
                            continue   


                    frame = apply_crop(frame, vw, vh)

                    if flip_horizontal.get(): frame = cv2.flip(frame, 1)
                    if flip_vertical.get(): frame = cv2.flip(frame, 0)
                    if rotate_angle.get(): frame = rotate_image(frame, rotate_angle.get())
                    if show_grid.get(): frame = show_calibration_grid(frame)

                    frame = apply_effects(frame)
                    cam.send(frame)
                    show_preview(frame)
                    cam.sleep_until_next_frame()

def listar_camaras_disponibles(max_index=10):
    disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.read()[0]:
            disponibles.append(f"Cam {i}")
        cap.release()
    return disponibles

def gui_thread():
    global top_offset, left_offset, zoom_level, rotate_angle, flip_horizontal, flip_vertical
    global show_grid, brightness, contrast, saturation, sharpen, stretch_x, stretch_y
    global input_width, input_height, output_width, output_height, monitor_index, root
    global source_mode, IMAGE_PATH, VIDEO_PATH
    global rtsp_profile
    audio_input_options = {
    "🎤 Micrófono": "Microphone Array (AMD Audio Device)",
    "🎧 Escritorio": "virtual-audio-capturer"  # Requiere VB-Cable o similar
}

    from tkinter import filedialog
    import os

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.geometry("520x560")
    root.title("Server RTSP ")
    root.configure(bg="#262626")
    root.resizable(False, False)
    root.overrideredirect(True)
    root.wm_attributes('-alpha', 0.95)
    top_offset = ctk.IntVar(value=0)
    left_offset = ctk.IntVar(value=0)
    zoom_level = ctk.IntVar(value=100)
    rotate_angle = ctk.IntVar(value=0)
    flip_horizontal = ctk.BooleanVar(value=False)
    flip_vertical = ctk.BooleanVar(value=False)
    show_grid = ctk.BooleanVar(value=False)
    brightness = ctk.IntVar(value=0)
    contrast = ctk.IntVar(value=50)
    saturation = ctk.DoubleVar(value=1.0)
    sharpen = ctk.DoubleVar(value=0.0)
    stretch_x = ctk.IntVar(value=100)
    stretch_y = ctk.IntVar(value=100)
    input_width = ctk.IntVar(value=default_virtual_width)
    input_height = ctk.IntVar(value=default_virtual_height)
    output_width = ctk.IntVar(value=default_virtual_width)
    output_height = ctk.IntVar(value=default_virtual_height)
    monitor_index = ctk.IntVar(value=1)
    source_mode = ctk.StringVar(value="pantalla")
    IMAGE_PATH = None
    VIDEO_PATH = None

    def start_move(e): root.x, root.y = e.x, e.y
    def stop_move(e): root.x, root.y = None, None
    def on_motion(e): root.geometry(f"+{e.x_root - root.x}+{e.y_root - root.y}")

    title_bar = ctk.CTkFrame(root, height=30, fg_color="#1f1f1f")
    title_bar.pack(fill="x")
    ctk.CTkLabel(title_bar, text="", text_color="white").pack(side="left", padx=10)
    title_bar.bind("<Button-1>", start_move)
    title_bar.bind("<ButtonRelease-1>", stop_move)
    title_bar.bind("<B1-Motion>", on_motion)

    tabview = ctk.CTkTabview(root, width=800, height=450)
    tabview.pack(padx=10, pady=(10, 0))
    
 

    def open_preview_modal():
        global preview_window
        if preview_window and preview_window.winfo_exists():
            preview_window.focus()
            return

        preview_window = ctk.CTkToplevel(root)
        preview_window.title("Live Stream")
        preview_window.geometry(f"{preview_width}x{preview_height}")
        preview_window.resizable(False, False)

        modal_label = ctk.CTkLabel(preview_window, text="")
        modal_label.pack(expand=True)
        
        def update_modal_image(img):
            if modal_label and isinstance(modal_label, ctk.CTkLabel):
                modal_label.configure(image=img)
                modal_label.image = img

        preview_window.update_modal_image = update_modal_image 

        def on_close():
            global preview_window
            preview_window.destroy()
            preview_window = None

        preview_window.protocol("WM_DELETE_WINDOW", on_close)
 
    live_tab = tabview.add("Live Stream")
    live_frame = ctk.CTkFrame(live_tab, fg_color="transparent")
    live_frame.pack(expand=True, fill="both")

    live_top_controls = ctk.CTkFrame(live_frame, fg_color="transparent")
    live_top_controls.pack(fill="x", pady=(10, 0))

    open_modal_btn = ctk.CTkButton(
        live_top_controls,
        text="Vista Flotante",
        command=open_preview_modal,
        width=150
    )
    open_modal_btn.pack(side="left", padx=20)
    live_label = ctk.CTkLabel(live_frame, text="", width=preview_width, height=preview_height)
    live_label.pack(expand=True, pady=(10, 20))  
    global preview_label
    preview_label = live_label
 
    def iniciar_rtsp():
        url = get_full_rtsp_url()

        width = output_width.get()
        height = output_height.get()
        profile = rtsp_profile.get()
        if not url:
            rtsp_status.configure(text="⚠️ Ingresa una URL válida", text_color="orange")
            return
        rtsp_status.configure(text="🔄 Iniciando transmisión...", text_color="yellow")
        selected_audio = audio_input_options[audio_source.get()]
        if not dispositivo_audio_existe(selected_audio):
            rtsp_status.configure(text=f"⚠️ No se encontró el dispositivo: {selected_audio}", text_color="red")
            return

        start_ffmpeg_stream("OBS Virtual Camera", selected_audio, url, f"{width}x{height}", rtsp_status, profile)

    def reiniciar_rtsp():
        url = get_full_rtsp_url()

        width = output_width.get()
        height = output_height.get()
        profile = rtsp_profile.get()
        if not url:
            rtsp_status.configure(text="⚠️ Ingresa una URL válida", text_color="orange")
            return
        rtsp_status.configure(text="🔁 Reiniciando transmisión...", text_color="yellow")
        selected_audio = audio_input_options[audio_source.get()]
        start_ffmpeg_stream("OBS Virtual Camera", selected_audio, url, f"{width}x{height}", rtsp_status, profile)
    screen_tab = tabview.add("Server RTSP")
    pantalla_frame = ctk.CTkFrame(screen_tab, fg_color="transparent")
    pantalla_frame.pack(expand=True, pady=20)

    # Obtener IP local (IPv4)
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "0.0.0.0"

    local_ip = get_local_ip()
 
 
    ip_label = ctk.CTkLabel(pantalla_frame, text=f"RTSP://{local_ip}:8554", text_color="green", font=("Arial", 14, "bold"))
    ip_label.pack(pady=(0, 5))
 
    rtsp_prefix = "rtsp://localhost:8554/"
    rtsp_path = ctk.StringVar(value="mistream")   
    url_frame = ctk.CTkFrame(pantalla_frame, fg_color="transparent")
    url_frame.pack(pady=(5, 15))
    ctk.CTkLabel(url_frame, text=rtsp_prefix, font=("Arial", 12)).pack(side="left")
    ctk.CTkEntry(url_frame, textvariable=rtsp_path, width=200).pack(side="left")
    def get_full_rtsp_url():
        return rtsp_prefix + rtsp_path.get().strip()



    top_controls = ctk.CTkFrame(pantalla_frame, fg_color="transparent")
    top_controls.pack(pady=(0, 15))

    rtsp_profile = ctk.StringVar(value="LOW 360px")
    audio_source = ctk.StringVar(value="🎤 Micrófono")

    ctk.CTkOptionMenu(top_controls, variable=rtsp_profile, values=list(rtsp_profiles.keys()), width=180).pack(side="left", padx=10)
    ctk.CTkOptionMenu(top_controls, variable=audio_source, values=list(audio_input_options.keys()), width=180).pack(side="left", padx=10)

    ctk.CTkButton(top_controls, text="▶", command=iniciar_rtsp, width=160).pack(side="left", padx=10)

    rtsp_button_column = ctk.CTkFrame(pantalla_frame, fg_color="transparent")
    rtsp_button_column.pack()

    ctk.CTkButton(rtsp_button_column, text="Reiniciar", command=reiniciar_rtsp, width=200).pack(pady=(0, 10))
    ctk.CTkButton(rtsp_button_column, text="⏹", command=stop_ffmpeg_stream, width=200).pack()

    rtsp_status = ctk.CTkLabel(pantalla_frame, text="", text_color="gray")
    rtsp_status.pack(pady=10)

    dim_frame = ctk.CTkFrame(pantalla_frame, fg_color="transparent")
    dim_frame.pack(pady=(10, 0))

    ctk.CTkLabel(dim_frame, text="Dimensiones de entrada").pack()
    entry_in = ctk.CTkFrame(dim_frame, fg_color="transparent")
    entry_in.pack()
    ctk.CTkEntry(entry_in, textvariable=input_width, width=100).pack(side="left", padx=5)
    ctk.CTkLabel(entry_in, text="x").pack(side="left")
    ctk.CTkEntry(entry_in, textvariable=input_height, width=100).pack(side="left", padx=5)

    ctk.CTkLabel(dim_frame, text="Dimensiones de salida").pack(pady=(10, 2))
    entry_out = ctk.CTkFrame(dim_frame, fg_color="transparent")
    entry_out.pack()
    ctk.CTkEntry(entry_out, textvariable=output_width, width=100).pack(side="left", padx=5)
    ctk.CTkLabel(entry_out, text="x").pack(side="left")
    ctk.CTkEntry(entry_out, textvariable=output_height, width=100).pack(side="left", padx=5)

    movement_tab = tabview.add("Transmisión de Pantallas")
    move_frame = ctk.CTkFrame(movement_tab, fg_color="transparent")
    move_frame.pack(expand=True)

 
    center_container = ctk.CTkFrame(move_frame, fg_color="transparent")
    center_container.pack(expand=True)


    selector_frame = ctk.CTkFrame(center_container, fg_color="transparent")
    selector_frame.pack(pady=(10, 5))

    ctk.CTkLabel(selector_frame, text=" ").pack(side="left", padx=(0, 10))

    monitors = mss.mss().monitors[1:]
    monitor_names = [f"Pantalla {i+1}" for i in range(len(monitors))]
    ctk.CTkOptionMenu(selector_frame, values=monitor_names,
        command=lambda val: monitor_index.set(monitor_names.index(val) + 1)
    ).pack(side="left")

   
    grid_check = ctk.CTkCheckBox(center_container, text="Cuadricula", variable=show_grid)
    grid_check.pack(pady=(5, 15))

    def add_horizontal_slider_with_entry(parent, label, var, from_, to_):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5, padx=30)

        ctk.CTkLabel(row, text=label, width=110, anchor="e").pack(side="left", padx=5)

        slider = ctk.CTkSlider(row, from_=from_, to=to_, variable=var, width=200)
        slider.pack(side="left", padx=5)

        entry = ctk.CTkEntry(row, width=60)
        entry.insert(0, str(var.get()))
        entry.pack(side="left", padx=5)

        def update_entry_from_var(*_):
            entry.delete(0, "end")
            entry.insert(0, f"{var.get():.2f}" if isinstance(var.get(), float) else str(var.get()))

        def update_var_from_entry(event=None):
            val = entry.get()
            if val.strip() == "":
                return  # No actualizar si está vacío
            try:
                value = float(val) if isinstance(var.get(), float) else int(val)
                var.set(value)
            except ValueError:
                update_entry_from_var()

        var.trace_add("write", lambda *args: update_entry_from_var())
        entry.bind("<Return>", update_var_from_entry)

    add_horizontal_slider_with_entry(center_container, "Posición Y", top_offset, 0, 3000)
    add_horizontal_slider_with_entry(center_container, "Posición X", left_offset, 0, 3000)
    add_horizontal_slider_with_entry(center_container, "Zoom (%)", zoom_level, 50, 300)
    add_horizontal_slider_with_entry(center_container, "Rotación", rotate_angle, -180, 180)

    def girar_90_grados():
        current_angle = rotate_angle.get()
        rotate_angle.set((current_angle + 90) % 360)

    rotate_button = ctk.CTkButton(center_container, text="🔄 90°", command=girar_90_grados, width=160)
    rotate_button.pack(pady=(10, 5))
    mirror_frame = ctk.CTkFrame(center_container, fg_color="transparent")
    mirror_frame.pack(pady=(10, 5))

    flip_h_check = ctk.CTkCheckBox(
        mirror_frame,
        text="Espejo Horizontal",
        variable=flip_horizontal
    )
    flip_h_check.pack(side="left", padx=10)

    flip_v_check = ctk.CTkCheckBox(
        mirror_frame,
        text="Espejo Vertical",
        variable=flip_vertical
    )
    flip_v_check.pack(side="left", padx=10)

 
    effects_tab = tabview.add("Efectos")
    effect_frame = ctk.CTkFrame(effects_tab, fg_color="transparent")
    effect_frame.pack(pady=10)

    def add_vertical_slider(parent, label, var, from_, to_):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(pady=2)

        ctk.CTkLabel(frame, text=label, font=("Arial", 10, "bold")).pack()

        value_frame = ctk.CTkFrame(frame, fg_color="transparent")
        value_frame.pack()

        slider = ctk.CTkSlider(value_frame, from_=from_, to=to_, variable=var, width=200)
        slider.pack(side="left", padx=(0, 10))

        value_entry = ctk.CTkEntry(value_frame, width=60)
        value_entry.insert(0, str(var.get()))
        value_entry.pack(side="left")

        def update_entry_from_var(*_):
            value_entry.delete(0, "end")
            value_entry.insert(0, f"{var.get():.2f}" if isinstance(var.get(), float) else str(var.get()))

        def update_var_from_entry(event=None):
            try:
                value = float(value_entry.get()) if isinstance(var.get(), float) else int(value_entry.get())
                var.set(value)
            except ValueError:
                update_entry_from_var()  

        var.trace_add("write", lambda *args: update_entry_from_var())
        value_entry.bind("<Return>", update_var_from_entry)


    add_vertical_slider(effect_frame, "Brillo", brightness, -100, 100)
    add_vertical_slider(effect_frame, "Contraste", contrast, 0, 100)
    add_vertical_slider(effect_frame, "Saturación", saturation, 0.0, 2.0)
    add_vertical_slider(effect_frame, "Nitidez", sharpen, 0.0, 2.0)

    stretch_frame = ctk.CTkFrame(effect_frame, fg_color="transparent")
    stretch_frame.pack(pady=5)

    def add_stretch_slider(label_text, var):
        row = ctk.CTkFrame(stretch_frame, fg_color="transparent")
        row.pack(pady=5)

        ctk.CTkLabel(row, text=label_text, width=140, anchor="e").pack(side="left", padx=5)

        slider = ctk.CTkSlider(row, from_=50, to=150, variable=var, width=180)
        slider.pack(side="left", padx=5)

        entry = ctk.CTkEntry(row, width=60)
        entry.insert(0, str(var.get()))
        entry.pack(side="left", padx=5)

        def update_entry(*_):
            entry.delete(0, "end")
            entry.insert(0, f"{var.get():.2f}" if isinstance(var.get(), float) else str(var.get()))

        def update_var(event=None):
            try:
                value = float(entry.get()) if isinstance(var.get(), float) else int(entry.get())
                var.set(value)
            except ValueError:
                update_entry()

        var.trace_add("write", lambda *args: update_entry())
        entry.bind("<Return>", update_var)
 
    add_stretch_slider("↔ Estiramiento X (%)", stretch_x)
    add_stretch_slider("↕ Estiramiento Y (%)", stretch_y)

    action_frame = ctk.CTkFrame(root, fg_color="transparent")
    action_frame.pack(pady=10)
    ctk.CTkButton(action_frame, text="Resetear valores", command=reset_defaults, width=180).pack(side="left", padx=15)
    root.mainloop()

if __name__ == "__main__":
    def delayed_virtual_cam_start():
        # Espera a que la GUI se inicialice antes de iniciar el loop de la cámara
        time.sleep(2)  # tiempo suficiente para que la GUI arranque
        start_preview_loop()

    iniciar_media_mtx_en_terminal()
    Thread(target=gui_thread, daemon=True).start()
    Thread(target=delayed_virtual_cam_start, daemon=True).start()

    while True:
        time.sleep(1)
