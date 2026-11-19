"""Windows Tk sender for the operator-enabled LAN ONNX inbox.

This program prepares an inactive model only. It never enables a model or
controls machinery. All network work runs outside the Tk UI thread.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

from model_transfer import (MAGIC, ModelReceiver, discover_receivers,
                            get_job, send_model)


def validate_model(path):
    from onnx_contract import inspect_yolo_onnx
    return inspect_yolo_onnx(Path(path))


def diagnose():
    """Local dependency check only: no discovery, credentials or board access."""
    import tkinter as tk
    from onnx_contract import inspect_yolo_onnx
    root = tk.Tk()
    try:
        root.withdraw()
        root.update_idletasks()
        version = root.tk.call('info', 'patchlevel')
    finally:
        root.destroy()
    return {'ok': True, 'python': sys.version.split()[0], 'tk': version,
            'onnx_contract': callable(inspect_yolo_onnx), 'protocol': MAGIC,
            'board_contacted': False}


def run_self_test():
    """Real loopback transport test; bytes are not an executed ONNX model."""
    result = diagnose()
    with tempfile.TemporaryDirectory() as directory:
        holder = {}

        def received(path, mode, job_id):
            holder['receiver'].update_job(job_id, 'ready', '本地传输自检完成')

        receiver = ModelReceiver(Path(directory) / 'inbox', received,
                                 host='127.0.0.1', port=0, discovery_port=0)
        holder['receiver'] = receiver
        try:
            receiver.start()
            receiver.allow(seconds=10)
            found = discover_receivers(timeout=.2, address='127.0.0.1',
                                         discovery_port=receiver.discovery_port)
            if len(found) != 1:
                raise RuntimeError('本地发现自检失败')
            source = Path(directory) / 'transport-only.onnx'
            source.write_bytes(b'loopback transport fixture, not an ONNX graph')
            job = send_model(found[0], source)
            state = get_job(found[0], job['job_id'])
            if state['state'] != 'ready':
                raise RuntimeError('本地状态轮询自检失败')
            result['transport_loopback'] = True
            result['model_executed'] = False
        finally:
            receiver.close()
    return result


class ModelSenderApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        self.root, self.tk, self.ttk = root, tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.messages = queue.Queue()
        self.receivers = []
        self.busy = False
        self.closed = threading.Event()
        self.file = tk.StringVar()
        self.ip = tk.StringVar()
        self.mode = tk.StringVar(value='特征图（现用软件默认）')
        self.status = tk.StringVar(value='先在板子上允许电脑发送，再选择文件并查找板子。')
        root.title('发送 ONNX 到 Jetson')
        root.geometry('730x530')
        root.minsize(650, 480)
        body = ttk.Frame(root, padding=20)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='把识别模型发送到板子', font=('Microsoft YaHei UI', 16)).pack(anchor='w')
        ttk.Label(body, text='① 板子上位机 → 模型管理 → 允许 Windows 发送（5分钟）\n'
                            '② 电脑与板子连接同一 Wi-Fi／热点，选择 ONNX 后发送。',
                  justify='left').pack(anchor='w', pady=(10, 16))
        file_row = ttk.Frame(body)
        file_row.pack(fill='x')
        ttk.Entry(file_row, textvariable=self.file, state='readonly').pack(side='left', fill='x', expand=True)
        self.choose = ttk.Button(file_row, text='选择 ONNX…', command=self.choose_file)
        self.choose.pack(side='left', padx=(8, 0))
        input_row = ttk.Frame(body)
        input_row.pack(fill='x', pady=(12, 3))
        ttk.Label(input_row, text='模型输入：').pack(side='left')
        self.mode_box = ttk.Combobox(input_row, textvariable=self.mode, state='readonly', width=31,
                                      values=('特征图（现用软件默认）', '原始 RGB（对应训练模型）'))
        self.mode_box.pack(side='left')
        ttk.Label(body, text='请按训练时的输入方式选择；ONNX 文件无法自动证明预处理方式。').pack(anchor='w')
        ip_row = ttk.Frame(body)
        ip_row.pack(fill='x', pady=(16, 6))
        ttk.Label(ip_row, text='板子 IP（可留空）：').pack(side='left')
        self.ip_entry = ttk.Entry(ip_row, textvariable=self.ip, width=22)
        self.ip_entry.pack(side='left')
        self.scan = ttk.Button(ip_row, text='查找板子', command=self.find_receivers)
        self.scan.pack(side='left', padx=8)
        self.listbox = tk.Listbox(body, height=3, exportselection=False)
        self.listbox.pack(fill='x')
        self.progress = ttk.Progressbar(body, maximum=100)
        self.progress.pack(fill='x', pady=(16, 8))
        ttk.Label(body, textvariable=self.status, wraplength=660, justify='left').pack(anchor='w')
        self.send = ttk.Button(body, text='发送并转换', command=self.send_selected)
        self.send.pack(anchor='e', pady=(14, 6))
        ttk.Label(body, text='转换完成后，到板子模型管理中手动启用。关闭本工具不会取消板端转换。',
                  wraplength=660).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.report_callback_exception = lambda _kind, error, _traceback: self.messagebox.showerror(
            '未能完成', str(error), parent=root)
        root.after(80, self.drain_messages)

    def choose_file(self):
        selected = self.filedialog.askopenfilename(title='选择 YOLO ONNX 模型',
                                                    filetypes=(('ONNX 模型', '*.onnx'),))
        if selected:
            self.file.set(selected)

    def set_busy(self, busy):
        self.busy = busy
        state = 'disabled' if busy else 'normal'
        for widget in (self.choose, self.scan, self.send, self.ip_entry, self.listbox):
            widget.configure(state=state)
        self.mode_box.configure(state='disabled' if busy else 'readonly')

    def background(self, operation):
        self.set_busy(True)

        def run():
            try:
                operation()
            except Exception as error:
                # Only user-facing messages are queued; never enqueue endpoints
                # or HTTP headers containing admission capabilities.
                self.messages.put(('error', str(error)))
            finally:
                self.messages.put(('idle', None))

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    def find_receivers(self):
        address = self.ip.get().strip() or None
        self.status.set('正在查找已允许接收的板子…')

        def scan():
            self.messages.put(('receivers', discover_receivers(address=address)))

        self.background(scan)

    def send_selected(self):
        selected = self.listbox.curselection()
        if not self.file.get():
            self.messagebox.showinfo('选择模型', '请先选择一个 .onnx 模型文件。')
            return
        if not selected:
            self.messagebox.showinfo('选择板子', '请先查找板子并选择接收设备。')
            return
        endpoint = dict(self.receivers[int(selected[0])])
        path = Path(self.file.get())
        mode = 'legacy_features' if self.mode.get().startswith('特征图') else 'raw_rgb'
        self.progress['value'] = 0
        self.status.set('正在检查 ONNX 结构和类别信息…')

        def upload():
            contract = validate_model(path)
            self.messages.put(('status', '结构检查通过：{} 类，开始传输…'.format(len(contract['names']))))
            last = [-1]

            def progress(sent, total):
                percent = int(sent * 100 / total)
                if percent != last[0]:
                    self.messages.put(('progress', percent))
                    last[0] = percent

            job = send_model(endpoint, path, mode, on_progress=progress)
            self.messages.put(('status', '上传完成，板子正在检查并转换模型…'))
            deadline = time.monotonic() + 3600
            while not self.closed.wait(1):
                try:
                    current = get_job(endpoint, job['job_id'])
                except (OSError, ValueError):
                    self.messages.put(('status', '状态连接中断，正在重试；板端转换继续进行。'))
                    if time.monotonic() < deadline:
                        continue
                    raise RuntimeError('未能取得转换结果，请到板子模型管理中查看。')
                state = current.get('state')
                self.messages.put(('status', current.get('message') or '板子正在准备模型…'))
                if state == 'ready':
                    self.messages.put(('ready', '模型已准备完成。请到板子模型管理中手动启用，并现场核对识别效果。'))
                    return
                if state in ('failed', 'cancelled'):
                    raise RuntimeError(current.get('message') or '板端模型准备失败，请查看板端诊断。')
                if time.monotonic() >= deadline:
                    raise RuntimeError('模型仍在准备中，请到板子查看；本工具不会取消转换。')

        self.background(upload)

    def drain_messages(self):
        if self.closed.is_set():
            return
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == 'receivers':
                self.receivers = value
                self.listbox.configure(state='normal')
                self.listbox.delete(0, 'end')
                for endpoint in value:
                    self.listbox.insert('end', '板子 {}   接收许可剩余约 {} 秒'.format(
                        endpoint['host'], endpoint.get('expires_in', '?')))
                if len(value) == 1:
                    self.listbox.selection_set(0)
                if self.busy:
                    self.listbox.configure(state='disabled')
                self.status.set('已找到 {} 台板子，请选择后发送。'.format(len(value)) if value else
                                '未找到接收设备。请在板子上允许接收；确认同一网络，或填写板子当前 IP 再查找。')
            elif kind == 'progress':
                self.progress['value'] = value
                self.status.set('正在上传：{}%'.format(value))
            elif kind == 'status':
                self.status.set(value)
            elif kind == 'ready':
                self.status.set(value)
                self.messagebox.showinfo('模型已准备', value)
            elif kind == 'error':
                self.status.set(value)
                self.messagebox.showerror('未能完成', value)
            elif kind == 'idle':
                self.set_busy(False)
        self.root.after(80, self.drain_messages)

    def close(self):
        self.closed.set()
        self.root.destroy()


def argument_parser():
    parser = argparse.ArgumentParser(description='YLERS 局域网模型发送工具')
    parser.add_argument('--diagnose', action='store_true', help='检查 Tk 和 ONNX 检查器，不连接板子')
    parser.add_argument('--self-test', action='store_true', help='运行本机回环传输自检，不连接板子')
    parser.add_argument('--report', help='将自检 JSON 写到指定文件，适用于无控制台的打包程序')
    return parser


def _show_fatal(message):
    text = '模型发送工具无法启动：\n' + message + '\n请使用完整工具包，或联系维护人员。'
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('无法启动', text, parent=root)
        root.destroy()
    except Exception:
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, text, 'YLERS 模型发送工具', 0x10)
                return
            except Exception:
                pass
        print(text, file=sys.stderr)


def main(argv=None):
    args = argument_parser().parse_args(argv)
    try:
        if args.diagnose or args.self_test:
            result = run_self_test() if args.self_test else diagnose()
            payload = json.dumps(result, ensure_ascii=False)
            if args.report:
                Path(args.report).write_text(payload, encoding='utf-8')
            if sys.stdout is not None:
                print(payload)
            return 0
        import tkinter as tk
        from onnx_contract import inspect_yolo_onnx  # fail clearly before opening the workflow
        root = tk.Tk()
        ModelSenderApp(root)
        root.mainloop()
        return 0
    except Exception as error:
        if args.diagnose or args.self_test:
            payload = json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False)
            if args.report:
                try:
                    Path(args.report).write_text(payload, encoding='utf-8')
                except OSError:
                    pass
            if sys.stdout is not None:
                print(payload)
        else:
            _show_fatal(str(error))
        return 1


if __name__ == '__main__':
    sys.exit(main())
