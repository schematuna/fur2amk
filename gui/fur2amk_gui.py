import tkinter as tk
from tkinter import ttk, filedialog
import threading
import logging
import json
import os
import re
import shutil
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fur2amk import run_conversion


def _resource_path(relative_path):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)

_PATH_RE = re.compile(r'^\s*#path\s*"([^"]+)"', re.IGNORECASE)


def _app_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))


def _load_state():
    try:
        with open(os.path.join(_app_dir(), 'fur2amk_gui_state.json'), 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(fur_path, amk_dir):
    try:
        with open(os.path.join(_app_dir(), 'fur2amk_gui_state.json'), 'w') as f:
            json.dump({'fur_path': fur_path, 'amk_dir': amk_dir}, f)
    except OSError:
        pass


def _extract_mml_path(mml_path):
    try:
        with open(mml_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                m = _PATH_RE.match(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def _copy_to_amk(mml_path, sample_dir, song_name, amk_dir):
    amk_dir = os.path.abspath(amk_dir)
    if not os.path.isdir(amk_dir):
        logging.error(f'AMK dir not found: {amk_dir}')
        return
    path_str = _extract_mml_path(mml_path) or song_name
    music_dst = os.path.join(amk_dir, 'music')
    samples_dst = os.path.join(amk_dir, 'samples', path_str)
    os.makedirs(music_dst, exist_ok=True)
    os.makedirs(samples_dst, exist_ok=True)
    shutil.copy2(mml_path, os.path.join(music_dst, os.path.basename(mml_path)))
    logging.info(f'Copied MML to {music_dst}')
    if sample_dir and os.path.isdir(sample_dir):
        for brr in glob(os.path.join(sample_dir, '*.brr')):
            shutil.copy2(brr, os.path.join(samples_dst, os.path.basename(brr)))
        logging.info(f'Copied BRRs to {samples_dst}')


class TextWidgetHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        self.widget.after(0, self._append, self.format(record) + '\n')

    def _append(self, msg):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, msg)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('fur2amk')
        self.resizable(True, True)
        icon_path = _resource_path('fuzzy.ico')
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        state = _load_state()
        self._build_ui(state)

    def report_callback_exception(self, exc_type, exc_val, exc_tb):
        import traceback
        from tkinter import messagebox
        messagebox.showerror('Unexpected error', ''.join(traceback.format_exception(exc_type, exc_val, exc_tb)))

    def _build_ui(self, state):
        pad = {'padx': 10, 'pady': 5}

        file_frame = ttk.Frame(self)
        file_frame.pack(fill='x', **pad)
        ttk.Label(file_frame, text='Furnace file:').pack(side='left')
        self.file_var = tk.StringVar(value=state.get('fur_path', ''))
        ttk.Entry(file_frame, textvariable=self.file_var, width=50).pack(side='left', padx=5)
        ttk.Button(file_frame, text='Browse...', command=self._browse).pack(side='left')

        amk_frame = ttk.Frame(self)
        amk_frame.pack(fill='x', **pad)
        ttk.Label(amk_frame, text='AMK directory:').pack(side='left')
        self.amk_var = tk.StringVar(value=state.get('amk_dir', ''))
        ttk.Entry(amk_frame, textvariable=self.amk_var, width=50).pack(side='left', padx=5)
        ttk.Button(amk_frame, text='Browse...', command=self._browse_amk).pack(side='left')

        opt_frame = ttk.Frame(self)
        opt_frame.pack(fill='x', **pad)
        self.nosmpl_var = tk.BooleanVar()
        self.verbose_var = tk.BooleanVar()
        self.noopt_var = tk.BooleanVar()
        ttk.Checkbutton(opt_frame, text='Skip samples', variable=self.nosmpl_var).pack(side='left')
        ttk.Checkbutton(opt_frame, text='Verbose', variable=self.verbose_var).pack(side='left', padx=10)
        ttk.Checkbutton(opt_frame, text="Don't optimize loops", variable=self.noopt_var).pack(side='left')

        self.convert_btn = ttk.Button(self, text='Convert', command=self._start_convert)
        self.convert_btn.pack(**pad)

        log_frame = ttk.Frame(self)
        log_frame.pack(fill='both', expand=True, **pad)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, width=70, height=15, state='disabled', wrap='none')
        vsb = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        hsb = ttk.Scrollbar(log_frame, orient='horizontal', command=self.log.xview)
        self.log.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.log.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[('Furnace files', '*.fur'), ('All files', '*.*')])
        if path:
            self.file_var.set(path)

    def _browse_amk(self):
        path = filedialog.askdirectory()
        if path:
            self.amk_var.set(path)

    def _start_convert(self):
        fur_path = self.file_var.get().strip()
        if not fur_path or not os.path.exists(fur_path):
            self.log.configure(state='normal')
            self.log.insert(tk.END, 'Error: please select a valid .fur file.\n')
            self.log.configure(state='disabled')
            return
        self.convert_btn.configure(state='disabled')
        self.log.configure(state='normal')
        self.log.delete('1.0', tk.END)
        self.log.configure(state='disabled')
        threading.Thread(target=self._run_convert, args=(fur_path,), daemon=True).start()

    def _run_convert(self, fur_path):
        handler = TextWidgetHandler(self.log)
        handler.setFormatter(logging.Formatter('%(levelname)-7s %(message)s'))
        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(logging.DEBUG if self.verbose_var.get() else logging.INFO)

        try:
            song_name = os.path.splitext(os.path.basename(fur_path))[0]
            script_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.join(script_dir, 'music')

            out_path, sample_dir = run_conversion(fur_path, out_dir, self.nosmpl_var.get(), optimize_loops=not self.noopt_var.get())

            amk_dir = self.amk_var.get().strip()
            if amk_dir:
                _copy_to_amk(out_path, sample_dir, song_name, amk_dir)
                logging.info(f'Done! Output: {os.path.abspath(amk_dir)}')
            else:
                logging.info(f'Done! Output: {out_dir}')
            _save_state(fur_path, amk_dir or '')
        except Exception as e:
            logging.error(f'Conversion failed: {e}')
        finally:
            self.after(0, lambda: self.convert_btn.configure(state='normal'))


if __name__ == '__main__':
    App().mainloop()
