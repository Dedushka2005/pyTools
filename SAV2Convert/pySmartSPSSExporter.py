import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import pyreadstat
import re

# Проверяем наличие библиотеки openpyxl для работы с Excel
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class SpssConverterApp:

    def __init__(self, root):
        self.root = root
        self.root.title("SPSS в Fixed ASCII Конвертер")
        self.root.geometry("520x510")  # Высота увеличена под все 5 кнопок управления
        self.root.resizable(False, False)

        self.selected_file_path = ""
        self.df = None  
        self.string_df = None  
        self.meta = None  # Хранилище метаданных SPSS (включая Value Labels)
        self.valid_variables = []  
        self.string_variables = []  # Список доступных текстовых переменных
        self.excluded_variables = set()  
        self.excluded_strings = set()  # Список исключенных текстовых переменных для Excel
        self.final_lengths = {}
        self.cleaned_char_dict = {}
        self.string_counts = {}  # Словарь для хранения количества заполненных ответов
        self.convert_lay_names_var = tk.BooleanVar(value=False)  # По умолчанию выключено

        self.create_main_gui()
        self.check_excel_library()

    def create_main_gui(self):
        # Главный заголовок окна
        self.lbl_title = tk.Label(
            self.root,
            text="Конвертер SPSS (.sav) файлов",
            font=("Arial", 14, "bold"),
        )
        self.lbl_title.pack(pady=(15, 5))

        # Статус-бар вверху формы
        self.lbl_status = tk.Label(
            self.root, 
            text="Файл не выбран", 
            font=("Arial", 10, "italic"), 
            wraplength=450
        )
        self.lbl_status.pack(pady=(0, 15))

        # 1. Выбор файла
        self.btn_select = tk.Button(
            self.root,
            text="1. Выбрать исходный .sav файл",
            command=self.select_file,
            width=40,
            font=("Arial", 10, "bold"),
            bg="#34495e",
            activebackground="#2c3e50",
            fg="white",
            activeforeground="white",
            bd=0,
            relief=tk.FLAT
        )
        self.btn_select.pack(pady=5)

        # 2a. Настройка структуры ASCII
        self.btn_configure = tk.Button(
            self.root,
            text="2a. Настроить переменные ASCII",
            command=self.show_variable_selection_window,
            state=tk.DISABLED,
            width=40,
            font=("Arial", 10, "bold"),
            bg="#3498db",
            activebackground="#2980b9",
            fg="white",
            activeforeground="white",
            disabledforeground="#aaaaaa",
            bd=0,
            relief=tk.FLAT
        )
        self.btn_configure.pack(pady=5)

        # 2b. Настройка структуры Excel
        self.btn_configure_excel = tk.Button(
            self.root,
            text="2b. Настроить строки для Excel",
            command=self.show_string_selection_window,
            state=tk.DISABLED,
            width=40,
            font=("Arial", 10, "bold"),
            bg="#16a085",
            activebackground="#1abc9c",
            fg="white",
            activeforeground="white",
            disabledforeground="#aaaaaa",
            bd=0,
            relief=tk.FLAT
        )
        self.btn_configure_excel.pack(pady=5)

        # Флажок для преобразования имен в .lay файле
        self.chk_convert_names = tk.Checkbutton(
            self.root,
            text="Преобразовывать имена переменных в .lay файле (через 'xr')",
            variable=self.convert_lay_names_var,
            font=("Arial", 9),
            anchor="w"
        )
        self.chk_convert_names.pack(pady=4)

        # 3. Экспорт ASCII
        self.btn_start = tk.Button(
            self.root,
            text="3. Запустить экспорт ASCII",
            command=self.start_export_thread, 
            state=tk.DISABLED,
            width=40,
            font=("Arial", 10, "bold"),
            bg="#2ecc71",
            activebackground="#27ae60",
            fg="white",
            activeforeground="white",
            disabledforeground="#aaaaaa",
            bd=0,
            relief=tk.FLAT
        )
        self.btn_start.pack(pady=5)

        # 4. Выгрузка чисто текстовых строк в Excel
        self.btn_export_excel = tk.Button(
            self.root,
            text="4. Выгрузить текстовые строки в Excel",
            command=self.export_strings_to_excel,
            state=tk.DISABLED,
            width=40,
            font=("Arial", 10, "bold"),
            bg="#9b59b6",
            activebackground="#8e44ad",
            fg="white",
            activeforeground="white",
            disabledforeground="#aaaaaa",
            bd=0,
            relief=tk.FLAT
        )
        self.btn_export_excel.pack(pady=5)

        # 5. Новая кнопка: Полная выгрузка с метками (Value Labels)
        self.btn_export_labels = tk.Button(
            self.root,
            text="5. Выгрузить все данные с метками в Excel",
            command=self.export_all_with_labels_to_excel,
            state=tk.DISABLED,
            width=40,
            font=("Arial", 10, "bold"),
            bg="#e67e22",
            activebackground="#d35400",
            fg="white",
            activeforeground="white",
            disabledforeground="#aaaaaa",
            bd=0,
            relief=tk.FLAT
        )
        self.btn_export_labels.pack(pady=5)

        # Прогресс-бар в самом низу
        self.progress = ttk.Progressbar(
            self.root, orient=tk.HORIZONTAL, length=350, mode="determinate"
        )
        self.progress.pack(pady=(15, 10))

    def check_excel_library(self):
        """Проверяет окружение на готовность к работе с Excel при запуске"""
        if not HAS_OPENPYXL:
            self.lbl_status.config(
                text="Внимание: библиотека 'openpyxl' не найдена.\nВыгрузка в Excel будет недоступна. Установите её через pip.",
                fg="red"
            )

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("SPSS Files", "*.sav")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.btn_select.config(state=tk.DISABLED)
            self.btn_configure.config(state=tk.DISABLED)
            self.btn_configure_excel.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_export_excel.config(state=tk.DISABLED)
            self.btn_export_labels.config(state=tk.DISABLED)
            
            threading.Thread(target=self._load_file_worker, args=(file_path,), daemon=True).start()

    def _load_file_worker(self, file_path):
        try:
            self._update_progress(15, "Шаг 1/4: Чтение файла с диска...")
            df_raw, self.meta = pyreadstat.read_sav(file_path)

            old_first_col = df_raw.columns[0]
            df_raw.rename(columns={old_first_col: "Rid"}, inplace=True)

            self._update_progress(45, "Шаг 2/4: Анализ типов данных переменных...")
            valid_cols = ["Rid"]  
            string_cols = []      

            for col in df_raw.columns:
                if col == "Rid":
                    continue

                non_null_vals = df_raw[col].dropna().astype(str).str.strip()
                non_null_vals = non_null_vals[non_null_vals != ""]

                # Проверка на числовой тип данных самого SPSS (работает даже для полностью пустых колонок)
                if pd.api.types.is_numeric_dtype(df_raw[col]):
                    valid_cols.append(col)
                else:
                    # Если колонка строковая в SPSS и ПУСТАЯ — мы её пропускаем (как и раньше для Excel)
                    if non_null_vals.empty:
                        continue
                        
                    is_numeric_string = non_null_vals.str.match(r"^-?[0-9]+([.,][0-9]+)?$").all()
                    if is_numeric_string:
                        df_raw[col] = df_raw[col].astype(str).str.replace(",", ".", regex=False)
                        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
                        valid_cols.append(col)
                    else:
                        string_cols.append(col)
                        self.string_counts[col] = len(non_null_vals)

            if len(valid_cols) <= 1:
                raise ValueError("Числовые переменные (кроме Rid) не найдены.")

            self.string_variables = string_cols
            self.excluded_strings.clear()

            if string_cols:
                self.string_df = df_raw[["Rid"] + string_cols].copy()
            else:
                self.string_df = df_raw[["Rid"]].copy()

            self.df = df_raw[valid_cols].copy()
            self.valid_variables = [col for col in self.df.columns]
            self.excluded_variables.clear()

            self._update_progress(75, "Шаг 3/4: Очистка чисел и расчет ширины колонок...")
            def clean_num_to_char(series):
                def format_val(val):
                    if pd.isna(val): return ""
                    if val == int(val): return str(int(val))
                    return f"{val}".rstrip("0").rstrip(".")
                return series.apply(format_val)

            initial_lengths = {}
            self.cleaned_char_dict = {}

            for col in self.valid_variables:
                char_series = clean_num_to_char(self.df[col])
                self.cleaned_char_dict[col] = char_series
                max_len = char_series.str.len().max()
                # ИСПРАВЛЕНИЕ: Если колонка пустая, max_len будет NaN или 0. Задаем ей длину 1.
                initial_lengths[col] = int(max_len) if pd.notna(max_len) and max_len > 0 else 1

            self._update_progress(90, "Шаг 4/4: Выравнивание длин по префиксам...")
            self.final_lengths = initial_lengths.copy()
            prefixes = {col: col.split("_")[0] for col in self.valid_variables}
            unique_prefixes = set(prefixes.values())

            for pref in unique_prefixes:
                matched_cols = [col for col in self.valid_variables if prefixes[col] == pref]
                if matched_cols:
                    max_len_in_group = max(initial_lengths[col] for col in matched_cols)
                    for col in matched_cols:
                        self.final_lengths[col] = max_len_in_group

            self.root.after(0, self._loading_finished_successfully)

        except Exception as e:
            self.root.after(0, lambda err=e: self._loading_failed(err))

    def _update_progress(self, value, text, color="blue"):
        self.root.after(0, lambda: self.progress.configure(value=value))
        self.root.after(0, lambda: self.lbl_status.configure(text=text, fg=color))

    def _loading_finished_successfully(self):
        self.progress.configure(value=100)
        self.lbl_status.configure(text="Файл успешно загружен.", fg="green")
        self._unlock_main_buttons()
        self.show_variable_selection_window()

    def _loading_failed(self, error):
        self.progress.configure(value=0)
        self.lbl_status.configure(text="Ошибка загрузки", fg="red")
        messagebox.showerror("Ошибка чтения", f"Не удалось прочитать файл:\n{error}")
        self.btn_select.config(state=tk.NORMAL)
        self.check_excel_library()

    def show_variable_selection_window(self):
        active_vars = [v for v in self.valid_variables if v not in self.excluded_variables]

        start_positions = []
        end_positions = []
        current_pos = 1

        for col in active_vars:
            length = self.final_lengths[col]
            start_positions.append(current_pos)
            end_positions.append(current_pos + length - 1)
            current_pos += length

        ranges = [f"0-{'9' * self.final_lengths[col]}" for col in active_vars]

        sel_window = tk.Toplevel(self.root)
        sel_window.title("Выбор переменных для выгрузки ASCII")
        sel_window.geometry("600x500")
        sel_window.transient(self.root)
        sel_window.grab_set()

        lbl_info = tk.Label(
            sel_window,
            text="Выделите переменные, которые хотите ИСКЛЮЧИТЬ, и нажмите красную кнопку.",
            font=("Arial", 10, "bold"),
            wraplength=560,
        )
        lbl_info.pack(pady=10)

        frame_table = tk.Frame(sel_window)
        frame_table.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        columns = ("variable", "start", "end", "length", "range")
        tree = ttk.Treeview(frame_table, columns=columns, show="headings", selectmode="extended")

        tree.heading("variable", text="Переменная")
        tree.heading("start", text="Начало")
        tree.heading("end", text="Конец")
        tree.heading("length", text="Длина")
        tree.heading("range", text="Диапазон")

        tree.column("variable", width=150, anchor="w")
        tree.column("start", width=70, anchor="center")
        tree.column("end", width=70, anchor="center")
        tree.column("length", width=70, anchor="center")
        tree.column("range", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(frame_table, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, col in enumerate(active_vars):
            tree.insert("", tk.END, iid=col, values=(col, start_positions[i], end_positions[i], self.final_lengths[col], ranges[i]))

        def exclude_selected():
            selected_items = tree.selection()
            if not selected_items:
                messagebox.showinfo("Информация", "Пожалуйста, выделите строки для удаления.")
                return

            if "Rid" in selected_items:
                messagebox.showwarning("Внимание", "Переменную 'Rid' нельзя исключить!")
                selected_items = [item for item in selected_items if item != "Rid"]
                if not selected_items: return

            for item in selected_items:
                self.excluded_variables.add(item)
                tree.delete(item)

        frame_actions = tk.Frame(sel_window)
        frame_actions.pack(pady=5)

        btn_exclude = tk.Button(
            frame_actions,
            text="Исключить выделенные",
            command=exclude_selected,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10),
            width=22,
            bd=0,
            relief=tk.FLAT
        )
        btn_exclude.pack(side=tk.LEFT, padx=5)

        def reset_exceptions():
            if not self.excluded_variables:
                messagebox.showinfo("Информация", "Структура уже находится в первоначальном виде.")
                return

            self.excluded_variables.clear()
            for item in tree.get_children():
                tree.delete(item)

            full_start_positions = []
            full_end_positions = []
            current_pos = 1

            for col in self.valid_variables:
                length = self.final_lengths[col]
                full_start_positions.append(current_pos)
                full_end_positions.append(current_pos + length - 1)
                current_pos += length

            full_ranges = [f"0-{'9' * self.final_lengths[col]}" for col in self.valid_variables]

            for i, col in enumerate(self.valid_variables):
                tree.insert("", tk.END, iid=col, values=(col, full_start_positions[i], full_end_positions[i], self.final_lengths[col], full_ranges[i]))
            messagebox.showinfo("Успех", "Структура переменных возвращена к первоначальному виду.")

        btn_reset = tk.Button(
            frame_actions,
            text="Сбросить исключения",
            command=reset_exceptions,
            bg="#f39c12",
            fg="white",
            font=("Arial", 10),
            width=22,
            bd=0,
            relief=tk.FLAT
        )
        btn_reset.pack(side=tk.LEFT, padx=5)

        def confirm_selection():
            remaining = list(tree.get_children())
            if not remaining:
                messagebox.showwarning("Внимание", "Должна остаться хотя бы одна переменная!")
                return
            sel_window.destroy()
            self.lbl_status.config(
                text=f"Файл готов к сборке.\nПеременных к выгрузке: {len(remaining)} (Исключено: {len(self.excluded_variables)})",
                fg="black",
            )

        btn_confirm = tk.Button(
            sel_window,
            text="Сохранить структуру и закрыть",
            command=confirm_selection,
            bg="#3498db",
            fg="white",
            width=30,
            font=("Arial", 10, "bold"),
            bd=0,
            relief=tk.FLAT
        )
        btn_confirm.pack(pady=15)

    def show_string_selection_window(self):
        """Окно выбора текстовых переменных для выгрузки в Excel с измененным порядком колонок"""
        if not self.string_variables:
            messagebox.showinfo("Информация", "В файле нет текстовых переменных для настройки (только Rid).")
            return

        active_strings = [s for s in self.string_variables if s not in self.excluded_strings]

        sel_window = tk.Toplevel(self.root)
        sel_window.title("Выбор текстовых переменных для Excel")
        sel_window.geometry("620x450")
        sel_window.transient(self.root)
        sel_window.grab_set()

        lbl_info = tk.Label(
            sel_window,
            text="Выделите переменные, которые ХОТИТЕ ИСКЛЮЧИТЬ из Excel, и нажмите красную кнопку.\nПолностью пустые столбцы уже удалены автоматически.",
            font=("Arial", 9, "bold"),
            wraplength=580,
            fg="#2c3e50"
        )
        lbl_info.pack(pady=10)

        frame_table = tk.Frame(sel_window)
        frame_table.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        columns = ("variable", "count", "type")
        tree = ttk.Treeview(frame_table, columns=columns, show="headings", selectmode="extended")

        tree.heading("variable", text="Текстовая переменная")
        tree.heading("count", text="Заполнено ответов")
        tree.heading("type", text="Тип данных")

        tree.column("variable", width=250, anchor="w")
        tree.column("count", width=140, anchor="center")
        tree.column("type", width=140, anchor="center")

        scrollbar = ttk.Scrollbar(frame_table, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for col in active_strings:
            count_vals = self.string_counts.get(col, 0)
            tree.insert("", tk.END, iid=col, values=(col, count_vals, "Текст (String)"))

        def exclude_selected_strings():
            selected_items = tree.selection()
            if not selected_items:
                messagebox.showinfo("Информация", "Пожалуйста, выделите строки для удаления.")
                return

            for item in selected_items:
                self.excluded_strings.add(item)
                tree.delete(item)

        frame_actions = tk.Frame(sel_window)
        frame_actions.pack(pady=5)

        btn_exclude = tk.Button(
            frame_actions,
            text="Исключить выделенные",
            command=exclude_selected_strings,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10),
            width=22,
            bd=0,
            relief=tk.FLAT
        )
        btn_exclude.pack(side=tk.LEFT, padx=5)

        def reset_string_exceptions():
            self.excluded_strings.clear()
            for item in tree.get_children():
                tree.delete(item)
            for col in self.string_variables:
                count_vals = self.string_counts.get(col, 0)
                tree.insert("", tk.END, iid=col, values=(col, count_vals, "Текст (String)"))
            messagebox.showinfo("Успех", "Все текстовые переменные возвращены в список выгрузки.")

        btn_reset = tk.Button(
            frame_actions,
            text="Сбросить исключения",
            command=reset_string_exceptions,
            bg="#f39c12",
            fg="white",
            font=("Arial", 10),
            width=22,
            bd=0,
            relief=tk.FLAT
        )
        btn_reset.pack(side=tk.LEFT, padx=5)

        def confirm_string_selection():
            sel_window.destroy()
            remaining_count = len(self.string_variables) - len(self.excluded_strings)
            self.lbl_status.config(
                text=f"Файл готов к сборке Excel.\nСтрок к выгрузке: {remaining_count} (Исключено вручную: {len(self.excluded_strings)})",
                fg="black",
            )

        btn_confirm = tk.Button(
            sel_window,
            text="Сохранить структуру Excel и закрыть",
            command=confirm_string_selection,
            bg="#16a085",
            fg="white",
            width=35,
            font=("Arial", 10, "bold"),
            bd=0,
            relief=tk.FLAT
        )
        btn_confirm.pack(pady=15)

    def start_export_thread(self):
        self.btn_start.config(state=tk.DISABLED)
        self.btn_configure.config(state=tk.DISABLED)
        self.btn_configure_excel.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        self.btn_export_excel.config(state=tk.DISABLED)
        if hasattr(self, 'btn_export_labels'):
            self.btn_export_labels.config(state=tk.DISABLED)
        self.progress.configure(value=0)

        threading.Thread(target=self._export_worker, daemon=True).start()

    def _export_worker(self):
        try:
            self._update_progress(20, "Экспорт: Фильтрация матриц...")
            export_vars = [v for v in self.valid_variables if v not in self.excluded_variables]
            df_final = self.df[export_vars].copy()

            file_dir = os.path.dirname(self.selected_file_path)
            file_base, _ = os.path.splitext(os.path.basename(self.selected_file_path))

            dat_path = os.path.join(file_dir, f"{file_base}_exp.dat")
            lay_path = os.path.join(file_dir, f"{file_base}_exp.lay")

            start_positions = []
            end_positions = []
            current_pos = 1

            for col in export_vars:
                length = self.final_lengths[col]
                start_positions.append(current_pos)
                end_positions.append(current_pos + length - 1)
                current_pos += length

            ranges = [f"0-{'9' * self.final_lengths[col]}" for col in export_vars]

            # Блок опционального преобразования имен для .lay файла
            lay_vars_output = []
            if self.convert_lay_names_var.get():
                import re
                for v in export_vars:
                    # Правило 1: Имя_Число1.Число2 -> ИмяxrЧисло2_Число1
                    # Строго ОДНО подчёркивание в имени, ровно ОДНА точка и числа на конце
                    if re.match(r"^([^\._]+)_(\d+)\.(\d+)$", v):
                        v_new = re.sub(r"^([^\._]+)_(\d+)\.(\d+)$", r"\1xr\3_\2", v)
                        lay_vars_output.append(v_new)
                    
                    # Правило 2: Имя.Число1 -> ИмяxrЧисло1
                    # Строго БЕЗ подчёркиваний и БЕЗ лишних точек в имени, ровно ОДНА точка и числа на конце
                    elif re.match(r"^([^\._]+)\.(\d+)$", v):  # ИСПРАВЛЕНО: добавлено _ в исключения
                        v_new = re.sub(r"^([^\._]+)\.(\d+)$", r"\1xr\2", v)
                        lay_vars_output.append(v_new)
                    
                    # Все остальные переменные (две точки, два подчёркивания + точка и т.д.) не трогаем
                    else:
                        lay_vars_output.append(v)
            else:
                lay_vars_output = export_vars

            # Формируем структуру .lay с обработанными именами колонок
            layout_df = pd.DataFrame({
                "Variable": lay_vars_output,
                "Start": start_positions,
                "End": end_positions,
                "Length": [self.final_lengths[v] for v in export_vars],
                "Range": ranges,
            })

            self._update_progress(60, "Экспорт: Форматирование текстовых строк fixed ASCII...")
            formatted_cols = []
            for col in export_vars:
                length = self.final_lengths[col]
                padded = self.cleaned_char_dict[col].str.rjust(length)
                formatted_cols.append(padded)

            ascii_data = pd.concat(formatted_cols, axis=1).apply("".join, axis=1)

            self._update_progress(85, "Экспорт: Сохранение файлов на диск в ANSI...")
            with open(dat_path, "w", encoding="cp1251") as f_dat:
                f_dat.write("\n".join(ascii_data) + "\n")

            layout_df.to_csv(lay_path, sep="\t", index=False, header=False, encoding="cp1251")

            self.root.after(0, self._export_finished_successfully)

        except Exception as e:
            self.root.after(0, lambda err=e: self._export_failed(err))

    def export_strings_to_excel(self):
        if not HAS_OPENPYXL:
            messagebox.showerror("Ошибка системы", "Модуль 'openpyxl' отсутствует. Экспорт невозможен.")
            return

        export_strings = [s for s in self.string_variables if s not in self.excluded_strings]
        columns_to_excel = ["Rid"] + export_strings

        if self.string_df is None or len(columns_to_excel) <= 1:
            messagebox.showinfo("Информация", "Нет выбранных текстовых строк для выгрузки (или выбран только Rid).")
            return

        file_base, _ = os.path.splitext(os.path.basename(self.selected_file_path))
        default_filename = f"{file_base}_open.xlsx"

        file_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")]
        )

        if file_path:
            if not file_path.endswith("_open.xlsx"):
                base_path, _ = os.path.splitext(file_path)
                file_path = f"{base_path}_open.xlsx"

            try:
                final_excel_df = self.string_df[columns_to_excel].copy()
                final_excel_df.to_excel(file_path, index=False)
                messagebox.showinfo("Успех", f"Текстовые строки успешно сохранены в файл:\n{os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить Excel файл:\n{e}")

    def export_all_with_labels_to_excel(self):
        """Безопасный запуск фонового потока для выгрузки всех данных с метками"""
        if not HAS_OPENPYXL:
            messagebox.showerror("Ошибка системы", "Модуль 'openpyxl' отсутствует. Экспорт невозможен.")
            return

        if self.selected_file_path == "" or self.meta is None:
            messagebox.showinfo("Информация", "Исходный файл не загружен или не содержит метаданных.")
            return

        file_base, _ = os.path.splitext(os.path.basename(self.selected_file_path))
        default_filename = f"{file_base}_labels.xlsx"

        file_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")]
        )

        if file_path:
            # Блокируем интерфейс на время выгрузки
            self.btn_start.config(state=tk.DISABLED)
            self.btn_configure.config(state=tk.DISABLED)
            self.btn_configure_excel.config(state=tk.DISABLED)
            self.btn_select.config(state=tk.DISABLED)
            self.btn_export_excel.config(state=tk.DISABLED)
            if hasattr(self, 'btn_export_labels'):
                self.btn_export_labels.config(state=tk.DISABLED)
            self.progress.configure(value=0)

            # Запускаем тяжелую работу в фоновом потоке, чтобы интерфейс НЕ зависал
            threading.Thread(
                target=self._export_labels_worker, 
                args=(file_path,), 
                daemon=True
            ).start()

    def _export_labels_worker(self, file_path):
        """Фоновый метод обработки меток по колонкам с детальным прогрессом"""
        try:
            self._update_progress(10, "Excel (Метки): Чтение всех столбцов файла...")
            df_all, _ = pyreadstat.read_sav(self.selected_file_path)
            
            # Переименовываем первую колонку в Rid до обработки
            old_first_col = df_all.columns[0]
            df_all.rename(columns={old_first_col: "Rid"}, inplace=True)

            # Получаем словарь меток из метаданных
            value_labels = self.meta.variable_value_labels if self.meta else {}
            total_cols = len(df_all.columns)

            self._update_progress(20, f"Excel (Метки): Подготовка к обработке {total_cols} колонок...")

            # Поколоночный цикл для отображения точного прогресса
            for index, col in enumerate(df_all.columns, start=1):
                # Если для колонки есть словарь меток в SPSS, подставляем текстовые значения
                if col in value_labels and value_labels[col]:
                    # Используем .map для замены кодов на текстовые метки, сохраняя ненайденные значения
                    df_all[col] = df_all[col].map(value_labels[col]).fillna(df_all[col])

                # Рассчитываем плавный прогресс в диапазоне от 20% до 75%
                current_percent = int(20 + (index / total_cols) * 55)
                
                # Каждые 5 колонок (или на последней) обновляем статус, чтобы не перегружать GUI
                if index % 5 == 0 or index == total_cols:
                    self._update_progress(
                        current_percent, 
                        f"Excel (Метки): Обработано колонок {index} из {total_cols}..."
                    )

            self._update_progress(80, "Excel (Метки): Запись матрицы в XLSX (это может занять время)...")
            df_all.to_excel(file_path, index=False)
            
            self.root.after(0, self._export_labels_finished_successfully, file_path)

        except Exception as e:
            self.root.after(0, lambda err=e: self._export_labels_failed(err))

    def _export_labels_finished_successfully(self, file_path):
        self.progress.configure(value=100)
        self.lbl_status.config(text="Файл с метками успешно сохранен!", fg="green")
        messagebox.showinfo("Успех", f"Все данные с метками успешно сохранены в файл:\n{os.path.basename(file_path)}")
        self._unlock_main_buttons()

    def _export_labels_failed(self, error):
        self.progress.configure(value=0)
        self.lbl_status.config(text="Ошибка экспорта меток", fg="red")
        messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить Excel файл с метками:\n{error}")
        self._unlock_main_buttons()

    def _export_finished_successfully(self):
        self.progress.configure(value=100)
        self.lbl_status.config(text="Экспорт успешно завершен!", fg="green")
        messagebox.showinfo("Успех", "Экспорт успешно завершен!")
        self._unlock_main_buttons()

    def _export_failed(self, error):
        self.progress.configure(value=0)
        self.lbl_status.config(text="Ошибка экспорта", fg="red")
        messagebox.showerror("Ошибка", f"Произошла ошибка при экспорте:\n{error}")
        self._unlock_main_buttons()

    def _unlock_main_buttons(self):
        self.btn_select.config(state=tk.NORMAL)
        self.btn_configure.config(state=tk.NORMAL)
        self.btn_configure_excel.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        if HAS_OPENPYXL:
            self.btn_export_excel.config(state=tk.NORMAL)
            if hasattr(self, 'btn_export_labels'):
                self.btn_export_labels.config(state=tk.NORMAL)
        else:
            self.btn_export_excel.config(state=tk.DISABLED)
            if hasattr(self, 'btn_export_labels'):
                self.btn_export_labels.config(state=tk.DISABLED)


if __name__ == "__main__":
    root_window = tk.Tk()
    app = SpssConverterApp(root_window)
    root_window.mainloop()
