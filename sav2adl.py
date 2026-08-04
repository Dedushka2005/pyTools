#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sav2adl.py — конвертер метаданных SPSS (.sav) в структуру переменных mtools.

Скрипт читает .sav файл (имена переменных, типы, метки переменных и метки
значений + сам массив данных), нормализует комплексные имена переменных,
схлопывает множественный выбор и выгружает:

  * <имя>.adl  — основной файл структуры данных;
  * <база>.smt — справочники блоков ответов (общие для гридов и для
                 одиночных вопросов с совпадающими наборами ответов).

Запуск:
    python sav2adl.py                # откроется диалог выбора файла (tkinter)
    python sav2adl.py data.sav       # без диалога
    python sav2adl.py data.sav --utf8
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pyreadstat
except ImportError:  # pragma: no cover - зависит от окружения
    sys.exit("Требуется пакет pyreadstat. Установите его: pip install pyreadstat")


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

#: Базы грид-вопросов, которые нужно выгружать как категориальные
#: (чистый формат "Метка"=код, без разделения на шкалу/спецкоды и без среднего).
#: Дополняется из командной строки ключом --categorical.
CATEGORICAL_GRID_BASES = {
    # "QB3",
    # "QC1",
}

#: Коды, начиная с которых альтернатива считается спецкодом.
SPECIAL_CODE_MIN = 90

#: Стоп-слова в тексте метки, по которым альтернатива считается спецкодом.
SPECIAL_LABEL_WORDS = (
    "затрудняюсь",
    "отказ",
    "трудно",
    "сложно",
    "не знаю",
    "other",
    "другой",
)

#: Строка расчёта среднего в mtools.
MEAN_LINE = '"Среднее"=m'

#: Отступ строк блока ответов.
INDENT = "\t"

#: Кодировки, которые перебираются при чтении .sav.
FALLBACK_ENCODINGS = ("cp1251", "utf-8", "cp1252", "koi8-r")

#: Кодировка выгрузки по умолчанию (ANSI).
ANSI_ENCODING = "cp1251"


# ---------------------------------------------------------------------------
# Регулярные выражения имён
# ---------------------------------------------------------------------------

# QA6_5.2  ->  QA6xr2_5   (5 — подпункт, 2 — строка грида)
RE_DOT = re.compile(r"^(?P<base>.+?)_(?P<sub>\d+)\.(?P<row>\d+)$")

# QA12_r8_c2 -> QA12xr8_2 (r8 — строка, c2 — колонка)
RE_RC = re.compile(r"^(?P<base>.+?)_r(?P<row>\d+)_c(?P<col>\d+)$", re.IGNORECASE)

# Хвостовой числовой суффикс: QA6xr2_5 -> база QA6xr2
RE_MULTI = re.compile(r"^(?P<base>.+?)_(?P<idx>\d+)$")

# Грид-строки в финальном (нормализованном) виде
RE_GRID_R = re.compile(r"^(?P<base>.+?)_r(?P<row>\d+)$", re.IGNORECASE)
RE_GRID_XR = re.compile(r"^(?P<base>.+?)xr(?P<row>\d+)$", re.IGNORECASE)

# Строка, целиком состоящая из числа ("00123", "12,5", "-7")
RE_NUMERIC_TEXT = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

# Разделитель сегментов внутри метки SPSS
RE_SEGMENT = re.compile(r"\s+[-–—]\s+")


# ---------------------------------------------------------------------------
# Мелкие утилиты
# ---------------------------------------------------------------------------


def is_missing(value) -> bool:
    """Системный пропуск или пустая строка."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def fmt_num(value) -> str:
    """Число в виде, пригодном для .adl: целые без дробной части."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return ("%g" % value)
    return str(value)


def to_code(value):
    """Код альтернативы: int, если значение целое."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return int(f) if f.is_integer() else f


def esc_label(text: str) -> str:
    """Метка внутри кавычек: убираем кавычки и лишние пробелы."""
    text = str(text).replace('"', "'")
    return re.sub(r"\s+", " ", text).strip()


def name_index(name: str) -> str:
    """Имя переменной -> человеческий индекс вопроса.

    QS5x1 -> S5.1, QB7_r1 -> B7.1, QA6xr2 -> A6.2, QA1 -> A1
    """
    s = re.sub(r"^[Qq]", "", name)
    s = re.sub(r"xr(\d+)", r".\1", s, flags=re.IGNORECASE)
    s = re.sub(r"_r(\d+)", r".\1", s, flags=re.IGNORECASE)
    s = re.sub(r"x(\d+)", r".\1", s, flags=re.IGNORECASE)
    s = re.sub(r"_(\d+)", r".\1", s)
    return s


def strip_q(name: str) -> str:
    """Отсекает техническую букву Q в начале базы вопроса."""
    return re.sub(r"^[Qq]", "", name)


def is_special(code, label: str) -> bool:
    """Спецкод: код >= 90 либо стоп-слово в тексте метки."""
    try:
        if float(code) >= SPECIAL_CODE_MIN:
            return True
    except (TypeError, ValueError):
        pass
    low = str(label).lower()
    return any(word in low for word in SPECIAL_LABEL_WORDS)


def split_scale(labels: Dict) -> Tuple[List, List]:
    """Делит альтернативы на «чистую шкалу» и «спецкоды».

    Чистая шкала — последовательные коды 1..K без стоп-слов; всё, что рвёт
    последовательность или помечено как спецкод, уходит в хвост.
    """
    plain = sorted(
        (c for c in labels if not is_special(c, labels[c])),
        key=lambda c: float(c),
    )
    scale: List = []
    expected = 1
    for code in plain:
        if float(code) == float(expected):
            scale.append(code)
            expected += 1
        else:
            break
    rest = [c for c in labels if c not in scale]
    rest.sort(key=lambda c: float(c))
    return scale, rest


# ---------------------------------------------------------------------------
# Модель данных
# ---------------------------------------------------------------------------


@dataclass
class VarInfo:
    """Переменная SPSS, прошедшая предварительную фильтрацию."""

    name: str                       # исходное имя в .sav
    label: str                      # исходная метка переменной
    value_labels: Dict              # {код: метка}
    values: List                    # непустые числовые значения из файла
    was_string: bool = False        # текстовая переменная, признанная числовой
    norm: str = ""                  # нормализованное имя
    is_complex: bool = False        # имя было комплексным (_N.M или _rN_cM)
    cbase: str = ""                 # база комплексного имени
    crow: int = 0                   # номер строки грида


@dataclass
class Entry:
    """Запись итогового .adl (одиночный вопрос, мульти или строка грида)."""

    name: str
    members: List[VarInfo]
    title: str = ""
    grid_base: str = ""
    grid_row: int = 0
    is_multi: bool = False
    mode: str = "clean"             # clean | scale | open
    labels: Dict = field(default_factory=dict)
    values: List = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    include: str = ""               # имя .smt файла вместо блока ответов


# ---------------------------------------------------------------------------
# Чтение .sav
# ---------------------------------------------------------------------------


def read_sav(path: str, encoding: Optional[str], user_missing: bool):
    """Читает .sav, перебирая кодировки, если файл не открывается напрямую."""
    attempts: List[Optional[str]] = [encoding] if encoding else [None]
    if not encoding:
        attempts.extend(FALLBACK_ENCODINGS)

    last_error: Optional[Exception] = None
    for enc in attempts:
        kwargs = {"user_missing": user_missing}
        if enc:
            kwargs["encoding"] = enc
        try:
            return pyreadstat.read_sav(path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - нужен перебор кодировок
            last_error = exc
    raise RuntimeError(f"Не удалось прочитать {path}: {last_error}")


def collect_variables(df, meta) -> Tuple[List[VarInfo], int, int]:
    """Предварительная фильтрация переменных.

    Отбрасывает полностью пустые переменные и текстовые переменные, кроме тех,
    что заполнены исключительно числами (они трактуются как числовые).
    """
    variables: List[VarInfo] = []
    dropped_empty = 0
    dropped_string = 0

    types = getattr(meta, "readstat_variable_types", {}) or {}
    labels_by_name = dict(zip(meta.column_names, meta.column_labels or []))

    for name in meta.column_names:
        column = df[name].tolist() if name in df else []
        is_string = types.get(name) == "string"

        raw = [v for v in column if not is_missing(v)]
        if not raw:
            dropped_empty += 1
            continue

        was_string = False
        if is_string:
            texts = [str(v).strip() for v in raw]
            if not all(RE_NUMERIC_TEXT.match(t) for t in texts):
                dropped_string += 1
                continue
            values = [float(t.replace(",", ".")) for t in texts]
            was_string = True
        else:
            values = []
            for v in raw:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
            if not values:
                dropped_empty += 1
                continue

        value_labels = {
            to_code(code): str(text)
            for code, text in (meta.variable_value_labels.get(name, {}) or {}).items()
        }

        variables.append(
            VarInfo(
                name=name,
                label=str(labels_by_name.get(name) or ""),
                value_labels=value_labels,
                values=values,
                was_string=was_string,
            )
        )

    return variables, dropped_empty, dropped_string


# ---------------------------------------------------------------------------
# Нормализация имён и группировка
# ---------------------------------------------------------------------------


def normalize_name(var: VarInfo) -> None:
    """Приводит комплексные имена к единому виду BASExrROW_INDEX."""
    m = RE_RC.match(var.name)
    if m:
        var.cbase = m.group("base")
        var.crow = int(m.group("row"))
        var.norm = "%sxr%s_%s" % (var.cbase, m.group("row"), int(m.group("col")))
        var.is_complex = True
        return

    m = RE_DOT.match(var.name)
    if m:
        var.cbase = m.group("base")
        var.crow = int(m.group("row"))
        var.norm = "%sxr%s_%s" % (var.cbase, m.group("row"), int(m.group("sub")))
        var.is_complex = True
        return

    var.norm = var.name


def group_entries(variables: Sequence[VarInfo]) -> List[Entry]:
    """Схлопывает множественный выбор и откатывает одиночные гриды."""
    groups: "OrderedDict[Tuple[str, str], List[VarInfo]]" = OrderedDict()
    for var in variables:
        m = RE_MULTI.match(var.norm)
        key = ("multi", m.group("base")) if m else ("single", var.norm)
        groups.setdefault(key, []).append(var)

    entries: List[Entry] = []
    for (kind, key), members in groups.items():
        if kind == "multi" and len(members) >= 2:
            entries.append(Entry(name=key, members=members, is_multi=True))
            continue

        var = members[0]
        name = var.norm
        if var.is_complex:
            # Комплексное имя без пары — не мульти, откатываем к BASE_rN.
            name = "%s_r%d" % (var.cbase, var.crow)
        entry = Entry(name=name, members=members)
        if len(members) > 1:
            entry.members = members
        entries.append(entry)

    for entry in entries:
        m = RE_GRID_R.match(entry.name) or RE_GRID_XR.match(entry.name)
        if m:
            entry.grid_base = m.group("base")
            entry.grid_row = int(m.group("row"))

    return entries


# ---------------------------------------------------------------------------
# Заголовки
# ---------------------------------------------------------------------------


def strip_tech_prefixes(label: str, names: Sequence[str]) -> Tuple[str, bool]:
    """Убирает технические префиксы «Имя - » и «[Имя] - » из метки.

    Возвращает очищенную метку и признак, что префикс действительно был.
    """
    ordered = sorted({n for n in names if n}, key=len, reverse=True)
    found = False
    changed = True
    while changed:
        changed = False
        for name in ordered:
            esc = re.escape(name)
            for pattern in (
                r"^\[\s*%s\s*\]\s*[-–—:]\s*" % esc,
                r"^%s\s*[-–—:]\s*" % esc,
            ):
                m = re.match(pattern, label, flags=re.IGNORECASE)
                if m:
                    label = label[m.end():]
                    changed = True
                    found = True
                    break
            if changed:
                break
    return label.strip(), found


def last_segment(label: str) -> str:
    """Последний смысловой сегмент метки (текст строки грида)."""
    parts = [p.strip() for p in RE_SEGMENT.split(label) if p.strip()]
    return parts[-1] if parts else label.strip()


def common_label(labels: Sequence[str]) -> str:
    """Общая (без хвоста-варианта) часть меток переменных мульти-блока."""
    labels = [l for l in labels if l]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]

    split = [[p.strip() for p in RE_SEGMENT.split(l) if p.strip()] for l in labels]
    common: List[str] = []
    for parts in zip(*split):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    if common:
        return " - ".join(common)
    first = split[0]
    return " - ".join(first[:-1]) if len(first) > 1 else labels[0]


def build_title(entry: Entry) -> str:
    """Формирует заголовок записи по правилам mtools."""
    strip_names = [entry.name]
    for var in entry.members:
        strip_names.extend([var.name, var.norm, var.cbase])
    if entry.grid_base:
        strip_names.append(entry.grid_base)
        strip_names.append(strip_q(entry.grid_base))
    strip_names.append(RE_MULTI.sub(r"\g<base>", entry.name))

    cleaned: List[str] = []
    had_prefix = False
    for var in entry.members:
        text, found = strip_tech_prefixes(var.label, strip_names)
        had_prefix = had_prefix or found
        cleaned.append(text)

    label = common_label(cleaned)
    index = name_index(entry.name)

    if entry.grid_base:
        # Для гридов индекс строки генерируется всегда: B7.1. Хумира
        text = last_segment(label)
        if not text:
            text = label
        if text.startswith(index):
            return text
        return "%s. %s" % (index, text) if text else index

    if not label:
        return index

    # Защита одиночек: если метка начиналась с собственного имени переменной,
    # индекс принудительно возвращается обратно в человеческом виде.
    if had_prefix and not re.match(r"^%s[\.\s]" % re.escape(index), label):
        return "%s. %s" % (index, label)
    return label


# ---------------------------------------------------------------------------
# Блоки ответов
# ---------------------------------------------------------------------------


def clean_lines(labels: Dict) -> List[str]:
    """Чистый формат: "Метка"=код по возрастанию кода."""
    return [
        '%s"%s"=%s' % (INDENT, esc_label(labels[code]), fmt_num(code))
        for code in sorted(labels, key=lambda c: float(c))
    ]


def scale_lines(labels: Dict) -> List[str]:
    """Шкала: чистая шкала, затем спецкоды, затем среднее."""
    scale, rest = split_scale(labels)
    lines = [
        '%s"%s"=%s' % (INDENT, esc_label(labels[code]), fmt_num(code))
        for code in list(scale) + list(rest)
    ]
    lines.append(INDENT + MEAN_LINE)
    return lines


def open_lines(values: Sequence) -> List[str]:
    """Открытый числовой вопрос: "18"=18 value=18 + среднее."""
    uniq = sorted({to_code(v) for v in values}, key=lambda c: float(c))
    lines = [
        '%s"%s"=%s value=%s' % (INDENT, fmt_num(v), fmt_num(v), fmt_num(v))
        for v in uniq
    ]
    lines.append(INDENT + MEAN_LINE)
    return lines


def build_lines(mode: str, labels: Dict, values: Sequence) -> List[str]:
    if mode == "open":
        return open_lines(values)
    if mode == "scale":
        return scale_lines(labels)
    return clean_lines(labels)


def merged_labels(entry: Entry) -> Dict:
    labels: Dict = {}
    for var in entry.members:
        labels.update(var.value_labels)
    return labels


def merged_values(entry: Entry) -> List:
    values: List = []
    for var in entry.members:
        values.extend(var.values)
    return values


def entry_mode(entry: Entry, categorical: set) -> str:
    if not entry.labels:
        return "open"
    if entry.grid_base:
        if entry.is_multi or entry.grid_base in categorical:
            return "clean"
        return "scale"
    return "clean"


def grid_mode(entries: Sequence[Entry], categorical: set) -> str:
    if all(e.mode == "open" for e in entries):
        return "open"
    if any(e.is_multi for e in entries):
        return "clean"
    if entries[0].grid_base in categorical:
        return "clean"
    return "scale"


# ---------------------------------------------------------------------------
# Сборка выгрузки
# ---------------------------------------------------------------------------


def build_output(entries: List[Entry], categorical: set) -> "OrderedDict[str, List[str]]":
    """Заполняет блоки ответов и формирует содержимое .smt файлов."""
    smt_files: "OrderedDict[str, List[str]]" = OrderedDict()

    for entry in entries:
        entry.labels = merged_labels(entry)
        entry.values = merged_values(entry)
        entry.mode = entry_mode(entry, categorical)
        entry.title = build_title(entry)
        entry.lines = build_lines(entry.mode, entry.labels, entry.values)

    # --- общий include для гридов -----------------------------------------
    grids: "OrderedDict[str, List[Entry]]" = OrderedDict()
    for entry in entries:
        if entry.grid_base:
            grids.setdefault(entry.grid_base, []).append(entry)

    for base, rows in grids.items():
        if len(rows) < 2:
            continue
        mode = grid_mode(rows, categorical)
        labels: Dict = {}
        values: List = []
        for row in rows:
            labels.update(row.labels)
            values.extend(row.values)
        smt_name = "%s.smt" % base
        smt_files[smt_name] = build_lines(mode, labels, values)
        for row in rows:
            row.include = smt_name

    # --- одинаковые блоки у одиночных переменных ---------------------------
    buckets: "OrderedDict[Tuple[str, ...], List[Entry]]" = OrderedDict()
    for entry in entries:
        if entry.include or not entry.lines:
            continue
        buckets.setdefault(tuple(entry.lines), []).append(entry)

    for lines, group in buckets.items():
        if len(group) < 2:
            continue
        smt_name = "%s.smt" % group[0].name
        suffix = 2
        while smt_name in smt_files:
            smt_name = "%s_%d.smt" % (group[0].name, suffix)
            suffix += 1
        smt_files[smt_name] = list(lines)
        for entry in group:
            entry.include = smt_name

    return smt_files


def render_adl(entries: Sequence[Entry], def_name: str) -> List[str]:
    out: List[str] = ["include %s.def" % def_name, ""]
    for entry in entries:
        out.append('["%s" where=%s]' % (esc_label(entry.title), entry.name))
        if entry.include:
            out.append("%sinclude %s" % (INDENT, entry.include))
        else:
            out.extend(entry.lines)
        out.append("")
    return out


def write_text(path: str, lines: Sequence[str], encoding: str) -> None:
    with open(path, "w", encoding=encoding, errors="replace", newline="\r\n") as fh:
        fh.write("\n".join(lines))
        if lines and lines[-1] != "":
            fh.write("\n")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def ask_for_file() -> str:
    """Системный диалог выбора .sav (только при запуске без аргументов)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # noqa: BLE001 - окружение без tkinter
        sys.exit(
            "Не удалось открыть диалог выбора файла (%s).\n"
            "Укажите путь к .sav в командной строке: python sav2adl.py data.sav" % exc
        )

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Выберите файл данных SPSS",
        filetypes=[("SPSS data files", "*.sav"), ("Все файлы", "*.*")],
    )
    root.destroy()
    if not path:
        sys.exit("Файл не выбран.")
    return path


def convert(
    sav_path: str,
    out_encoding: str = ANSI_ENCODING,
    in_encoding: Optional[str] = None,
    categorical: Optional[set] = None,
    user_missing: bool = False,
    quiet: bool = False,
) -> str:
    categorical = set(CATEGORICAL_GRID_BASES) | set(categorical or ())

    df, meta = read_sav(sav_path, in_encoding, user_missing)
    variables, dropped_empty, dropped_string = collect_variables(df, meta)
    if not variables:
        sys.exit("В файле не осталось переменных после фильтрации.")

    for var in variables:
        normalize_name(var)

    entries = group_entries(variables)
    smt_files = build_output(entries, categorical)

    folder = os.path.dirname(os.path.abspath(sav_path))
    stem = os.path.splitext(os.path.basename(sav_path))[0]
    adl_path = os.path.join(folder, stem + ".adl")

    write_text(adl_path, render_adl(entries, stem), out_encoding)
    for smt_name, lines in smt_files.items():
        write_text(os.path.join(folder, smt_name), lines, out_encoding)

    if not quiet:
        print("Прочитано переменных: %d" % len(meta.column_names))
        print("  исключено пустых:   %d" % dropped_empty)
        print("  исключено строковых:%d" % dropped_string)
        print("Записей в структуре:  %d" % len(entries))
        print("Файл структуры:       %s" % adl_path)
        for smt_name in smt_files:
            print("  справочник:         %s" % os.path.join(folder, smt_name))

    return adl_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Конвертер метаданных SPSS (.sav) в структуру переменных mtools (.adl/.smt)."
    )
    parser.add_argument("sav", nargs="?", help="путь к .sav (без него откроется диалог выбора)")
    parser.add_argument(
        "--utf8",
        action="store_true",
        help="выгружать .adl/.smt в UTF-8 (по умолчанию ANSI/Windows-1251)",
    )
    parser.add_argument(
        "--encoding",
        help="принудительная кодировка чтения .sav (по умолчанию автоопределение)",
    )
    parser.add_argument(
        "--categorical",
        default="",
        help="базы категориальных гридов через запятую (в дополнение к CATEGORICAL_GRID_BASES)",
    )
    parser.add_argument(
        "--user-missing",
        action="store_true",
        help="читать пользовательские пропуски как значения, а не как NaN",
    )
    parser.add_argument("--quiet", action="store_true", help="не печатать отчёт")
    args = parser.parse_args(argv)

    sav_path = args.sav or ask_for_file()
    if not os.path.isfile(sav_path):
        sys.exit("Файл не найден: %s" % sav_path)

    categorical = {p.strip() for p in args.categorical.split(",") if p.strip()}
    convert(
        sav_path,
        out_encoding="utf-8" if args.utf8 else ANSI_ENCODING,
        in_encoding=args.encoding,
        categorical=categorical,
        user_missing=args.user_missing,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
