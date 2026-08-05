# SAV2Convert — конвертеры файлов SPSS (.sav)

Два независимых инструмента для работы с выгрузками SPSS. Общего кода у них нет,
но зависимости и предметная область совпадают, поэтому они лежат рядом.

| инструмент | что делает | интерфейс | документация |
|---|---|---|---|
| `sav2adl.py` | метаданные `.sav` → структура переменных `.adl` и справочники `.smt` для mtools | командная строка и системный диалог выбора файла | [README_sav2adl.md](README_sav2adl.md) |
| `pySmartSPSSExporter.py` | `.sav` → фиксированный ASCII (`.dat` + `.lay`) и выгрузки в Excel | оконный, tkinter | [README_pySmartSPSSExporter.md](README_pySmartSPSSExporter.md) |

## Установка

```bash
pip install -r requirements.txt
```

Один файл зависимостей на оба инструмента: `pyreadstat` и `pandas` нужны обоим,
`openpyxl` — только экспортеру. `tkinter` берётся из стандартной библиотеки, но
в некоторых сборках Linux ставится отдельно (`python3-tk`).

## Запуск

```bash
python sav2adl.py                # откроется диалог выбора .sav
python sav2adl.py data.sav       # консольный запуск

python pySmartSPSSExporter.py    # оконное приложение
```

## Тесты

```bash
python tests/test_sav2adl.py
```

Тесты покрывают `sav2adl.py`: генерируется тестовый `.sav`, прогоняется
конвертация и проверяется формат выгрузки. Для `pySmartSPSSExporter.py` тестов
нет — логика переплетена с виджетами tkinter и в текущем виде не вызывается без
запуска окна.
