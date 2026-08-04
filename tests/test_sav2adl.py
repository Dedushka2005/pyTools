#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Смоук-тест sav2adl.py на сгенерированном .sav.

Запуск: python tests/test_sav2adl.py   (или pytest tests/test_sav2adl.py)
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import sav2adl  # noqa: E402


def build_outputs(tmpdir):
    sav = os.path.join(tmpdir, "sample.sav")
    subprocess.check_call([sys.executable, os.path.join(HERE, "make_sample_sav.py"), sav])
    sav2adl.convert(sav, out_encoding="utf-8", categorical={"QB3"}, quiet=True)

    def read(name):
        with open(os.path.join(tmpdir, name), encoding="utf-8") as fh:
            return fh.read()

    return read("sample.adl"), read, os.listdir(tmpdir)


def test_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        adl, read, files = build_outputs(tmpdir)

        # 1. Директива include .def первой строкой + пустая строка
        head = adl.splitlines()
        assert head[0] == "include sample.def", head[0]
        assert head[1] == "", head[:3]

        # 2. Фильтрация: пустая и текстовая переменные исключены,
        #    текстовая из одних цифр сохранена как числовая
        assert "where=QA2]" not in adl
        assert "where=QA3]" not in adl
        assert "where=QA4]" in adl

        # 3. Нормализация имён и схлопывание множественного выбора
        assert '"A6.1. Строка 1" where=QA6xr1' in adl      # QA6_5.1/QA6_6.1
        assert '"A6.2. Строка 2" where=QA6xr2' in adl      # QA6_5.2/QA6_6.2
        assert '"A12.8. Строка 8" where=QA12xr8' in adl    # QA12_r8_c1/_c2
        assert "where=QA6_5.2]" not in adl
        assert "where=QA12_r8_c1]" not in adl

        # 3.1 Одиночный грид: QA8_r1_c1 -> откат к QA8_r1, не мульти
        assert "where=QA8_r1]" in adl
        assert "QA8xr1" not in adl

        # 4. Заголовки
        assert '"A1. В среднем сколько лет пациенту?"' in adl   # индекс сохранён
        assert '"B7.1. Хумира (адалимумаб)"' in adl             # индекс грида, Q отсечена
        assert '"S5.1. Ревматоидный артрит" where=QS5x1' in adl  # защита одиночек

        # 5.1 Открытый числовой вопрос без меток
        assert '\t"18"=18 value=18' in adl
        assert '\t"25"=25 value=25' in adl

        # 5.2 Множественный выбор — чистый формат, без value= и без среднего
        multi = adl.split('"A12.8. Строка 8" where=QA12xr8]')[1].split("\n\n")[0]
        assert '"Интернет"=1' in multi
        assert "value=" not in multi and "Среднее" not in multi

        # 6.1 Общий include для гридов
        for row in ("QB7_r1", "QB7_r2", "QB7_r3"):
            assert 'where=%s]\n\tinclude QB7.smt' % row in adl
        assert "QB7.smt" in files

        # 6.2 Сортировка внутри .smt: шкала, спецкоды, среднее
        qb7 = [l.strip() for l in read("QB7.smt").splitlines() if l.strip()]
        assert qb7 == [
            '"Совсем не удовлетворён"=1',
            '"Скорее не удовлетворён"=2',
            '"Нейтрально"=3',
            '"Скорее удовлетворён"=4',
            '"Полностью удовлетворён"=5',
            '"Затрудняюсь ответить"=6',   # стоп-слово -> спецкод, уходит в хвост
            '"Отказ от ответа"=99',
            '"Среднее"=m',
        ], qb7

        # 6.3 Категориальный грид — чистый формат без среднего
        qb3 = [l.strip() for l in read("QB3.smt").splitlines() if l.strip()]
        assert qb3 == ['"Каждый день"=1', '"Раз в неделю"=2', '"Реже"=3'], qb3

        # 6.4 Дубликаты у одиночных переменных вынесены в общий .smt
        assert 'where=QD1]\n\tinclude QD1.smt' in adl
        assert 'where=QD2]\n\tinclude QD1.smt' in adl

        print("OK: все проверки пройдены")


def test_ansi_output():
    """Выгрузка в ANSI (cp1251) читается как cp1251."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sav = os.path.join(tmpdir, "sample.sav")
        subprocess.check_call([sys.executable, os.path.join(HERE, "make_sample_sav.py"), sav])
        sav2adl.convert(sav, categorical={"QB3"}, quiet=True)
        with open(os.path.join(tmpdir, "sample.adl"), encoding="cp1251") as fh:
            text = fh.read()
        assert "Хумира" in text
        print("OK: выгрузка в ANSI (cp1251)")


if __name__ == "__main__":
    test_all()
    test_ansi_output()
