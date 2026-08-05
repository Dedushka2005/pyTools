#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты детерминированной части cbc_hb.py — кодирования и генератора плана.

MCMC здесь не запускается: проверяется всё, что должно быть верным
безотносительно сэмплирования, поэтому тест отрабатывает за секунды и не
требует установленного PyMC.

Запуск: python tests/test_cbc_hb.py   (или pytest tests/test_cbc_hb.py)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cbc_hb import (  # noqa: E402
    CBCDesignGenerator,
    CBCHierarchicalBayesEstimator,
    EffectsCoder,
    _relative_d_efficiency,
    _sort_levels,
    interaction_cell_counts,
)

SPACE = {
    "Brand": ["Alpha", "Beta", "Gamma", "Delta"],
    "Price": [299, 399, 499, 599],
    "Storage": ["128GB", "256GB", "512GB"],
}
PAIR = ("Brand", "Price")


def make_generator(**kwargs):
    params = dict(
        source=SPACE,
        attribute_cols=list(SPACE),
        concepts_per_task=3,
        tasks_per_respondent=10,
        num_respondents=6,
        control_attributes="Brand",
        interactions=[PAIR],
        n_starts=3,
        random_state=42,
    )
    params.update(kwargs)
    return CBCDesignGenerator(**params)


def test_level_detection_and_ordering():
    # числовые уровни сортируются по значению, а не лексикографически
    assert _sort_levels([599, 299, 1099, 399]) == [299, 399, 599, 1099]
    assert _sort_levels(["599", "299", "1099"]) == ["299", "599", "1099"]
    assert _sort_levels(["Beta", "Alpha", "Gamma"]) == ["Alpha", "Beta", "Gamma"]

    # уровни распознаются автоматически из данных, без ручного словаря
    df = pd.DataFrame({"Brand": ["Beta", "Alpha", "Beta"], "Price": [499, 299, 399]})
    coder = EffectsCoder(["Brand", "Price"]).fit(df)
    assert coder.levels_["Brand"] == ["Alpha", "Beta"]
    assert coder.levels_["Price"] == [299, 399, 499]
    assert coder.reference_levels_ == {"Brand": "Beta", "Price": 499}
    print("ok: автораспознавание уровней и порядок сортировки")


def test_effects_coding_shape_and_values():
    coder = EffectsCoder(list(SPACE), interactions=[PAIR]).fit(SPACE)
    assert coder.n_main == 3 + 3 + 2
    assert coder.n_interaction == 3 * 3
    assert coder.n_params == coder.n_main + coder.n_interaction
    assert len(coder.coded_columns_) == coder.n_params
    assert len(coder.utility_columns_) == 4 + 4 + 3

    # уровни строк упорядочены лексикографически, поэтому референс Brand — Gamma
    assert coder.levels_["Brand"] == ["Alpha", "Beta", "Delta", "Gamma"]
    reference_brand = coder.reference_levels_["Brand"]
    assert reference_brand == "Gamma"

    df = pd.DataFrame(
        {
            "Brand": ["Alpha", reference_brand],
            "Price": [299, 599],
            "Storage": ["128GB", "512GB"],
        }
    )
    x = coder.transform(df)
    assert set(np.unique(x)).issubset({-1.0, 0.0, 1.0})
    # референсный уровень кодируется вектором из -1 по всем колонкам атрибута
    brand_slice = coder.coded_slices_["Brand"]
    assert np.array_equal(x[1, brand_slice], np.array([-1.0, -1.0, -1.0]))
    assert np.array_equal(x[0, brand_slice], np.array([1.0, 0.0, 0.0]))
    print("ok: эффект-кодирование даёт значения -1/0/1 и верную размерность")


def test_main_utilities_sum_to_zero():
    coder = EffectsCoder(list(SPACE), interactions=[PAIR]).fit(SPACE)
    rng = np.random.default_rng(0)
    beta = rng.normal(size=(5, coder.n_params))
    full = coder.expand_main(beta)
    assert full.shape == (5, len(coder.utility_columns_))
    for attr, sl in coder.utility_slices_.items():
        sums = full[:, sl].sum(axis=1)
        assert np.abs(sums).max() < 1e-10, f"{attr}: сумма полезностей не равна нулю"
    print("ok: сумма полезностей внутри атрибута строго равна нулю")


def test_interaction_table_is_double_centred():
    coder = EffectsCoder(list(SPACE), interactions=[PAIR]).fit(SPACE)
    rng = np.random.default_rng(1)
    beta = rng.normal(size=(4, coder.n_params))
    table = coder.expand_interaction(beta, PAIR)
    assert table.shape == (4, 4, 4)
    assert np.abs(table.sum(axis=-1)).max() < 1e-10, "суммы по строкам не нулевые"
    assert np.abs(table.sum(axis=-2)).max() < 1e-10, "суммы по столбцам не нулевые"

    # левый верхний блок полной таблицы совпадает с исходными коэффициентами
    gamma = beta[:, coder.interaction_slices_[PAIR]].reshape(4, 3, 3)
    assert np.allclose(table[:, :3, :3], gamma)

    # обращение порядка атрибутов даёт транспонированную таблицу
    assert np.allclose(coder.expand_interaction(beta, ("Price", "Brand")), table.swapaxes(-1, -2))
    print("ok: таблица взаимодействия имеет нулевые суммы по обеим осям")


def test_prohibitions_are_never_violated():
    generator = make_generator()
    design = generator.generate()
    assert generator.check_prohibitions() == {}

    # прямая проверка: внутри каждой задачи бренды уникальны
    grouped = design.groupby(["respondent_id", "task_id"])["Brand"]
    assert (grouped.nunique() == grouped.size()).all()
    print("ok: уровни контрольного атрибута не повторяются внутри задачи")


def test_infeasible_prohibition_raises():
    try:
        make_generator(concepts_per_task=5)  # у Brand всего 4 уровня
    except ValueError as exc:
        assert "невыполнимое ограничение" in str(exc)
        print("ok: невыполнимый запрет отвергается с внятным сообщением")
        return
    raise AssertionError("ожидалось ValueError для concepts_per_task > числа уровней Brand")


def test_level_balance_and_efficiency():
    generator = make_generator()
    generator.generate()
    balance = generator.balance_report()
    assert balance["deviation_pct"].abs().max() < 10.0, balance
    assert 0.0 < generator.min_d_efficiency_ <= generator.d_efficiency_ <= 1.0
    print(
        f"ok: баланс уровней в пределах {balance['deviation_pct'].abs().max():.1f}%, "
        f"D-эффективность {generator.d_efficiency_:.3f}"
    )


def test_d_efficiency_degenerate_design():
    coder = EffectsCoder(list(SPACE), interactions=[PAIR]).fit(SPACE)
    ideal = coder.ideal_information()
    # план из одинаковых профилей вырожден
    df = pd.DataFrame({"Brand": ["Alpha"] * 20, "Price": [299] * 20, "Storage": ["128GB"] * 20})
    assert _relative_d_efficiency(coder.transform(df), ideal) == 0.0
    print("ok: вырожденный план получает нулевую D-эффективность")


def test_sparsity_report_counts_empty_cells():
    # у респондента 2 нет сочетания Alpha x 299
    design = pd.DataFrame(
        {
            "respondent_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "task_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "concept_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "Brand": ["Alpha", "Beta", "Alpha", "Beta", "Beta", "Beta", "Beta", "Beta"],
            "Price": [299, 399, 399, 299, 299, 399, 299, 399],
        }
    )
    report = interaction_cell_counts(design, [("Brand", "Price")])
    row = report.iloc[0]
    assert row["n_cells"] == 4
    assert row["min_cell_count"] == 0
    assert row["min_cells_filled"] == 2
    assert row["respondents_with_empty_cells"] == 1
    print("ok: отчёт о разреженности видит непоказанные ячейки")


def test_response_validation():
    generator = make_generator(num_respondents=2, tasks_per_respondent=4)
    design = generator.generate()
    estimator = CBCHierarchicalBayesEstimator(list(SPACE), interactions=[PAIR])
    estimator.coder.fit(design)

    responses = design[["respondent_id", "task_id", "concept_id"]].copy()
    responses["chosen"] = (responses["concept_id"] == 1).astype(int)
    x, y, respondents, _ = estimator._prepare(design, responses)
    assert x.shape == (2, 4, 3, estimator.coder.n_params)
    assert y.shape == (2, 4) and (y == 0).all()
    assert list(respondents) == [1, 2]

    # два выбора в одной задаче — режим single_choice нарушен
    broken = responses.copy()
    broken.loc[1, "chosen"] = 1
    try:
        estimator._prepare(design, broken)
    except ValueError as exc:
        assert "single_choice" in str(exc)
        print("ok: нарушение режима single_choice отвергается")
        return
    raise AssertionError("ожидалось ValueError при двух выборах в одной задаче")


def test_unbalanced_tasks_rejected():
    generator = make_generator(num_respondents=2, tasks_per_respondent=4)
    design = generator.generate()
    # у респондента 2 отрезана одна задача
    trimmed = design[~((design["respondent_id"] == 2) & (design["task_id"] == 4))]
    responses = trimmed[["respondent_id", "task_id", "concept_id"]].copy()
    responses["chosen"] = (responses["concept_id"] == 1).astype(int)

    estimator = CBCHierarchicalBayesEstimator(list(SPACE), interactions=[PAIR])
    estimator.coder.fit(trimmed)
    try:
        estimator._prepare(trimmed, responses)
    except ValueError as exc:
        assert "одинакового числа задач" in str(exc)
        print("ok: несбалансированный план отвергается с внятным сообщением")
        return
    raise AssertionError("ожидалось ValueError для разного числа задач у респондентов")


def test_unknown_level_rejected():
    coder = EffectsCoder(["Brand"]).fit(SPACE)
    try:
        coder.transform(pd.DataFrame({"Brand": ["Omega"]}))
    except ValueError as exc:
        assert "неизвестные уровни" in str(exc)
        print("ok: неизвестный уровень атрибута отвергается")
        return
    raise AssertionError("ожидалось ValueError для уровня, отсутствующего в обучении")


def test_all():
    test_level_detection_and_ordering()
    test_effects_coding_shape_and_values()
    test_main_utilities_sum_to_zero()
    test_interaction_table_is_double_centred()
    test_prohibitions_are_never_violated()
    test_infeasible_prohibition_raises()
    test_level_balance_and_efficiency()
    test_d_efficiency_degenerate_design()
    test_sparsity_report_counts_empty_cells()
    test_response_validation()
    test_unbalanced_tasks_rejected()
    test_unknown_level_rejected()


if __name__ == "__main__":
    test_all()
    print("\nвсе проверки пройдены")
