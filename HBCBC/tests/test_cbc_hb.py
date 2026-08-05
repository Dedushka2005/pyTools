#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты детерминированной части cbc_hb.py — кодирования и генератора плана.

MCMC здесь не запускается: проверяется всё, что должно быть верным
безотносительно сэмплирования, поэтому тест отрабатывает за секунды.
Единственная проверка, которой нужен PyMC, — сверка двух ветвей правдоподобия;
без установленного PyMC она пропускается, остальные работают.

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
    simulate_choices,
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
    x, response, respondents, _ = estimator._prepare(design, responses)
    assert x.shape == (2, 4, 3, estimator.coder.n_params)
    assert response.shape == (2, 4, 3)
    assert (response[..., 0] == 1).all() and (response[..., 1:] == 0).all()
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


def test_duplicate_keys_rejected():
    generator = make_generator(num_respondents=2, tasks_per_respondent=4)
    design = generator.generate()
    estimator = CBCHierarchicalBayesEstimator(list(SPACE), interactions=[PAIR])
    estimator.coder.fit(design)

    responses = design[["respondent_id", "task_id", "concept_id"]].copy()
    responses["response"] = (responses["concept_id"] == 1).astype(int)

    for label, frames in (
        ("responses_df", (design, pd.concat([responses, responses.iloc[[0]]], ignore_index=True))),
        ("design_df", (pd.concat([design, design.iloc[[0]]], ignore_index=True), responses)),
    ):
        try:
            estimator._prepare(*frames)
        except ValueError as exc:
            assert label in str(exc)
            assert "уникальной" in str(exc)
        else:
            raise AssertionError(f"ожидалось ValueError при дубле ключа в {label}")
    print("ok: повтор тройки идентификаторов отвергается в обеих таблицах")


def test_response_column_is_one_schema():
    # response / chosen / allocation — синонимы одной колонки, а не три формата
    frame = pd.DataFrame({"respondent_id": [1], "task_id": [1], "concept_id": [1]})
    for name in ("response", "chosen", "allocation"):
        candidate = frame.copy()
        candidate[name] = 1
        assert CBCHierarchicalBayesEstimator.resolve_response_column(candidate) == name
    try:
        CBCHierarchicalBayesEstimator.resolve_response_column(frame)
    except KeyError as exc:
        assert "нет колонки с ответом" in str(exc)
        print("ok: колонка ответа распознаётся по единой схеме")
        return
    raise AssertionError("ожидался KeyError при отсутствии колонки ответа")


def test_allocation_validation():
    generator = make_generator(num_respondents=2, tasks_per_respondent=4)
    design = generator.generate()
    estimator = CBCHierarchicalBayesEstimator(
        list(SPACE), interactions=[PAIR], response_mode="allocation"
    )
    estimator.coder.fit(design)

    responses = design[["respondent_id", "task_id", "concept_id"]].copy()
    responses["allocation"] = [50.0, 30.0, 20.0] * (len(design) // 3)
    x, response, _, _ = estimator._prepare(design, responses)
    assert response.shape == (2, 4, 3)
    assert np.allclose(response.sum(axis=-1), 100.0)
    assert estimator.allocation_total == 100.0

    # сумма не равна заявленной — режим allocation нарушен
    broken = responses.copy()
    broken.loc[0, "allocation"] = 10.0
    estimator.allocation_total = 100.0
    try:
        estimator._prepare(design, broken)
    except ValueError as exc:
        assert "сумма по задаче" in str(exc)
        print("ok: нарушение суммы баллов в режиме allocation отвергается")
        return
    raise AssertionError("ожидалось ValueError при неверной сумме баллов")


def test_none_concept_coding():
    coder = EffectsCoder(list(SPACE), interactions=[PAIR], include_none=True).fit(SPACE)
    # колонка None стоит между главными эффектами и взаимодействиями
    assert coder.coded_columns_[coder.n_main] == "None"
    assert coder.n_structural == coder.n_main + 1
    assert coder.none_index_ == coder.n_main
    assert coder.n_params == coder.n_main + 1 + coder.n_interaction

    df = pd.DataFrame(
        {
            "Brand": ["Alpha", None],
            "Price": [299, None],
            "Storage": ["128GB", None],
            "is_none": [0, 1],
        }
    )
    x = coder.transform(df)
    # у None-концепта обнулены и главные эффекты, и взаимодействия
    assert x[1, coder.n_main] == 1.0
    assert not x[1, : coder.n_main].any()
    assert not x[1, coder.n_structural :].any()
    # у обычного профиля колонка None равна нулю
    assert x[0, coder.n_main] == 0.0
    assert x[0, : coder.n_main].any()

    # колонка None исключена из расчёта D-эффективности
    assert coder.n_main not in coder.design_column_indices_.tolist()
    assert len(coder.design_column_indices_) == coder.n_main + coder.n_interaction
    print("ok: None-концепт кодируется отдельной константой альтернативы")


def test_none_concept_design():
    generator = make_generator(include_none=True, num_respondents=3, tasks_per_respondent=6)
    design = generator.generate()
    per_task = design.groupby(["respondent_id", "task_id"]).size()
    assert (per_task == 4).all(), "3 реальных профиля плюс None"
    assert (design.groupby(["respondent_id", "task_id"])["is_none"].sum() == 1).all()

    # None не ломает ни запреты, ни баланс уровней
    assert generator.check_prohibitions() == {}
    balance = generator.balance_report()
    assert balance["count"].sum() == (design["is_none"] == 0).sum() * len(SPACE)
    assert 0.0 < generator.d_efficiency_ <= 1.0
    print("ok: None-концепт добавлен в каждую задачу и исключён из отчётов")


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


def test_allocation_likelihood_matches_categorical():
    """При one-hot ответах и весе 1 обе ветки правдоподобия обязаны совпасть.

    Режим allocation с долями 0/1 — это в точности принудительный выбор, так что
    взвешенное логарифмическое правдоподобие должно дать ровно то же значение,
    что и pm.Categorical. Проверка связывает новую ветку с проверенной.
    """
    try:
        import pymc  # noqa: F401
    except ImportError:
        print("пропуск: PyMC не установлен — проверка правдоподобия не выполняется")
        return

    generator = make_generator(num_respondents=4, tasks_per_respondent=6, interactions=None)
    design = generator.generate()
    rng = np.random.default_rng(11)
    beta_true = rng.normal(size=(4, generator.coder.n_params)) * 0.8
    responses = simulate_choices(design, generator.coder, beta_true, rng)

    def build(mode):
        estimator = CBCHierarchicalBayesEstimator(
            list(SPACE), response_mode=mode, allocation_weight=1.0, chains=1
        )
        estimator.coder.fit(design)
        x, response, ids, _ = estimator._prepare(design, responses)
        estimator.respondent_ids_ = ids
        estimator.cov_mode_ = estimator._resolve_cov_mode()
        return estimator._build_model(x, response, ids)

    choice_model, allocation_model = build("single_choice"), build("allocation")
    choice_logp = choice_model.compile_logp()
    allocation_logp = allocation_model.compile_logp()

    for seed in range(4):
        point = choice_model.initial_point(random_seed=seed)
        assert np.isclose(choice_logp(point), allocation_logp(point), atol=1e-8)
    print("ok: правдоподобие allocation совпадает с категориальным на one-hot ответах")


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
    test_duplicate_keys_rejected()
    test_response_column_is_one_schema()
    test_allocation_validation()
    test_none_concept_coding()
    test_none_concept_design()
    test_allocation_likelihood_matches_categorical()
    test_unbalanced_tasks_rejected()
    test_unknown_level_rejected()


if __name__ == "__main__":
    test_all()
    print("\nвсе проверки пройдены")
