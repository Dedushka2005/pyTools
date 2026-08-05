#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CBC/HB — Choice-Based Conjoint: генерация плана опроса и оценка индивидуальных
полезностей методом иерархического байесовского анализа (Hierarchical Bayes).

Модуль состоит из трёх независимых частей:

* :class:`EffectsCoder` — автоматическое распознавание структуры атрибутов и
  эффект-кодирование (-1 / 0 / 1), включая взаимодействия первого порядка.
* :class:`CBCDesignGenerator` — сбалансированный рандомизированный план CBC с
  запретом на дублирование уровней контрольных атрибутов внутри задачи.
* :class:`CBCHierarchicalBayesEstimator` — иерархическая байесовская MNL-модель
  на PyMC (NUTS), расчёт индивидуальных полезностей и важности атрибутов.

Соответствие Sawtooth Software CBC/HB
-------------------------------------
Реализация повторяет Sawtooth по спецификации модели и по постобработке:
эффект-кодирование, нулевая сумма полезностей внутри атрибута, важность через
размах полезностей, нормализация zero-centered diffs, метрика RLH.

Байесовский движок отличается сознательно: Sawtooth использует
MH-within-Gibbs с приором inverse-Wishart на ковариацию верхнего уровня, здесь
же применяется NUTS с приором LKJ на фактор Холецкого. LKJ гарантирует
положительную определённость и разделяет дисперсии и корреляции, что заметно
устойчивее численно. Следствие: числа не обязаны совпадать с выгрузкой
Sawtooth побитово, хотя содержательно эквивалентны.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "EffectsCoder",
    "CBCDesignGenerator",
    "CBCHierarchicalBayesEstimator",
]

ID_COLUMNS = ("respondent_id", "task_id", "concept_id")

#: Колонка-признак None-концепта («не выбрал бы ничего») в плане опроса.
NONE_COLUMN = "is_none"
#: Имя индивидуального параметра None-концепта.
NONE_PARAM = "None"

#: Колонки, в которых может лежать ответ респондента, в порядке приоритета.
RESPONSE_COLUMNS = ("response", "chosen", "allocation")


def _sort_levels(values: Iterable[Any]) -> list[Any]:
    """Детерминированный порядок уровней.

    Числовые уровни (в том числе записанные строками) сортируются по значению,
    остальные — лексикографически. Порядок важен: последний уровень становится
    референсным, а от него зависят имена колонок кодирования.
    """
    vals = list(values)
    try:
        return sorted(vals, key=lambda v: (float(v), str(v)))
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def _effects_basis(n_levels: int) -> np.ndarray:
    """Матрица эффект-кодирования (n_levels, n_levels - 1).

    Строка i для i < K-1 — единичный вектор, последняя строка — вектор из -1.
    Столбцы такой матрицы в сумме по строкам дают ноль, поэтому полезности
    уровней внутри атрибута автоматически центрированы вокруг нуля.
    """
    basis = np.zeros((n_levels, n_levels - 1), dtype=float)
    basis[: n_levels - 1, :] = np.eye(n_levels - 1)
    basis[n_levels - 1, :] = -1.0
    return basis


class EffectsCoder:
    """Автораспознавание уровней атрибутов и эффект-кодирование.

    Пользователь не прописывает словарь атрибутов руками: уровни считываются
    из данных (или из явного словаря) методом :meth:`fit`.

    Parameters
    ----------
    attribute_cols
        Колонки-атрибуты продукта.
    interactions
        Пары атрибутов, для которых добавляются эффекты взаимодействия первого
        порядка. Для пары (A, B) добавляется (K_a - 1) * (K_b - 1) колонок,
        равных построчным произведениям колонок главных эффектов.
    include_none
        Добавить колонку None-концепта. Строки плана с ``is_none == 1`` не имеют
        атрибутов: все колонки главных эффектов и взаимодействий у них равны
        нулю, а единица стоит в колонке None. Соответствующий параметр — это
        индивидуальная константа альтернативы (ASC), она не принадлежит ни
        одному атрибуту и потому не участвует ни в ограничении нулевой суммы,
        ни в расчёте важности.

    Порядок колонок кодирования: главные эффекты, затем None, затем
    взаимодействия. Такой порядок делает блок «не-взаимодействий» непрерывным,
    что нужно для блочной структуры ковариации верхнего уровня.
    """

    def __init__(
        self,
        attribute_cols: Sequence[str],
        interactions: Sequence[tuple[str, str]] | None = None,
        include_none: bool = False,
    ) -> None:
        self.include_none = bool(include_none)
        self.attribute_cols = list(attribute_cols)
        if not self.attribute_cols:
            raise ValueError("attribute_cols не может быть пустым")
        if len(set(self.attribute_cols)) != len(self.attribute_cols):
            raise ValueError("attribute_cols содержит повторяющиеся имена")

        pairs: list[tuple[str, str]] = []
        seen: set[frozenset[str]] = set()
        for pair in interactions or []:
            if len(pair) != 2:
                raise ValueError(f"взаимодействие должно быть парой атрибутов, получено: {pair!r}")
            a, b = pair
            if a == b:
                raise ValueError(f"взаимодействие атрибута с самим собой недопустимо: {a!r}")
            for name in (a, b):
                if name not in self.attribute_cols:
                    raise ValueError(f"атрибут {name!r} из взаимодействия отсутствует в attribute_cols")
            key = frozenset((a, b))
            if key in seen:
                raise ValueError(f"взаимодействие {a!r} x {b!r} указано дважды")
            seen.add(key)
            pairs.append((a, b))
        self.interactions = pairs

        self.levels_: dict[str, list[Any]] | None = None
        self.reference_levels_: dict[str, Any] | None = None
        self.basis_: dict[str, np.ndarray] = {}
        self.main_columns_: list[str] = []
        self.none_columns_: list[str] = []
        self.interaction_columns_: list[str] = []
        self.coded_columns_: list[str] = []
        self.utility_columns_: list[str] = []

    # ------------------------------------------------------------------ fit

    def fit(self, source: pd.DataFrame | Mapping[str, Sequence[Any]]) -> "EffectsCoder":
        """Определить уровни атрибутов по DataFrame или по словарю."""
        if isinstance(source, pd.DataFrame):
            missing = [c for c in self.attribute_cols if c not in source.columns]
            if missing:
                raise KeyError(f"в данных нет колонок-атрибутов: {missing}")
            levels = {c: _sort_levels(pd.unique(source[c].dropna())) for c in self.attribute_cols}
        elif isinstance(source, Mapping):
            missing = [c for c in self.attribute_cols if c not in source]
            if missing:
                raise KeyError(f"в словаре уровней нет атрибутов: {missing}")
            levels = {c: _sort_levels(source[c]) for c in self.attribute_cols}
        else:
            raise TypeError("source должен быть DataFrame или Mapping[str, Sequence]")

        for attr, lv in levels.items():
            if len(lv) < 2:
                raise ValueError(f"атрибут {attr!r}: нужно минимум 2 уровня, найдено {len(lv)}")

        self.levels_ = levels
        self.reference_levels_ = {a: lv[-1] for a, lv in levels.items()}
        self.basis_ = {a: _effects_basis(len(lv)) for a, lv in levels.items()}

        self.main_columns_ = [
            f"{attr}={lvl}" for attr in self.attribute_cols for lvl in levels[attr][:-1]
        ]
        self.none_columns_ = [NONE_PARAM] if self.include_none else []
        self.interaction_columns_ = [
            f"{a}={la} x {b}={lb}"
            for a, b in self.interactions
            for la in levels[a][:-1]
            for lb in levels[b][:-1]
        ]
        self.coded_columns_ = (
            self.main_columns_ + self.none_columns_ + self.interaction_columns_
        )
        self.utility_columns_ = [
            f"{attr}={lvl}" for attr in self.attribute_cols for lvl in levels[attr]
        ]
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ структура

    def _check_fitted(self) -> None:
        if self.levels_ is None:
            raise RuntimeError("EffectsCoder не обучен: сначала вызовите fit()")

    @property
    def n_main(self) -> int:
        """Число колонок главных эффектов."""
        self._check_fitted()
        return len(self.main_columns_)

    @property
    def n_none(self) -> int:
        """Число колонок None-концепта (0 или 1)."""
        self._check_fitted()
        return len(self.none_columns_)

    @property
    def n_interaction(self) -> int:
        """Число колонок взаимодействий."""
        self._check_fitted()
        return len(self.interaction_columns_)

    @property
    def n_params(self) -> int:
        self._check_fitted()
        return len(self.coded_columns_)

    @property
    def n_structural(self) -> int:
        """Размер блока «не-взаимодействий»: главные эффекты плюс None.

        Именно этот блок получает полную LKJ-ковариацию в режиме ``block_lkj``.
        """
        return self.n_main + self.n_none

    @property
    def none_index_(self) -> int | None:
        """Позиция колонки None-концепта в закодированном пространстве."""
        self._check_fitted()
        return self.n_main if self.include_none else None

    @property
    def design_column_indices_(self) -> np.ndarray:
        """Индексы колонок, участвующих в оценке D-эффективности плана.

        Колонка None исключается: она равна единице ровно в одной альтернативе
        каждой задачи по построению, её среднее не равно нулю, и включение её в
        расчёт нарушило бы блочно-диагональную структуру ортогонального идеала.
        """
        self._check_fitted()
        keep = list(range(self.n_main))
        keep += list(range(self.n_structural, self.n_params))
        return np.array(keep, dtype=int)

    @property
    def coded_slices_(self) -> dict[str, slice]:
        """Срезы колонок главных эффектов в закодированном пространстве."""
        self._check_fitted()
        out: dict[str, slice] = {}
        start = 0
        for attr in self.attribute_cols:
            width = len(self.levels_[attr]) - 1
            out[attr] = slice(start, start + width)
            start += width
        return out

    @property
    def interaction_slices_(self) -> dict[tuple[str, str], slice]:
        """Срезы интеракционных колонок в закодированном пространстве."""
        self._check_fitted()
        out: dict[tuple[str, str], slice] = {}
        start = self.n_structural
        for a, b in self.interactions:
            width = (len(self.levels_[a]) - 1) * (len(self.levels_[b]) - 1)
            out[(a, b)] = slice(start, start + width)
            start += width
        return out

    @property
    def utility_slices_(self) -> dict[str, slice]:
        """Срезы атрибутов в пространстве полных полезностей (со всеми уровнями)."""
        self._check_fitted()
        out: dict[str, slice] = {}
        start = 0
        for attr in self.attribute_cols:
            width = len(self.levels_[attr])
            out[attr] = slice(start, start + width)
            start += width
        return out

    # ------------------------------------------------------------ transform

    def none_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Булев признак строк None-концепта."""
        self._check_fitted()
        if not self.include_none or NONE_COLUMN not in df.columns:
            return np.zeros(len(df), dtype=bool)
        return df[NONE_COLUMN].fillna(0).to_numpy().astype(bool)

    def _main_blocks(self, df: pd.DataFrame) -> list[np.ndarray]:
        self._check_fitted()
        missing = [c for c in self.attribute_cols if c not in df.columns]
        if missing:
            raise KeyError(f"в данных нет колонок-атрибутов: {missing}")
        if self.include_none and NONE_COLUMN not in df.columns:
            raise KeyError(
                f"include_none=True, но в плане нет колонки {NONE_COLUMN!r}: "
                f"строки None-концепта неотличимы от обычных профилей"
            )

        is_none = self.none_mask(df)
        blocks: list[np.ndarray] = []
        for attr in self.attribute_cols:
            lv = self.levels_[attr]
            index = {level: i for i, level in enumerate(lv)}
            pos = df[attr].map(index)
            # у None-концепта атрибутов нет, поэтому пропуски в его строках
            # ожидаемы и не считаются неизвестным уровнем
            invalid = pos.isna().to_numpy() & ~is_none
            if invalid.any():
                unknown = sorted({str(v) for v in df.loc[invalid, attr].unique()})
                raise ValueError(f"атрибут {attr!r}: неизвестные уровни {unknown}")
            block = np.zeros((len(df), len(lv) - 1))
            real = ~is_none
            if real.any():
                block[real] = self.basis_[attr][pos.to_numpy()[real].astype(int)]
            blocks.append(block)
        return blocks

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Матрица плана: главные эффекты, затем None, затем взаимодействия."""
        blocks = self._main_blocks(df)
        by_attr = dict(zip(self.attribute_cols, blocks))
        if self.include_none:
            blocks.append(self.none_mask(df).astype(float).reshape(-1, 1))
        for a, b in self.interactions:
            ea, eb = by_attr[a], by_attr[b]
            # построчное внешнее произведение -> (n, (Ka-1) * (Kb-1));
            # у строк None оба сомножителя нулевые, поэтому произведение тоже
            blocks.append((ea[:, :, None] * eb[:, None, :]).reshape(len(ea), -1))
        return np.hstack(blocks)

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """То же, что :meth:`transform`, но с именованными колонками."""
        return pd.DataFrame(self.transform(df), columns=self.coded_columns_, index=df.index)

    # ------------------------------------------------------- восстановление

    def expand_main(self, beta: np.ndarray) -> np.ndarray:
        """Восстановить полезности всех уровней, включая референсный.

        Референсный уровень каждого атрибута равен минус сумме оценённых
        уровней, поэтому сумма полезностей внутри атрибута строго равна нулю.
        Принимает массив формы (..., n_params) или (..., n_main).
        """
        self._check_fitted()
        beta = np.asarray(beta, dtype=float)
        if beta.shape[-1] not in (self.n_main, self.n_params):
            raise ValueError(
                f"ожидалось {self.n_main} или {self.n_params} колонок, получено {beta.shape[-1]}"
            )
        parts: list[np.ndarray] = []
        for attr, sl in self.coded_slices_.items():
            block = beta[..., sl]
            reference = -block.sum(axis=-1, keepdims=True)
            parts.append(np.concatenate([block, reference], axis=-1))
        return np.concatenate(parts, axis=-1)

    def expand_interaction(self, beta: np.ndarray, pair: tuple[str, str]) -> np.ndarray:
        """Полная таблица взаимодействия (..., K_a, K_b).

        Восстанавливается как ``E_a @ Gamma @ E_b.T``; суммы по строкам и по
        столбцам полученной таблицы строго равны нулю.
        """
        self._check_fitted()
        a, b = pair
        slices = self.interaction_slices_
        if (a, b) not in slices:
            if (b, a) in slices:
                return np.swapaxes(self.expand_interaction(beta, (b, a)), -1, -2)
            raise KeyError(f"взаимодействие {a!r} x {b!r} не объявлено")

        beta = np.asarray(beta, dtype=float)
        if beta.shape[-1] != self.n_params:
            raise ValueError(f"ожидалось {self.n_params} колонок, получено {beta.shape[-1]}")

        ka, kb = len(self.levels_[a]), len(self.levels_[b])
        gamma = beta[..., slices[(a, b)]].reshape(*beta.shape[:-1], ka - 1, kb - 1)
        return np.einsum("ip,...pq,jq->...ij", self.basis_[a], gamma, self.basis_[b])

    def ideal_information(self) -> np.ndarray:
        """Матрица X'X/n для идеально сбалансированного ортогонального плана.

        Для атрибута с K уровнями блок главных эффектов имеет 2/K на диагонали
        и 1/K вне её. Для взаимодействия блок равен кронекерову произведению
        блоков сомножителей, а все кросс-блоки равны нулю (следствие
        независимости и нулевого среднего эффект-кодированных колонок).
        Используется как знаменатель при расчёте D-эффективности.
        """
        self._check_fitted()
        blocks: dict[str, np.ndarray] = {}
        for attr, lv in self.levels_.items():
            k = len(lv)
            m = np.full((k - 1, k - 1), 1.0 / k)
            np.fill_diagonal(m, 2.0 / k)
            blocks[attr] = m

        ordered = [blocks[attr] for attr in self.attribute_cols]
        ordered += [np.kron(blocks[a], blocks[b]) for a, b in self.interactions]

        size = sum(m.shape[0] for m in ordered)
        ideal = np.zeros((size, size))
        start = 0
        for m in ordered:
            width = m.shape[0]
            ideal[start : start + width, start : start + width] = m
            start += width
        return ideal


def _relative_d_efficiency(x: np.ndarray, ideal: np.ndarray) -> float:
    """D-эффективность матрицы плана относительно ортогонального идеала.

    Возвращает ``(det(X'X/n) / det(M_ideal)) ** (1/p)``: 1.0 — идеально
    сбалансированный ортогональный план, 0.0 — вырожденный.
    """
    n, p = x.shape
    if n < p:
        return 0.0
    info = x.T @ x / n
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    sign_ideal, logdet_ideal = np.linalg.slogdet(ideal)
    if sign_ideal <= 0:
        raise ValueError("вырожденная матрица ортогонального идеала")
    return float(np.exp((logdet - logdet_ideal) / p))


def interaction_cell_counts(
    design_df: pd.DataFrame, pairs: Sequence[tuple[str, str]]
) -> pd.DataFrame:
    """Заполненность ячеек взаимодействия в разрезе респондентов.

    Ключевая диагностика разреженности: если у респондента ячейка (уровень A,
    уровень B) ни разу не показана, соответствующий индивидуальный параметр не
    идентифицирован данными и держится исключительно на шринкедже от модели
    верхнего уровня.
    """
    # строки None-концепта не несут атрибутов и в статистику ячеек не входят
    if NONE_COLUMN in design_df.columns:
        design_df = design_df[design_df[NONE_COLUMN].fillna(0) == 0]

    respondents = _sort_levels(pd.unique(design_df["respondent_id"]))
    rows = []
    for a, b in pairs:
        levels_a = _sort_levels(pd.unique(design_df[a].dropna()))
        levels_b = _sort_levels(pd.unique(design_df[b].dropna()))
        n_cells = len(levels_a) * len(levels_b)

        # полный декартов индекс: отсутствующие сочетания обязаны попасть в
        # отчёт нулями, иначе пустая ячейка просто исчезнет из статистики
        full_index = pd.MultiIndex.from_product(
            [respondents, levels_a, levels_b], names=["respondent_id", a, b]
        )
        counts = (
            design_df.groupby(["respondent_id", a, b], observed=True)
            .size()
            .reindex(full_index, fill_value=0)
            .to_numpy()
            .reshape(len(respondents), n_cells)
        )
        filled = (counts > 0).sum(axis=1)
        rows.append(
            {
                "pair": f"{a} x {b}",
                "n_cells": n_cells,
                "min_cell_count": int(counts.min()),
                "mean_cell_count": float(counts.mean()),
                "min_cells_filled": int(filled.min()),
                "respondents_with_empty_cells": int((filled < n_cells).sum()),
            }
        )
    return pd.DataFrame(rows)


class CBCDesignGenerator:
    """Генератор рандомизированного сбалансированного плана CBC.

    Parameters
    ----------
    source
        DataFrame, из которого автоматически считываются уровни атрибутов,
        либо словарь ``{атрибут: [уровни]}``.
    attribute_cols
        Колонки-атрибуты продукта.
    concepts_per_task
        Число профилей на одном экране.
    tasks_per_respondent
        Число задач выбора у одного респондента.
    num_respondents
        Число респондентов.
    control_attributes
        Атрибуты, уровни которых запрещено повторять внутри одной задачи
        (between-concept prohibitions). Защищает от доминирующих альтернатив
        вида «тот же бренд дважды, но дешевле».
    interactions
        Пары атрибутов, для которых планируется оценка взаимодействий. План
        оптимизируется с учётом интеракционных колонок, иначе ячейки A x B
        оказываются несбалансированными.
    include_none
        Добавлять в каждую задачу None-концепт («не выбрал бы ничего»). Он
        получает ``concept_id = concepts_per_task + 1``, признак ``is_none = 1``
        и пропуски во всех колонках-атрибутах. В баланс уровней, запреты и
        расчёт D-эффективности он не входит: у него нет атрибутов.
        Обратите внимание, что ``concepts_per_task`` задаёт число *реальных*
        профилей, а None добавляется сверх него.
    n_starts
        Число случайных перезапусков; выбирается план с лучшей средней
        D-эффективностью по респондентам.
    """

    def __init__(
        self,
        source: pd.DataFrame | Mapping[str, Sequence[Any]],
        attribute_cols: Sequence[str],
        concepts_per_task: int,
        tasks_per_respondent: int,
        num_respondents: int,
        control_attributes: Sequence[str] | str | None = None,
        interactions: Sequence[tuple[str, str]] | None = None,
        include_none: bool = False,
        n_starts: int = 20,
        random_state: int | None = None,
    ) -> None:
        if concepts_per_task < 2:
            raise ValueError("concepts_per_task должно быть >= 2")
        if tasks_per_respondent < 1:
            raise ValueError("tasks_per_respondent должно быть >= 1")
        if num_respondents < 1:
            raise ValueError("num_respondents должно быть >= 1")
        if n_starts < 1:
            raise ValueError("n_starts должно быть >= 1")

        self.coder = EffectsCoder(
            attribute_cols, interactions=interactions, include_none=include_none
        ).fit(source)
        self.attribute_cols = self.coder.attribute_cols
        self.levels_ = self.coder.levels_
        self.include_none = bool(include_none)
        self.concepts_per_task = int(concepts_per_task)
        self.tasks_per_respondent = int(tasks_per_respondent)
        self.num_respondents = int(num_respondents)
        self.n_starts = int(n_starts)
        self.random_state = random_state

        if control_attributes is None:
            controls: list[str] = []
        elif isinstance(control_attributes, str):
            controls = [control_attributes]
        else:
            controls = list(control_attributes)
        for attr in controls:
            if attr not in self.attribute_cols:
                raise ValueError(f"control_attribute {attr!r} отсутствует в attribute_cols")
            n_levels = len(self.levels_[attr])
            if n_levels < self.concepts_per_task:
                raise ValueError(
                    f"невыполнимое ограничение: у атрибута {attr!r} {n_levels} уровней, "
                    f"а в задаче {self.concepts_per_task} концептов — запретить дублирование "
                    f"невозможно. Увеличьте число уровней либо уменьшите concepts_per_task."
                )
        self.control_attributes = controls

        self.design_: pd.DataFrame | None = None
        self.d_efficiency_: float | None = None
        self.min_d_efficiency_: float | None = None

    # ------------------------------------------------------- случайный план

    def _balanced_deck(self, levels: Sequence[Any], n: int, rng: np.random.Generator) -> list[Any]:
        """Колода уровней длины n с максимально ровными частотами."""
        k = len(levels)
        deck = list(levels) * (n // k)
        remainder = n - len(deck)
        if remainder:
            extra = rng.permutation(k)[:remainder]
            deck += [levels[i] for i in extra]
        order = rng.permutation(len(deck))
        return [deck[i] for i in order]

    def _control_column(self, levels: Sequence[Any], rng: np.random.Generator) -> list[Any]:
        """Уровни контрольного атрибута: уникальные внутри задачи, ровные по частоте.

        Внутри задачи уровни выбираются без повторов, поэтому запрет соблюдается
        по построению. Среди допустимых кандидатов предпочитается наименее
        использованный уровень, что выравнивает частоты по всему плану.
        """
        counts = dict.fromkeys(levels, 0)
        out: list[Any] = []
        for _ in range(self.tasks_per_respondent):
            chosen: list[Any] = []
            for _ in range(self.concepts_per_task):
                available = [lv for lv in levels if lv not in chosen]
                fewest = min(counts[lv] for lv in available)
                candidates = [lv for lv in available if counts[lv] == fewest]
                pick = candidates[int(rng.integers(len(candidates)))]
                chosen.append(pick)
                counts[pick] += 1
            order = rng.permutation(len(chosen))
            out.extend(chosen[i] for i in order)
        return out

    def _build_design(self, rng: np.random.Generator) -> pd.DataFrame:
        n_slots = self.tasks_per_respondent * self.concepts_per_task
        frames = []
        for respondent in range(1, self.num_respondents + 1):
            data: dict[str, Any] = {
                "respondent_id": respondent,
                "task_id": np.repeat(
                    np.arange(1, self.tasks_per_respondent + 1), self.concepts_per_task
                ),
                "concept_id": np.tile(
                    np.arange(1, self.concepts_per_task + 1), self.tasks_per_respondent
                ),
            }
            for attr in self.attribute_cols:
                levels = self.levels_[attr]
                if attr in self.control_attributes:
                    data[attr] = self._control_column(levels, rng)
                else:
                    data[attr] = self._balanced_deck(levels, n_slots, rng)
            frame = pd.DataFrame(data)
            if self.include_none:
                frame[NONE_COLUMN] = 0
                frame = pd.concat([frame, self._none_rows(respondent)], ignore_index=True)
                frame = frame.sort_values(list(ID_COLUMNS), ignore_index=True)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _none_rows(self, respondent: int) -> pd.DataFrame:
        """По одной строке None-концепта на каждую задачу респондента."""
        rows = pd.DataFrame(
            {
                "respondent_id": respondent,
                "task_id": np.arange(1, self.tasks_per_respondent + 1),
                "concept_id": self.concepts_per_task + 1,
                NONE_COLUMN: 1,
            }
        )
        for attr in self.attribute_cols:
            # у None-концепта атрибутов нет: пропуск, а не служебный уровень,
            # иначе он был бы распознан как обычный уровень атрибута
            rows[attr] = pd.NA
        return rows

    def _real_rows(self, design: pd.DataFrame) -> pd.DataFrame:
        """Строки реальных профилей: без None-концепта."""
        if NONE_COLUMN not in design.columns:
            return design
        return design[design[NONE_COLUMN].fillna(0) == 0]

    def _mean_efficiency(self, design: pd.DataFrame) -> tuple[float, float]:
        """Средняя и минимальная D-эффективность по индивидуальным блокам плана.

        Для HB значима именно индивидуальная эффективность: полезности
        оцениваются для каждого респондента отдельно. Строки и колонка
        None-концепта в расчёт не входят — у него нет атрибутов, а его колонка
        не центрирована и нарушила бы структуру ортогонального идеала.
        """
        real = self._real_rows(design)
        ideal = self.coder.ideal_information()
        x = self.coder.transform(real)[:, self.coder.design_column_indices_]
        respondents = real["respondent_id"].to_numpy()
        scores = [
            _relative_d_efficiency(x[respondents == r], ideal)
            for r in np.unique(respondents)
        ]
        return float(np.mean(scores)), float(np.min(scores))

    def generate(self) -> pd.DataFrame:
        """Построить план: n_starts случайных стартов, лучший по D-эффективности."""
        rng = np.random.default_rng(self.random_state)
        best: pd.DataFrame | None = None
        best_scores = (-np.inf, -np.inf)
        for _ in range(self.n_starts):
            candidate = self._build_design(rng)
            scores = self._mean_efficiency(candidate)
            if scores[0] > best_scores[0]:
                best, best_scores = candidate, scores

        assert best is not None
        violations = self.check_prohibitions(best)
        if violations:
            raise RuntimeError(f"нарушен запрет на дублирование уровней: {violations}")

        self.design_ = best
        self.d_efficiency_, self.min_d_efficiency_ = best_scores
        return best

    # ------------------------------------------------------------- отчёты

    def check_prohibitions(self, design: pd.DataFrame | None = None) -> dict[str, int]:
        """Число задач, где уровень контрольного атрибута повторился."""
        design = self.design_ if design is None else design
        if design is None:
            raise RuntimeError("план не сгенерирован: сначала вызовите generate()")
        # None-концепт исключается: у него нет уровня, и пропуск в подсчёте
        # уникальных значений выглядел бы как ложное нарушение запрета
        real = self._real_rows(design)
        violations: dict[str, int] = {}
        for attr in self.control_attributes:
            duplicated = (
                real.groupby(["respondent_id", "task_id"], observed=True)[attr]
                .agg(lambda s: s.nunique() != len(s))
                .sum()
            )
            if duplicated:
                violations[attr] = int(duplicated)
        return violations

    def balance_report(self, design: pd.DataFrame | None = None) -> pd.DataFrame:
        """Частоты уровней по всему плану и отклонение от идеального баланса."""
        design = self.design_ if design is None else design
        if design is None:
            raise RuntimeError("план не сгенерирован: сначала вызовите generate()")
        real = self._real_rows(design)
        rows = []
        total = len(real)
        for attr in self.attribute_cols:
            expected = total / len(self.levels_[attr])
            counts = real[attr].value_counts()
            for level in self.levels_[attr]:
                count = int(counts.get(level, 0))
                rows.append(
                    {
                        "attribute": attr,
                        "level": level,
                        "count": count,
                        "expected": expected,
                        "deviation_pct": 100.0 * (count - expected) / expected,
                    }
                )
        return pd.DataFrame(rows)

    def interaction_report(self, design: pd.DataFrame | None = None) -> pd.DataFrame:
        """Заполненность ячеек взаимодействий (диагностика разреженности)."""
        design = self.design_ if design is None else design
        if design is None:
            raise RuntimeError("план не сгенерирован: сначала вызовите generate()")
        if not self.coder.interactions:
            return pd.DataFrame(
                columns=["pair", "n_cells", "min_cell_count", "mean_cell_count",
                         "min_cells_filled", "respondents_with_empty_cells"]
            )
        return interaction_cell_counts(design, self.coder.interactions)

    def design_matrix(self, design: pd.DataFrame | None = None) -> pd.DataFrame:
        """План с эффект-кодированными колонками рядом с исходными метками."""
        design = self.design_ if design is None else design
        if design is None:
            raise RuntimeError("план не сгенерирован: сначала вызовите generate()")
        coded = self.coder.transform_frame(design)
        return pd.concat([design.reset_index(drop=True), coded.reset_index(drop=True)], axis=1)


class CBCHierarchicalBayesEstimator:
    """Иерархическая байесовская оценка индивидуальных полезностей (CBC/HB).

    Структура модели
    ----------------
    Нижний уровень — мультиномиальный логит: вероятность выбора альтернативы j
    респондентом i в задаче t есть softmax от ``X_itj @ beta_i``.

    Верхний уровень — многомерная нормальная модель популяции
    ``beta_i ~ N(alpha, D)``, реализованная в **нецентрированном** виде
    ``beta_i = alpha + L @ z_i``, где ``z_i ~ N(0, I)``, а ``L`` — фактор
    Холецкого из приора LKJ. Прямая (центрированная) запись создаёт
    воронкообразную геометрию апостериорного распределения, на которой NUTS
    систематически расходится; нецентрированная запись эту геометрию убирает.

    Взаимодействия и структура ковариации
    -------------------------------------
    Добавление эффектов взаимодействия резко увеличивает размерность: полная
    матрица D содержит ``P (P + 1) / 2`` свободных параметров, оцениваемых по
    числу респондентов. Для 19 параметров это 190 значений — при выборке в
    30 человек модель переобучается, а цепи расходятся. Параметр
    ``upper_level_cov`` управляет структурой:

    ``"full_lkj"``
        Полная ковариация LKJ по всем параметрам.
    ``"block_lkj"``
        LKJ по блоку главных эффектов, диагональ по блоку взаимодействий,
        кросс-корреляции между блоками зануляются.
    ``"diagonal"``
        Независимые дисперсии по всем параметрам.
    ``"auto"``
        ``"full_lkj"`` без взаимодействий, ``"block_lkj"`` с ними.

    Дополнительно интеракционные члены регуляризуются: их средние получают
    более узкий приор, а масштабы — ``HalfNormal`` с малым сигма, что даёт
    целенаправленный шринкедж к нулю.

    Режим allocation и вес наблюдения
    ---------------------------------
    Распределение баллов моделируется взвешенным логарифмическим
    правдоподобием, а не ``pm.Multinomial`` с ``n = allocation_total``.
    Разница принципиальна: мультиномиальное правдоподобие трактует 100
    распределённых баллов как 100 независимых выборов, тогда как это одно
    суждение, выраженное на шкале. На замере (8 респондентов, 10 задач) переход
    от веса 1 к весу 100 сужает апостериорные интервалы индивидуальных
    полезностей примерно втрое; коэффициент зависит от того, насколько сильно
    шринкедж верхнего уровня удерживает индивидуальные оценки, и растёт с
    числом задач. Опасность в том, что диагностика при этом молчит: ``r_hat``,
    ESS и число дивергенций выглядят нормально, а интервалы уже занижены.

    Поэтому доли баллов переводятся в дробный вес задачи, а ``allocation_weight``
    задаёт информативность одной задачи в единицах «одного выбора». Значение
    1.0 консервативно; значения 2-3 защитимы, если считать, что аллокация несёт
    больше информации, чем принудительный выбор. Установка веса равным сумме
    баллов воспроизводит наивное поведение и не рекомендуется.

    Parameters
    ----------
    attribute_cols
        Колонки-атрибуты продукта.
    interactions
        Пары атрибутов для оценки взаимодействий первого порядка.
    response_mode
        Режим ответов. ``"single_choice"`` — респондент выбирает ровно один
        концепт. ``"allocation"`` — распределяет фиксированную сумму (обычно
        100%) между концептами. Схема ``responses_df`` в обоих случаях одна:
        идентификаторы плюс одна числовая колонка ответа; меняются только
        правила её проверки.
    include_none
        Учитывать None-концепт («не выбрал бы ничего»). Требует, чтобы в плане
        была колонка ``is_none`` и ровно один такой концепт в каждой задаче.
        Оценивается как индивидуальная константа альтернативы; в важность
        атрибутов не входит.
    allocation_total
        Ожидаемая сумма баллов в задаче для режима ``allocation``. ``None`` —
        определить по первой задаче и потребовать того же от остальных.
    allocation_weight
        Вес одной задачи с аллокацией в правдоподобии. По умолчанию 1.0: задача
        несёт столько же информации, сколько один принудительный выбор.
        Увеличивайте осознанно — см. ниже.
    draws, tune, chains, cores
        Параметры NUTS. ``tune`` — итерации прогрева (burn-in).
    target_accept
        Целевая доля принятия. ``None`` — 0.9 без взаимодействий и 0.95 с ними.
    upper_level_cov
        Структура ковариации верхнего уровня (см. выше).
    normalization
        ``"zcd"`` — zero-centered diffs: полезности респондента домножаются на
        ``100 * n_attributes / сумма размахов``, что делает их сопоставимыми
        между людьми. ``"raw"`` — без масштабирования.
    importance_mode
        ``"main"`` — важность считается по главным эффектам, как в Sawtooth.
        ``"joint"`` — взаимодействующая пара отчитывается одной строкой по
        размаху совместной таблицы полезностей.
    """

    _COV_MODES = ("auto", "full_lkj", "block_lkj", "diagonal")
    _RESPONSE_MODES = ("single_choice", "allocation")

    def __init__(
        self,
        attribute_cols: Sequence[str],
        interactions: Sequence[tuple[str, str]] | None = None,
        response_mode: str = "single_choice",
        include_none: bool = False,
        allocation_total: float | None = None,
        allocation_weight: float = 1.0,
        draws: int = 1000,
        tune: int = 500,
        chains: int = 4,
        cores: int | None = None,
        target_accept: float | None = None,
        upper_level_cov: str = "auto",
        normalization: str = "zcd",
        importance_mode: str = "main",
        prior_alpha_sd: float = 5.0,
        prior_alpha_sd_interaction: float = 1.0,
        interaction_sd_prior: float = 0.5,
        lkj_eta: float = 2.0,
        random_seed: int | None = 42,
        progressbar: bool = True,
    ) -> None:
        if upper_level_cov not in self._COV_MODES:
            raise ValueError(f"upper_level_cov должен быть одним из {self._COV_MODES}")
        if normalization not in ("zcd", "raw"):
            raise ValueError("normalization должен быть 'zcd' или 'raw'")
        if importance_mode not in ("main", "joint"):
            raise ValueError("importance_mode должен быть 'main' или 'joint'")

        if response_mode not in self._RESPONSE_MODES:
            raise ValueError(f"response_mode должен быть одним из {self._RESPONSE_MODES}")
        if allocation_weight <= 0:
            raise ValueError("allocation_weight должен быть положительным")

        self.coder = EffectsCoder(
            attribute_cols, interactions=interactions, include_none=include_none
        )
        self.attribute_cols = self.coder.attribute_cols
        self.interactions = self.coder.interactions
        self.include_none = self.coder.include_none
        self.response_mode = response_mode
        self.allocation_total = allocation_total
        self.allocation_weight = float(allocation_weight)
        self.draws = int(draws)
        self.tune = int(tune)
        self.chains = int(chains)
        self.cores = cores
        self.target_accept = target_accept
        self.upper_level_cov = upper_level_cov
        self.normalization = normalization
        self.importance_mode = importance_mode
        self.prior_alpha_sd = float(prior_alpha_sd)
        self.prior_alpha_sd_interaction = float(prior_alpha_sd_interaction)
        self.interaction_sd_prior = float(interaction_sd_prior)
        self.lkj_eta = float(lkj_eta)
        self.random_seed = random_seed
        self.progressbar = progressbar

        self.idata_: Any = None
        self.model_: Any = None
        self.respondent_ids_: np.ndarray | None = None
        self.cov_mode_: str | None = None
        self.individual_results_: pd.DataFrame | None = None
        self.population_summary_: pd.DataFrame | None = None
        self.interaction_tables_: dict[tuple[str, str], pd.DataFrame] = {}
        self.diagnostics_: dict[str, Any] = {}
        self.sparsity_report_: pd.DataFrame | None = None
        self.utilities_raw_: pd.DataFrame | None = None

    # ---------------------------------------------------------- подготовка

    @staticmethod
    def resolve_response_column(responses_df: pd.DataFrame) -> str:
        """Найти колонку с ответом среди допустимых имён.

        Схема ``responses_df`` одна на оба режима: идентификаторы плюс одна
        числовая колонка ответа. Имя берётся первым из
        ``response`` / ``chosen`` / ``allocation`` — это синонимы, а не
        разные форматы.
        """
        for name in RESPONSE_COLUMNS:
            if name in responses_df.columns:
                return name
        raise KeyError(
            f"responses_df: нет колонки с ответом, ожидалась одна из {list(RESPONSE_COLUMNS)}"
        )

    def _validate_responses(self, values: np.ndarray, keys: pd.DataFrame) -> None:
        """Проверить ответы по правилам выбранного режима.

        Обе ветки работают с одним и тем же массивом ``(задачи, альтернативы)``:
        режим меняет правила, но не структуру данных.
        """
        if np.isnan(values).any():
            raise ValueError("в ответах есть пропуски")
        if (values < 0).any():
            raise ValueError("ответы не могут быть отрицательными")

        totals = values.sum(axis=1)
        if self.response_mode == "single_choice":
            one_hot = np.isin(values, (0.0, 1.0)).all(axis=1) & (totals == 1.0)
            bad = np.flatnonzero(~one_hot)
            if bad.size:
                sample = keys.iloc[bad[:3]].to_dict("records")
                raise ValueError(
                    f"режим single_choice требует ровно один выбор на задачу "
                    f"(значения 0/1); нарушено в {bad.size} задачах, например: {sample}"
                )
            return

        expected = self.allocation_total
        if expected is None:
            expected = float(totals[0])
            self.allocation_total = expected
        if expected <= 0:
            raise ValueError("allocation_total должен быть положительным")
        bad = np.flatnonzero(~np.isclose(totals, expected))
        if bad.size:
            sample = keys.iloc[bad[:3]].to_dict("records")
            raise ValueError(
                f"режим allocation требует, чтобы сумма по задаче равнялась "
                f"{expected:g}; нарушено в {bad.size} задачах, например: {sample} "
                f"(суммы {np.round(totals[bad[:3]], 3).tolist()})"
            )

    def _prepare(
        self, design_df: pd.DataFrame, responses_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        for name, frame in (("design_df", design_df), ("responses_df", responses_df)):
            missing = [c for c in ID_COLUMNS if c not in frame.columns]
            if missing:
                raise KeyError(f"{name}: отсутствуют колонки {missing}")
        column = self.resolve_response_column(responses_df)

        merged = design_df.merge(
            responses_df[[*ID_COLUMNS, column]].rename(columns={column: "response"}),
            on=list(ID_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        if merged["response"].isna().any():
            n = int(merged["response"].isna().sum())
            raise ValueError(f"для {n} строк плана нет ответа в responses_df")

        merged = merged.sort_values(list(ID_COLUMNS)).reset_index(drop=True)

        sizes = merged.groupby(list(ID_COLUMNS[:2]), observed=True).size()
        n_concepts = int(sizes.iloc[0])
        if not (sizes == n_concepts).all():
            raise ValueError(
                "модель требует одинакового числа концептов во всех задачах; "
                f"найдены задачи с {sorted(sizes.unique().tolist())} концептами"
            )
        n_tasks_per = merged.groupby("respondent_id", observed=True)["task_id"].nunique()
        n_tasks = int(n_tasks_per.iloc[0])
        if not (n_tasks_per == n_tasks).all():
            raise ValueError(
                "модель требует одинакового числа задач у всех респондентов; "
                f"найдено от {n_tasks_per.min()} до {n_tasks_per.max()}"
            )

        if self.include_none:
            if NONE_COLUMN not in merged.columns:
                raise KeyError(
                    f"include_none=True, но в плане нет колонки {NONE_COLUMN!r}: "
                    f"строки None-концепта неотличимы от обычных профилей"
                )
            per_task_none = merged.groupby(list(ID_COLUMNS[:2]), observed=True)[NONE_COLUMN].sum()
            if not (per_task_none == 1).all():
                raise ValueError(
                    "include_none=True требует ровно один None-концепт в каждой задаче; "
                    f"найдено от {per_task_none.min()} до {per_task_none.max()}"
                )

        respondents = merged["respondent_id"].drop_duplicates().to_numpy()
        n_resp = len(respondents)

        values = merged["response"].to_numpy(dtype=float)
        self._validate_responses(
            values.reshape(-1, n_concepts),
            merged.iloc[::n_concepts][list(ID_COLUMNS[:2])].reset_index(drop=True),
        )

        x = self.coder.transform(merged).reshape(n_resp, n_tasks, n_concepts, self.coder.n_params)
        response = values.reshape(n_resp, n_tasks, n_concepts)
        return x, response, respondents, merged

    def _check_sparsity(self, design_df: pd.DataFrame) -> None:
        """Проверить заполненность ячеек взаимодействий до запуска MCMC."""
        if not self.interactions:
            self.sparsity_report_ = pd.DataFrame()
            return
        report = interaction_cell_counts(design_df, self.interactions)
        self.sparsity_report_ = report
        for row in report.itertuples():
            if row.min_cell_count == 0:
                warnings.warn(
                    f"взаимодействие {row.pair}: у {row.respondents_with_empty_cells} респондентов "
                    f"есть непоказанные ячейки (минимум заполнено {row.min_cells_filled} из "
                    f"{row.n_cells}). Соответствующие индивидуальные параметры не идентифицированы "
                    f"данными и определяются шринкеджем от модели верхнего уровня.",
                    stacklevel=3,
                )
            elif row.mean_cell_count < 2.0:
                warnings.warn(
                    f"взаимодействие {row.pair}: в среднем {row.mean_cell_count:.2f} наблюдения "
                    f"на ячейку. Оценка будет крайне зашумлённой — увеличьте число задач.",
                    stacklevel=3,
                )

    def _resolve_cov_mode(self) -> str:
        if self.upper_level_cov != "auto":
            return self.upper_level_cov
        return "block_lkj" if self.coder.n_interaction else "full_lkj"

    # ---------------------------------------------------------------- модель

    def _build_model(self, x: np.ndarray, response: np.ndarray, respondents: np.ndarray) -> Any:
        import pymc as pm
        import pytensor.tensor as pt

        n_resp, n_tasks, n_concepts, n_params = x.shape
        # блок «не-взаимодействий» — главные эффекты плюс константа None
        n_structural = self.coder.n_structural
        n_int = self.coder.n_interaction
        mode = self.cov_mode_

        coords = {
            "param": self.coder.coded_columns_,
            "param_structural": self.coder.main_columns_ + self.coder.none_columns_,
            "param_interaction": self.coder.interaction_columns_,
            "respondent": respondents.tolist(),
        }

        alpha_sd = np.concatenate(
            [
                np.full(n_structural, self.prior_alpha_sd),
                np.full(n_int, self.prior_alpha_sd_interaction),
            ]
        )

        with pm.Model(coords=coords) as model:
            x_t = pt.as_tensor_variable(x)
            alpha = pm.Normal("alpha", mu=0.0, sigma=alpha_sd, dims="param")

            if mode == "diagonal":
                sigma_sd = np.concatenate(
                    [np.ones(n_structural), np.full(n_int, self.interaction_sd_prior)]
                )
                sigma = pm.HalfNormal("sigma", sigma=sigma_sd, dims="param")
                z = pm.Normal("z", 0.0, 1.0, dims=("respondent", "param"))
                beta_raw = alpha + z * sigma
            elif mode == "full_lkj" or n_int == 0:
                chol, _, _ = pm.LKJCholeskyCov(
                    "L",
                    n=n_params,
                    eta=self.lkj_eta,
                    sd_dist=pm.Exponential.dist(1.0, shape=n_params),
                    compute_corr=True,
                )
                z = pm.Normal("z", 0.0, 1.0, dims=("respondent", "param"))
                beta_raw = alpha + pt.dot(z, chol.T)
            else:  # block_lkj
                chol, _, _ = pm.LKJCholeskyCov(
                    "L_structural",
                    n=n_structural,
                    eta=self.lkj_eta,
                    sd_dist=pm.Exponential.dist(1.0, shape=n_structural),
                    compute_corr=True,
                )
                z_structural = pm.Normal(
                    "z_structural", 0.0, 1.0, dims=("respondent", "param_structural")
                )
                beta_structural = alpha[:n_structural] + pt.dot(z_structural, chol.T)

                sigma_int = pm.HalfNormal(
                    "sigma_interaction", sigma=self.interaction_sd_prior, dims="param_interaction"
                )
                z_int = pm.Normal("z_interaction", 0.0, 1.0, dims=("respondent", "param_interaction"))
                beta_int = alpha[n_structural:] + z_int * sigma_int
                beta_raw = pt.concatenate([beta_structural, beta_int], axis=1)

            beta = pm.Deterministic("beta", beta_raw, dims=("respondent", "param"))
            utility = (x_t * beta[:, None, None, :]).sum(axis=-1)

            if self.response_mode == "single_choice":
                pm.Categorical(
                    "choice",
                    logit_p=utility.reshape((n_resp * n_tasks, n_concepts)),
                    observed=response.argmax(axis=-1).reshape(n_resp * n_tasks),
                )
            else:
                # Взвешенное правдоподобие вместо pm.Multinomial: доли баллов
                # переводятся в дробный вес задачи. Мультиномиальное
                # правдоподобие с n = allocation_total считало бы 100 баллов
                # сотней независимых выборов и сузило бы апостериорные интервалы
                # примерно в 10 раз без единого предупреждения в диагностике.
                weights = response / response.sum(axis=-1, keepdims=True)
                weights = weights * self.allocation_weight
                log_p = pt.special.log_softmax(utility, axis=-1)
                pm.Potential("allocation", (pt.as_tensor_variable(weights) * log_p).sum())
        return model

    # ------------------------------------------------------------------ fit

    def fit(
        self, design_df: pd.DataFrame, responses_df: pd.DataFrame
    ) -> "CBCHierarchicalBayesEstimator":
        """Оценить модель и рассчитать индивидуальные полезности."""
        import pymc as pm

        self.coder.fit(design_df)
        self._check_sparsity(design_df)
        x, response, respondents, _ = self._prepare(design_df, responses_df)
        self.respondent_ids_ = respondents
        self.cov_mode_ = self._resolve_cov_mode()

        n_resp, n_params = len(respondents), self.coder.n_params
        if self.cov_mode_ == "full_lkj" and n_params * (n_params + 1) // 2 > 2 * n_resp:
            warnings.warn(
                f"полная ковариация верхнего уровня требует "
                f"{n_params * (n_params + 1) // 2} свободных параметров при {n_resp} респондентах. "
                f"Риск переобучения и расхождения цепей высок — рассмотрите "
                f"upper_level_cov='block_lkj' или 'diagonal'.",
                stacklevel=2,
            )

        target_accept = self.target_accept
        if target_accept is None:
            target_accept = 0.95 if self.coder.n_interaction else 0.9

        self.model_ = self._build_model(x, response, respondents)
        with self.model_:
            self.idata_ = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=target_accept,
                random_seed=self.random_seed,
                # прогрев сохраняется: без него нечего показывать в серой зоне
                # графика истории итераций
                discard_tuned_samples=False,
                progressbar=self.progressbar,
            )

        self._collect_diagnostics()
        self._postprocess(x, response)
        return self

    # -------------------------------------------------------- диагностика

    def _collect_diagnostics(self) -> None:
        import arviz as az

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = az.summary(self.idata_, var_names=["alpha", "beta"], kind="diagnostics")

        divergences = int(self.idata_.sample_stats["diverging"].to_numpy().sum())
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
        self.diagnostics_ = {
            "max_r_hat": max_rhat,
            "min_ess_bulk": min_ess,
            "divergences": divergences,
            "n_params": self.coder.n_params,
            "upper_level_cov": self.cov_mode_,
            "summary": summary,
        }

        if divergences:
            warnings.warn(
                f"NUTS зафиксировал {divergences} расходящихся переходов — оценки смещены. "
                f"Поднимите target_accept, увеличьте tune либо упростите структуру "
                f"ковариации верхнего уровня.",
                stacklevel=3,
            )
        if max_rhat > 1.01:
            warnings.warn(
                f"цепи не сошлись: max r_hat = {max_rhat:.3f} > 1.01. Увеличьте draws/tune.",
                stacklevel=3,
            )
        if min_ess < 400:
            warnings.warn(
                f"низкий эффективный размер выборки: min ESS = {min_ess:.0f}. "
                f"Оценки хвостов ненадёжны.",
                stacklevel=3,
            )

    # -------------------------------------------------------- постобработка

    def _posterior_beta(self) -> np.ndarray:
        """Апостериорные выборки индивидуальных параметров, форма (S, N, P)."""
        return (
            self.idata_.posterior["beta"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "respondent", "param")
            .to_numpy()
        )

    def _attribute_ranges(self, full: np.ndarray) -> np.ndarray:
        """Размах полезностей внутри каждого атрибута, форма (..., n_attributes)."""
        slices = self.coder.utility_slices_
        return np.stack(
            [full[..., sl].max(axis=-1) - full[..., sl].min(axis=-1) for sl in slices.values()],
            axis=-1,
        )

    def _importance_groups(
        self, beta_samples: np.ndarray, full: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Метки групп и их размахи полезностей.

        В режиме ``"main"`` группа — это атрибут, а взаимодействия в расчёт
        важности не входят: интеракционный член не принадлежит ни одному
        атрибуту по отдельности, и Sawtooth в этой ситуации отчитывает важность
        по главным эффектам. В режиме ``"joint"`` взаимодействующая пара
        схлопывается в одну группу с размахом по совместной таблице.
        """
        if self.importance_mode == "main" or not self.interactions:
            return list(self.coder.attribute_cols), self._attribute_ranges(full)

        member_of: dict[str, tuple[str, str]] = {}
        for pair in self.interactions:
            for attr in pair:
                if attr in member_of:
                    raise ValueError(
                        f"importance_mode='joint' требует, чтобы атрибут участвовал не более чем "
                        f"в одном взаимодействии; {attr!r} входит и в {member_of[attr]}, "
                        f"и в {pair}. Используйте importance_mode='main'."
                    )
                member_of[attr] = pair

        slices = self.coder.utility_slices_
        labels: list[str] = []
        ranges: list[np.ndarray] = []
        handled: set[tuple[str, str]] = set()
        for attr in self.coder.attribute_cols:
            pair = member_of.get(attr)
            if pair is None:
                block = full[..., slices[attr]]
                labels.append(attr)
                ranges.append(block.max(axis=-1) - block.min(axis=-1))
                continue
            if pair in handled:
                continue
            handled.add(pair)
            a, b = pair
            joint = (
                full[..., slices[a]][..., :, None]
                + full[..., slices[b]][..., None, :]
                + self.coder.expand_interaction(beta_samples, pair)
            )
            flat = joint.reshape(*joint.shape[:-2], -1)
            labels.append(f"{a}*{b}")
            ranges.append(flat.max(axis=-1) - flat.min(axis=-1))
        return labels, np.stack(ranges, axis=-1)

    def _root_likelihood(
        self, beta_samples: np.ndarray, x: np.ndarray, response: np.ndarray
    ) -> np.ndarray:
        """RLH — среднее геометрическое вероятности, взвешенное по ответу.

        Для ``single_choice`` веса — это индикатор выбранной альтернативы, и
        формула сводится к классическому RLH Sawtooth. Для ``allocation`` веса —
        доли распределённых баллов, то есть обобщение той же величины на
        непрерывный ответ. Считается на каждой итерации и усредняется по
        апостериорной выборке; ориентир случайного выбора — 1 / число
        альтернатив.
        """
        n_samples = beta_samples.shape[0]
        budget = 40_000_000
        cost = n_samples * response.size
        if cost > budget:
            stride = int(np.ceil(cost / budget))
            beta_samples = beta_samples[::stride]
        utility = np.einsum("snp,ntjp->sntj", beta_samples, x)
        shift = utility.max(axis=-1, keepdims=True)
        log_norm = shift + np.log(np.exp(utility - shift).sum(axis=-1, keepdims=True))
        log_prob = utility - log_norm
        shares = response / response.sum(axis=-1, keepdims=True)
        weighted = (shares[None, ...] * log_prob).sum(axis=-1)
        return np.exp(weighted.mean(axis=-1)).mean(axis=0)

    def _postprocess(self, x: np.ndarray, response: np.ndarray) -> None:
        beta_samples = self._posterior_beta()
        self.beta_mean_ = beta_samples.mean(axis=0)
        self.respondent_index_ = {rid: i for i, rid in enumerate(self.respondent_ids_)}

        full = self.coder.expand_main(beta_samples)  # (S, N, L)
        labels, group_ranges = self._importance_groups(beta_samples, full)

        # важность считается на каждой итерации и усредняется: range() нелинеен,
        # поэтому importance(mean(beta)) != mean(importance(beta))
        totals = group_ranges.sum(axis=-1, keepdims=True)
        safe = np.where(totals > 0, totals, 1.0)
        importance = np.where(totals > 0, 100.0 * group_ranges / safe, np.nan).mean(axis=0)

        # zero-centered diffs: приводит средний размах атрибута к 100 и делает
        # полезности сопоставимыми между респондентами с разным масштабом логита
        if self.normalization == "zcd":
            main_totals = self._attribute_ranges(full).sum(axis=-1, keepdims=True)
            scale = np.where(
                main_totals > 0, 100.0 * len(self.coder.attribute_cols) / np.where(main_totals > 0, main_totals, 1.0), 1.0
            )
        else:
            scale = np.ones((*full.shape[:2], 1))

        utilities = (full * scale).mean(axis=0)
        self.utilities_raw_ = pd.DataFrame(
            full.mean(axis=0), index=self.respondent_ids_, columns=self.coder.utility_columns_
        )
        self.utilities_raw_.index.name = "respondent_id"

        frame = pd.DataFrame(
            utilities, index=self.respondent_ids_, columns=self.coder.utility_columns_
        )
        frame.index.name = "respondent_id"

        # полные таблицы взаимодействий с восстановленными референсными ячейками
        self.interaction_tables_ = {}
        for pair in self.interactions:
            a, b = pair
            table = self.coder.expand_interaction(beta_samples, pair) * scale[..., None]
            mean_table = table.mean(axis=0)  # (N, Ka, Kb)
            levels_a, levels_b = self.coder.levels_[a], self.coder.levels_[b]
            columns = [f"{a}={la} x {b}={lb}" for la in levels_a for lb in levels_b]
            flat = pd.DataFrame(
                mean_table.reshape(len(self.respondent_ids_), -1),
                index=self.respondent_ids_,
                columns=columns,
            )
            flat.index.name = "respondent_id"
            self.interaction_tables_[pair] = flat
            frame = pd.concat([frame, flat], axis=1)

        # Константа None-концепта: это полезность, но не уровень какого-либо
        # атрибута, поэтому она не входит ни в нулевую сумму, ни в важность
        if self.coder.include_none:
            none_utility = (beta_samples[..., self.coder.none_index_] * scale[..., 0]).mean(axis=0)
            frame[NONE_PARAM] = none_utility
            self.utilities_raw_[NONE_PARAM] = beta_samples[..., self.coder.none_index_].mean(axis=0)

        for i, label in enumerate(labels):
            frame[f"{label}_Importance"] = importance[:, i]
        frame["RLH"] = self._root_likelihood(beta_samples, x, response)
        frame["RLH_null"] = 1.0 / x.shape[2]
        self.individual_results_ = frame

        self.population_summary_ = self._build_population_summary(frame, labels)

    def _build_population_summary(self, frame: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
        rows = []
        for attr, levels in self.coder.levels_.items():
            for level in levels:
                column = f"{attr}={level}"
                rows.append(
                    {
                        "kind": "utility",
                        "attribute": attr,
                        "level": str(level),
                        "mean": frame[column].mean(),
                        "std": frame[column].std(),
                    }
                )
        for pair, table in self.interaction_tables_.items():
            for column in table.columns:
                rows.append(
                    {
                        "kind": "interaction",
                        "attribute": f"{pair[0]} x {pair[1]}",
                        "level": column,
                        "mean": frame[column].mean(),
                        "std": frame[column].std(),
                    }
                )
        if self.coder.include_none:
            rows.append(
                {
                    "kind": "none",
                    "attribute": NONE_PARAM,
                    "level": "",
                    "mean": frame[NONE_PARAM].mean(),
                    "std": frame[NONE_PARAM].std(),
                }
            )
        for label in labels:
            column = f"{label}_Importance"
            rows.append(
                {
                    "kind": "importance",
                    "attribute": label,
                    "level": "",
                    "mean": frame[column].mean(),
                    "std": frame[column].std(),
                }
            )
        rows.append(
            {"kind": "fit", "attribute": "RLH", "level": "",
             "mean": frame["RLH"].mean(), "std": frame["RLH"].std()}
        )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- прогноз

    def _check_fitted(self) -> None:
        if self.individual_results_ is None:
            raise RuntimeError("модель не оценена: сначала вызовите fit()")

    def predict_utilities(self, design_df: pd.DataFrame) -> np.ndarray:
        """Суммарная полезность каждой строки плана для её респондента."""
        self._check_fitted()
        unknown = set(design_df["respondent_id"].unique()) - set(self.respondent_index_)
        if unknown:
            raise ValueError(f"респонденты отсутствуют в оценённой модели: {sorted(unknown)}")
        x = self.coder.transform(design_df)
        rows = design_df["respondent_id"].map(self.respondent_index_).to_numpy(dtype=int)
        return (x * self.beta_mean_[rows]).sum(axis=1)

    def _merge_for_validation(
        self, design_df: pd.DataFrame, responses_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Свести план с ответами и посчитать предсказанные полезности."""
        self._check_fitted()
        column = self.resolve_response_column(responses_df)
        merged = design_df.merge(
            responses_df[[*ID_COLUMNS, column]].rename(columns={column: "response"}),
            on=list(ID_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        if merged["response"].isna().any():
            raise ValueError("для части строк плана нет ответа в responses_df")
        merged = merged.sort_values(list(ID_COLUMNS)).reset_index(drop=True)
        merged["utility"] = self.predict_utilities(merged)
        return merged

    def hit_rate(self, design_df: pd.DataFrame, responses_df: pd.DataFrame) -> float:
        """Доля задач, где предсказанная лучшая альтернатива совпала с фактической.

        Метрика инвариантна к масштабу полезностей, поэтому пригодна для
        валидации на отложенных (holdout) задачах. В режиме ``allocation``
        фактической считается альтернатива с наибольшей долей баллов —
        это first-choice hit rate; для аллокаций содержательнее
        :meth:`share_metrics`, поскольку hit rate игнорирует всё распределение
        баллов, кроме максимума.
        """
        merged = self._merge_for_validation(design_df, responses_df)
        grouped = merged.groupby(list(ID_COLUMNS[:2]), observed=True)
        predicted = grouped["utility"].idxmax()
        actual = grouped["response"].idxmax()
        return float((predicted.to_numpy() == actual.to_numpy()).mean())

    def share_metrics(self, design_df: pd.DataFrame, responses_df: pd.DataFrame) -> dict[str, float]:
        """Согласие предсказанных и фактических долей выбора.

        Основная метрика качества для режима ``allocation``: сравниваются доли
        внутри каждой задачи, а не только победившая альтернатива. В режиме
        ``single_choice`` тоже осмысленна — фактические доли там равны 0/1.
        """
        merged = self._merge_for_validation(design_df, responses_df)
        keys = list(ID_COLUMNS[:2])
        grouped = merged.groupby(keys, observed=True)

        shifted = merged["utility"] - grouped["utility"].transform("max")
        exponentiated = np.exp(shifted)
        merged["predicted_share"] = exponentiated / exponentiated.groupby(
            [merged[k] for k in keys]
        ).transform("sum")
        merged["actual_share"] = merged["response"] / grouped["response"].transform("sum")

        predicted = merged["predicted_share"].to_numpy()
        actual = merged["actual_share"].to_numpy()
        return {
            "mae": float(np.abs(predicted - actual).mean()),
            "rmse": float(np.sqrt(((predicted - actual) ** 2).mean())),
            "correlation": float(np.corrcoef(predicted, actual)[0, 1]),
        }

    # -------------------------------------------------- график итераций

    def plot_trace_history(
        self,
        var_names: Sequence[str] = ("alpha",),
        max_params: int = 12,
        path: str | None = None,
        figsize: tuple[float, float] | None = None,
        show: bool = False,
    ):
        """График истории итераций (Trace Plot) с выделенной зоной прогрева.

        Серая зона — итерации прогрева (burn-in), белая — финальные draws.
        Прогрев доступен потому, что :meth:`fit` вызывает ``pm.sample`` с
        ``discard_tuned_samples=False``; иначе PyMC выбрасывает tune-итерации и
        рисовать в серой зоне было бы нечего.
        """
        self._check_fitted()
        import matplotlib

        if path is not None and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        has_warmup = "warmup_posterior" in self.idata_.groups()
        if not has_warmup:
            warnings.warn(
                "прогрев не сохранён в idata — серая зона burn-in отображена не будет",
                stacklevel=2,
            )

        series: list[tuple[str, np.ndarray | None, np.ndarray]] = []
        for var in var_names:
            if var not in self.idata_.posterior:
                raise KeyError(f"переменной {var!r} нет в апостериорной выборке")
            post = self.idata_.posterior[var]
            warm = self.idata_.warmup_posterior[var] if has_warmup else None
            extra = [d for d in post.dims if d not in ("chain", "draw")]
            if not extra:
                series.append((var, None if warm is None else warm.to_numpy(), post.to_numpy()))
                continue
            post_flat = post.stack(_flat=extra)
            warm_flat = None if warm is None else warm.stack(_flat=extra)
            for i, key in enumerate(post_flat["_flat"].to_numpy()):
                name = "/".join(map(str, key)) if isinstance(key, tuple) else str(key)
                series.append(
                    (
                        f"{var}[{name}]",
                        None if warm_flat is None else warm_flat.isel(_flat=i).to_numpy(),
                        post_flat.isel(_flat=i).to_numpy(),
                    )
                )

        truncated = len(series) > max_params
        series = series[:max_params]
        n_warmup = 0 if series[0][1] is None else series[0][1].shape[1]

        figsize = figsize or (11.0, 1.6 * len(series) + 1.0)
        fig, axes = plt.subplots(len(series), 1, figsize=figsize, sharex=True, squeeze=False)
        axes = axes.ravel()

        for ax, (label, warm, post) in zip(axes, series):
            n_chains = post.shape[0]
            for chain in range(n_chains):
                values = post[chain] if warm is None else np.concatenate([warm[chain], post[chain]])
                ax.plot(np.arange(len(values)), values, linewidth=0.6, alpha=0.8,
                        label=f"цепь {chain + 1}" if ax is axes[0] else None)
            if n_warmup:
                ax.axvspan(0, n_warmup, color="0.85", zorder=0)
                ax.axvline(n_warmup, color="0.35", linestyle="--", linewidth=0.9)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=8)

        if n_warmup:
            axes[0].set_title(
                f"История итераций: серая зона — прогрев ({n_warmup} итераций), "
                f"белая — финальные draws ({series[0][2].shape[1]})",
                fontsize=10,
            )
        else:
            axes[0].set_title("История итераций (только финальные draws)", fontsize=10)
        axes[0].legend(fontsize=8, ncol=4, loc="upper right")
        axes[-1].set_xlabel("итерация")
        if truncated:
            fig.text(0.5, 0.005, f"показаны первые {max_params} параметров", ha="center", fontsize=8)
        fig.tight_layout()

        if path:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        if show:
            plt.show()
        return fig


# --------------------------------------------------------------------------
# Демонстрация: симуляция гетерогенных респондентов и восстановление их
# полезностей иерархической байесовской моделью.
# --------------------------------------------------------------------------

ATTRIBUTE_SPACE: dict[str, list[Any]] = {
    "Brand": ["Alpha", "Beta", "Gamma", "Delta"],
    "Price": [299, 399, 499, 599],
    "Storage": ["128GB", "256GB", "512GB"],
    "Warranty": ["1 year", "2 years", "3 years"],
}

# Сегменты с качественно разными предпочтениями: значения задаются по меткам
# уровней, а не по позициям, и центрируются при построении.
SEGMENTS: dict[str, dict[str, dict[Any, float]]] = {
    "price_sensitive": {
        "Brand": {"Alpha": 0.3, "Beta": 0.1, "Gamma": -0.1, "Delta": -0.3},
        "Price": {299: 2.4, 399: 0.9, 499: -0.9, 599: -2.4},
        "Storage": {"128GB": -0.5, "256GB": 0.1, "512GB": 0.4},
        "Warranty": {"1 year": -0.4, "2 years": 0.1, "3 years": 0.3},
    },
    "brand_loyal": {
        "Brand": {"Alpha": 2.2, "Beta": 0.4, "Gamma": -0.9, "Delta": -1.7},
        "Price": {299: 0.6, 399: 0.2, 499: -0.2, 599: -0.6},
        "Storage": {"128GB": -0.3, "256GB": 0.0, "512GB": 0.3},
        "Warranty": {"1 year": -0.2, "2 years": 0.0, "3 years": 0.2},
    },
    "balanced": {
        "Brand": {"Alpha": 1.0, "Beta": 0.3, "Gamma": -0.4, "Delta": -0.9},
        "Price": {299: 1.3, 399: 0.5, 499: -0.5, 599: -1.3},
        "Storage": {"128GB": -0.7, "256GB": 0.1, "512GB": 0.6},
        "Warranty": {"1 year": -0.6, "2 years": 0.1, "3 years": 0.5},
    },
}

# Сила истинного взаимодействия Brand x Price по сегментам: премиальный бренд
# менее чувствителен к росту цены, дешёвый — более.
INTERACTION_STRENGTH = {"price_sensitive": 0.35, "brand_loyal": 0.9, "balanced": 0.6}

# Истинная полезность отказа от покупки: чувствительные к цене отказываются
# охотнее, лояльные к бренду — реже.
NONE_UTILITY = {"price_sensitive": -0.6, "brand_loyal": -1.8, "balanced": -1.2}


def _true_interaction_table(coder: EffectsCoder, pair: tuple[str, str]) -> np.ndarray:
    """Базовая таблица взаимодействия с нулевыми суммами по строкам и столбцам."""
    a, b = pair
    levels_a, levels_b = coder.levels_[a], coder.levels_[b]
    premium = {"Alpha": 1.0, "Beta": 0.3, "Gamma": -0.4, "Delta": -0.9}
    brand = np.array([premium[l] for l in levels_a])
    price = np.array([float(l) for l in levels_b])
    price = (price - price.mean()) / price.std()
    raw = np.outer(brand, price)
    # двойное центрирование гарантирует нулевые суммы по обеим осям
    return raw - raw.mean(axis=0, keepdims=True) - raw.mean(axis=1, keepdims=True) + raw.mean()


def simulate_true_utilities(
    coder: EffectsCoder, n_respondents: int, rng: np.random.Generator, noise_sd: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """Истинные индивидуальные параметры и метки сегментов.

    Полезности строятся сразу в эффект-кодированном пространстве с нулевой
    суммой внутри атрибута — в том же виде, в каком их оценивает модель.
    """
    segment_names = list(SEGMENTS)
    segments = np.array([segment_names[i % len(segment_names)] for i in range(n_respondents)])
    beta = np.zeros((n_respondents, coder.n_params))
    base_table = {pair: _true_interaction_table(coder, pair) for pair in coder.interactions}

    coded_slices = coder.coded_slices_
    interaction_slices = coder.interaction_slices_

    for i, segment in enumerate(segments):
        for attr in coder.attribute_cols:
            levels = coder.levels_[attr]
            values = np.array([SEGMENTS[segment][attr][lvl] for lvl in levels], dtype=float)
            values = values + rng.normal(0.0, noise_sd, size=values.shape)
            values -= values.mean()  # нулевая сумма внутри атрибута
            beta[i, coded_slices[attr]] = values[:-1]
        for pair, table in base_table.items():
            strength = INTERACTION_STRENGTH[segment] * rng.normal(1.0, 0.25)
            scaled = table * strength
            ka = len(coder.levels_[pair[0]])
            kb = len(coder.levels_[pair[1]])
            # Gamma — левый верхний блок полной таблицы: остальные ячейки
            # восстанавливаются из ограничений нулевых сумм
            beta[i, interaction_slices[pair]] = scaled[: ka - 1, : kb - 1].reshape(-1)
        if coder.include_none:
            # отрицательная константа: отказ от покупки выбирают, но нечасто
            beta[i, coder.none_index_] = NONE_UTILITY[segment] + rng.normal(0.0, 0.4)
    return beta, segments


def _systematic_utility(
    design_df: pd.DataFrame, coder: EffectsCoder, beta_true: np.ndarray
) -> np.ndarray:
    x = coder.transform(design_df)
    index = {rid: i for i, rid in enumerate(pd.unique(design_df["respondent_id"]))}
    rows = design_df["respondent_id"].map(index).to_numpy(dtype=int)
    return (x * beta_true[rows]).sum(axis=1)


def simulate_choices(
    design_df: pd.DataFrame,
    coder: EffectsCoder,
    beta_true: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Ответы по модели MNL: аддитивная ошибка Гумбеля и выбор максимума."""
    utility = _systematic_utility(design_df, coder, beta_true)
    utility = utility + rng.gumbel(0.0, 1.0, size=len(design_df))

    responses = design_df[list(ID_COLUMNS)].copy()
    responses["utility"] = utility
    winners = responses.groupby(list(ID_COLUMNS[:2]), observed=True)["utility"].idxmax()
    responses["response"] = 0
    responses.loc[winners, "response"] = 1
    return responses.drop(columns="utility")


def simulate_allocations(
    design_df: pd.DataFrame,
    coder: EffectsCoder,
    beta_true: np.ndarray,
    rng: np.random.Generator,
    total: float = 100.0,
    concentration: float = 12.0,
) -> pd.DataFrame:
    """Распределение баллов между концептами.

    Ожидаемые доли равны логит-вероятностям выбора, а разброс вокруг них задаёт
    ``concentration``: чем оно выше, тем ближе ответ к теоретическим долям.
    Это ровно та модель, для которой предназначен режим ``allocation``, — одно
    суждение о пропорциях, а не серия независимых выборов.
    """
    utility = _systematic_utility(design_df, coder, beta_true)
    responses = design_df[list(ID_COLUMNS)].copy()
    responses["utility"] = utility

    keys = list(ID_COLUMNS[:2])
    grouped = responses.groupby(keys, observed=True)["utility"]
    exponentiated = np.exp(responses["utility"] - grouped.transform("max"))
    shares = exponentiated / exponentiated.groupby(
        [responses[k] for k in keys]
    ).transform("sum")

    values = np.empty(len(responses))
    for _, index in responses.groupby(keys, observed=True).indices.items():
        drawn = rng.dirichlet(concentration * shares.to_numpy()[index] + 1e-9)
        values[index] = drawn * total
    responses["response"] = values
    return responses.drop(columns="utility")


def _scale_normalised_mae(true: np.ndarray, estimated: np.ndarray) -> float:
    """MAE после приведения масштаба.

    Полезности MNL идентифицированы лишь с точностью до положительного
    множителя, поэтому сырой MAE измерял бы разницу масштабов, а не качество
    восстановления структуры предпочтений.
    """
    scale = float((true * estimated).sum() / (estimated * estimated).sum())
    return float(np.abs(true - scale * estimated).mean())


def _mean_within_respondent_correlation(true: np.ndarray, estimated: np.ndarray) -> float:
    values = [
        np.corrcoef(t, e)[0, 1]
        for t, e in zip(true, estimated)
        if t.std() > 1e-12 and e.std() > 1e-12
    ]
    return float(np.mean(values))


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(
    response_mode: str = "single_choice",
    include_none: bool = True,
    allocation_total: float = 100.0,
) -> None:
    n_respondents = 30
    concepts_per_task = 3
    tasks_total = 18
    holdout_tasks = 3
    interactions = [("Brand", "Price")]
    rng = np.random.default_rng(20240605)

    _rule("МОДУЛЬ 1. Генерация экспериментального плана CBC")
    generator = CBCDesignGenerator(
        ATTRIBUTE_SPACE,
        attribute_cols=list(ATTRIBUTE_SPACE),
        concepts_per_task=concepts_per_task,
        tasks_per_respondent=tasks_total,
        num_respondents=n_respondents,
        control_attributes="Brand",
        interactions=interactions,
        include_none=include_none,
        n_starts=25,
        random_state=20240605,
    )
    design = generator.generate()
    coder = generator.coder
    print(
        f"строк плана: {len(design)} | параметров: {coder.n_params} "
        f"(главных {coder.n_main} + None {coder.n_none} + "
        f"взаимодействий {coder.n_interaction})"
    )
    print(
        f"альтернатив в задаче: {concepts_per_task} профилей"
        + (" + None-концепт" if include_none else "")
    )
    print(
        f"D-эффективность по респондентам: средняя {generator.d_efficiency_:.4f}, "
        f"минимальная {generator.min_d_efficiency_:.4f}"
    )
    print(f"нарушений запрета на повтор Brand внутри задачи: {generator.check_prohibitions() or 0}")
    balance = generator.balance_report()
    print(f"максимальное отклонение частоты уровня от идеала: "
          f"{balance['deviation_pct'].abs().max():.2f}%")
    print("\nЗаполненность ячеек взаимодействия:")
    print(generator.interaction_report().to_string(index=False))

    _rule("Симуляция гетерогенных респондентов")
    beta_true, segments = simulate_true_utilities(coder, n_respondents, rng)
    if response_mode == "single_choice":
        responses = simulate_choices(design, coder, beta_true, rng)
    else:
        responses = simulate_allocations(design, coder, beta_true, rng, total=allocation_total)
    for name in SEGMENTS:
        print(f"  сегмент {name:<16} — {int((segments == name).sum())} респондентов")
    print(f"режим ответов: {response_mode}", end="")
    if response_mode == "allocation":
        print(f" (по {allocation_total:g} баллов на задачу)")
    else:
        print()

    fit_mask = design["task_id"] <= tasks_total - holdout_tasks
    design_fit, design_holdout = design[fit_mask], design[~fit_mask]
    responses_fit = responses[responses["task_id"] <= tasks_total - holdout_tasks]
    responses_holdout = responses[responses["task_id"] > tasks_total - holdout_tasks]
    print(f"\nзадач на оценку: {tasks_total - holdout_tasks}, отложено на holdout: {holdout_tasks}")

    _rule("МОДУЛЬ 2. Иерархическая байесовская оценка")
    estimator = CBCHierarchicalBayesEstimator(
        attribute_cols=list(ATTRIBUTE_SPACE),
        interactions=interactions,
        response_mode=response_mode,
        include_none=include_none,
        allocation_total=allocation_total if response_mode == "allocation" else None,
        # 20 параметров при 30 респондентах требуют более длинной адаптации,
        # чем ориентировочные 500/1000: на коротких цепях r_hat не сходится
        draws=1500,
        tune=1500,
        chains=4,
        upper_level_cov="auto",
        normalization="zcd",
        importance_mode="main",
        random_seed=20240605,
        progressbar=False,
    )
    estimator.fit(design_fit, responses_fit)

    diagnostics = estimator.diagnostics_
    print(f"структура ковариации верхнего уровня: {diagnostics['upper_level_cov']}")
    print(
        f"max r_hat = {diagnostics['max_r_hat']:.4f} | "
        f"min ESS = {diagnostics['min_ess_bulk']:.0f} | "
        f"расходящихся переходов: {diagnostics['divergences']}"
    )

    _rule("Индивидуальные полезности и важность атрибутов")
    results = estimator.individual_results_
    importance_cols = [c for c in results.columns if c.endswith("_Importance")]
    preview = [c for c in results.columns if c.startswith("Brand=") and " x " not in c]
    print(results[preview + importance_cols + ["RLH"]].head(6).round(2).to_string())
    print(f"\nсумма важностей по респонденту: "
          f"{results[importance_cols].sum(axis=1).min():.4f}..{results[importance_cols].sum(axis=1).max():.4f}")

    _rule("Сводка по популяции")
    summary = estimator.population_summary_
    print(summary[summary["kind"] != "interaction"].round(3).to_string(index=False))

    _rule("Валидация: восстановлены ли истинные параметры")
    true_full = coder.expand_main(beta_true)
    estimated_full = estimator.utilities_raw_.to_numpy()
    overall = float(np.corrcoef(true_full.ravel(), estimated_full.ravel())[0, 1])
    within = _mean_within_respondent_correlation(true_full, estimated_full)
    mae = _scale_normalised_mae(true_full, estimated_full)

    true_importance = np.stack(
        [
            (lambda r: 100.0 * r / r.sum())(
                np.array(
                    [
                        true_full[i, sl].max() - true_full[i, sl].min()
                        for sl in coder.utility_slices_.values()
                    ]
                )
            )
            for i in range(n_respondents)
        ]
    )
    estimated_importance = results[[f"{a}_Importance" for a in coder.attribute_cols]].to_numpy()
    importance_corr = float(
        np.corrcoef(true_importance.ravel(), estimated_importance.ravel())[0, 1]
    )

    true_interaction = coder.expand_interaction(beta_true, ("Brand", "Price"))
    estimated_interaction = coder.expand_interaction(estimator.beta_mean_, ("Brand", "Price"))
    interaction_corr = float(
        np.corrcoef(true_interaction.ravel(), estimated_interaction.ravel())[0, 1]
    )

    n_alternatives = concepts_per_task + (1 if include_none else 0)
    hit = estimator.hit_rate(design_holdout, responses_holdout)
    shares = estimator.share_metrics(design_holdout, responses_holdout)
    print(f"корреляция полезностей (все респонденты)    : {overall:.3f}")
    print(f"корреляция полезностей (средняя по людям)   : {within:.3f}")
    print(f"MAE после приведения масштаба               : {mae:.3f}")
    print(f"корреляция важностей атрибутов              : {importance_corr:.3f}")
    print(f"корреляция таблиц взаимодействия Brand*Price: {interaction_corr:.3f}")
    if include_none:
        true_none = beta_true[:, coder.none_index_]
        none_corr = float(np.corrcoef(true_none, estimator.utilities_raw_[NONE_PARAM])[0, 1])
        print(f"корреляция константы None-концепта          : {none_corr:.3f}")
    print(f"hit rate на {holdout_tasks} holdout-задачах             : {hit:.3f} "
          f"(случайный выбор — {1 / n_alternatives:.3f})")
    print(f"доли выбора на holdout: MAE {shares['mae']:.3f}, "
          f"корреляция {shares['correlation']:.3f}")
    print(f"средний RLH                                 : {results['RLH'].mean():.3f} "
          f"(случайный выбор — {1 / n_alternatives:.3f})")

    _rule("График истории итераций")
    path = f"cbc_hb_trace_{response_mode}.png"
    estimator.plot_trace_history(var_names=("alpha",), max_params=10, path=path)
    print(f"сохранён: {path}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single_choice"
    if mode not in CBCHierarchicalBayesEstimator._RESPONSE_MODES:
        raise SystemExit(
            f"использование: python cbc_hb.py "
            f"[{'|'.join(CBCHierarchicalBayesEstimator._RESPONSE_MODES)}]"
        )
    main(response_mode=mode)
