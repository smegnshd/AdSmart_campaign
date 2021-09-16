# -*- coding: utf-8 -*-
import pandas as pd
from numpy import std, mean
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


def get_conversion_rates(df: pd.DataFrame, grouping_column: str, calculation_column: str) -> pd.DataFrame:

    # Grouping the dataframe by the grouping column and getting calculation column
    conversion_rates = df.groupby(grouping_column)[calculation_column]

    # defining methods to generate the standard deviation and standard error of mean
    def std_p(x): return std(x, ddof=0)
    def se_p(x): return stats.sem(x, ddof=0)

    # creating the conversion datagrame
    conversion_rates = conversion_rates.agg([mean, std_p, se_p])
    conversion_rates.columns = [
        'conversion_rate', 'std_deviation', 'std_error']
    return conversion_rates


def get_group_result(df: pd.DataFrame, from_column: str, val_type: str, value_column: str) -> pd.Series:
    # Retrieve the group results
    group_results = df[df[from_column] == val_type][value_column]
    return group_results


def get_count(group_result: pd.Series) -> int:
    return group_result.count()


def get_sum(group_result: pd.Series) -> int:
    return group_result.sum()


def form_success(control_group_result, treatment_group_result) -> list:
    control_sum = get_sum(control_group_result)
    treatment_sum = get_sum(treatment_group_result)
    return [control_sum, treatment_sum]


def form_noob(control_group_result, treatment_group_result) -> list:
    control_count = get_count(control_group_result)
    treatment_count = get_count(treatment_group_result)
    return [control_count, treatment_count]


def run_ztest(successes: list, nobs: list) -> tuple:
    z_stat, pval = proportions_ztest(successes, nobs=nobs)
    return (z_stat, pval)


def get_lower_upper_bounds(successes: list, nobs: list, alpha: float = 0.05) -> list:
    # Getting the lower and upper bounds
    (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(
        successes, nobs=nobs, alpha=alpha)

    return [(lower_con, lower_treat), (upper_con, upper_treat)]


def print_results(ztest_result: tuple, lower_upper_bound_result: list) -> None:
    # Printing calculated results
    print(f'z-statistic: {ztest_result[0]:.3f}')
    print(f'p-value: {ztest_result[1]:.3f}')
    (lower_con, lower_treat) = lower_upper_bound_result[0]
    (upper_con, upper_treat) = lower_upper_bound_result[1]
    print(f'ci 95% for control group: [{lower_con:.2%},{upper_con:.2%}]')
    print(f'ci 95% for treatment group: [{lower_treat:.2%},{upper_treat:.2%}]')


