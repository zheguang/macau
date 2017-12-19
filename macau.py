#!/usr/bin/env python
'''
macau.py tests hypotheses statistically using resampling.
Copyright (C) 2017  Zheguang Zhao <zheguang.zhao@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import sys
sys.path.append(".")

import json
import pandas as pd
import numpy as np
import sys
import math
import re

# a tool to test simple hpyotheses against structured data using permutations.

alpha = 0.05

seed = 1 # TODO: turn this off in production run

class ObjectDict(dict):
    def __init__(self, json_str):
        self.json_str = json_str

        json_dict = json.loads(json_str)
        for k, v in json_dict.items():
            attr_k = '_'.join(k.strip().lower().split())
            self[attr_k] = v

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError('attribute {} not found'.format(attr))

    def __str__(self):
        return json.dumps({k: str(v) for k, v in self.__dict__.items()})


class Table(object):
    def __init__(self, df):
        self.df = df

    @staticmethod
    def fromFile(data_path):
        df = pd.read_csv(data_path)
        df.rename(columns=lambda x: x.split(':', 1)[0], inplace=True)
        return Table(df)

    def where(self, predicate=''):
        if len(predicate.strip()) == 0:
            return Table(self.df)
        else:
            return Table(self.df.query(predicate))

    def select(self, attrs=[]):
        assert type(attrs) is list
        return Table(self.df.loc[:, attrs])

    def permute(self, attr):
        """Permute a column in the table in place. """
        self.df[attr] = pd.Series(np.random.permutation(self.df[attr]), index=self.df.index)
        return self

    def copy(self):
        return Table(self.df.copy())

    def mean(self, attr):
        return np.average(self.df[attr])

    def histogram(self, attr, bucket_width, bucket_ref, bucket_agg):
        """Construct a histogram based on the table.

        The table has to be well-formed: contains only one column to histogramize.

        attr -- the sole column in the table to histogramize
        """
        #assert bucket_agg == 'count', 'only support count'
        #assert len(self.df.columns) == 1, 'should project down to one attr first, current schema size: {}'.format(len(self.df.columns))
        if bucket_agg == 'count':
            assert attr in self.df.columns
        elif bucket_agg.startswith('avg'):
            attr_agg = bucket_agg.split(',')[1].strip()
            assert attr_agg in self.df.columns, '{}, {}'.format(attr_agg, self.df.columns)
            assert attr in self.df.columns
        else:
            raise RuntimeError('unsupported bucket_agg')

        if bucket_ref == -1 and bucket_width == 1:
            # nominal attr
            bin_labels = sorted(set(self.df[attr]))
            bin_label_ids = dict(zip(bin_labels, range(len(bin_labels))))
            groups = ((self.df[attr].apply(lambda x: bin_label_ids[x]) - bucket_ref) / bucket_width).apply(math.floor) * bucket_width
        else:
            groups = ((self.df[attr] - bucket_ref) / bucket_width).apply(math.floor) * bucket_width
            bin_labels = list(map(lambda x: x + bucket_ref, sorted(set(groups))))
        self.df['_group'] = groups

        if bucket_agg == 'count':
            grouped = self.df.groupby('_group').count()
        elif bucket_agg.startswith('avg'):
            grouped = self.df.groupby('_group').mean()
        else:
            raise RuntimeError('unsupported bucket_agg')

        xs_lower = np.array(grouped.index)
        xs = list(map(lambda x: (x, x + bucket_width), xs_lower))
        if bucket_agg == 'count':
            ys = list(grouped[attr])
        elif bucket_agg.startswith('avg'):
            ys = list(grouped[attr_agg])
        else:
            raise RuntimeError('unsupported bucket_agg')
        return Histogram(xs, ys, bin_labels=bin_labels)

    def variance(self, attr):
        return self.covariance(attr, attr)

    def covariance(self, attr_0, attr_1):
        return np.average((self.df[attr_0] - self.mean(attr_0)) * (self.df[attr_1] - self.mean(attr_1)))

    def correlation_coefficient(self, attr_0, attr_1):
        return self.covariance(attr_0, attr_1) / math.sqrt(self.variance(attr_0) * self.variance(attr_1))

    def count(self):
        return len(self.df)


class Histogram(ObjectDict):
    def __init__(self, xs, ys, bin_labels=[]):
        """A horizontal histogram.
        xs is the buckets, and ys is the statistic in the bucket. A bucket is a (low, hi] interval.
        xs is a list of tuples, e.g. [(0, 9), (10, 19)]
        ys is a list of numbers, e.g. [13, 0]
        """
        self.xs = xs
        self.ys = ys
        self.bin_labels = bin_labels

    def distance(self, other):
        """Returns distance measure to the other histogram. The histograms are normalized to points in space."""
        assert type(other) is Histogram
        (ys_0, ys_1) = self.normalize_with(other)
        return np.linalg.norm(np.array(ys_0) - np.array(ys_1), 2)

    def extrema_distance(self, other, expect_extrema_labels, bucket_feature, bucket_junction):
        assert type(other) is Histogram
        assert len(expect_extrema_labels) > 0, 'should have observed exprema labels'
        assert set(expect_extrema_labels) < set(self.bin_labels), 'expect extrema labels have to exist in this histogram: {}, {}'.format(expect_extrema_labels, self.bin_labels)
        assert set(expect_extrema_labels) < set(other.bin_labels), 'expect extrema labels have to exist in other histogram'

        n_buckets = len(expect_extrema_labels)
        if bucket_feature == 'min':
            extrema = tuple(map(lambda x: x.sorted_labels()[:n_buckets], [self, other]))
        elif bucket_feature == 'max':
            extrema = tuple(map(lambda x: x.sorted_labels()[-n_buckets:], [self, other]))
        else:
            raise RuntimeError('unsupported bucket feature')

        if bucket_junction == 'all':
            d = 1 if set(extrema[0]) == set(extrema[1]) else 0
        elif bucket_junction == 'either':
            d = 1 if set(extrema[0]) & set(extrema[1]) != set() else 0
        elif bucket_junction == 'rank':
            d = 1 if extrema[0] == extrema[1] else 0
        else:
            raise RuntimeError('unsupported bucket junction')

        return d

    def normalize_with(self, other):
        assert type(other) is Histogram

        # union, sort by left boundary
        xs_norm = sorted(set(self.xs) | set(other.xs), key=lambda x: x[0])

        # check illforms: invalid interval, overlapped, etc
        for i in range(len(xs_norm) - 1):
            assert xs_norm[i][0] < xs_norm[i][1], 'invalid interval'
            assert xs_norm[i][1] <= xs_norm[i + 1][0], 'overlapped left bound'
            assert xs_norm[i][1] < xs_norm[i + 1][1], 'overlapped right bound'

        return (self.ys_norm(xs_norm), other.ys_norm(xs_norm))

    def ys_norm(self, xs_norm):
        xys = dict(zip(self.xs, self.ys))
        return list(map(lambda x: xys[x] if x in xys else 0, xs_norm))

    def sorted(self):
        return sorted(self.ys)

    def sorted_labels(self):
        return list(map(lambda x: x[1], sorted(zip(self.ys, self.bin_labels), key=lambda x: x[0])))

    @staticmethod
    def uniform_bernoulli(xs, bin_labels, n_trials):
        ys = np.zeros(len(xs))
        for b in map(lambda _: math.floor(np.random.rand() * len(xs)), range(n_trials)):
            ys[b] += 1

        return Histogram(xs, ys, bin_labels)

    @staticmethod
    def uniform_exact(xs, bin_labels, n_trials):
        ys = np.ones(len(xs)) / float(n_trials)
        return Histogram(xs, ys, bin_labels)

    def copy(self):
        return Histogram(self.xs[:], self.ys[:], bin_labels=self.bin_labels[:])


class Resampler(object):
    def __init__(self, src_table, discovery, n_perms, exec_mode):
        self.src_table = src_table
        self.d = discovery
        self.n_perms = n_perms
        self.perm_table = src_table.copy()
        self.exec_mode = exec_mode

        if seed != None:
            np.random.seed(seed)

    def test_on_same_variance(self, one_sided=None):
        # None {'dist_null': '', 'test': 'variance_smaller', 'prediction': 'positive', 'dist_alt': 'work_per_week >=60 and work_per_week < 120 and stress_level >= 3 and stress_level < 6', 'dimension': 'hours_of_sleep'}

        def perm(table):
            return table.permute(self.d.dimension)

        def t(table):
            diff = table.where(self.d.dist_alt).variance(self.d.dimension) - table.where(self.d.dist_null).variance(self.d.dimension)
            if one_sided == None:
                return abs(diff)
            elif one_sided == 'higher':
                return diff
            elif one_sided == 'smaller':
                return -diff
            else:
                raise RuntimeError('unsupported one_sided argument: {}'.format(one_sided))

        p_est = self.mc_kernel(t, perm)
        return p_est


    def test_on_same_mean(self, one_sided=None):
        def perm(table):
            return table.permute(self.d.dimension)

        def t(table):
            diff = table.where(self.d.dist_alt).mean(self.d.dimension) - table.where(self.d.dist_null).mean(self.d.dimension)
            if one_sided == None:
                return abs(diff)
            elif one_sided == 'higher':
                return diff
            elif one_sided == 'smaller':
                return -diff
            else:
                raise RuntimeError('unsupported one_sided argument: {}'.format(one_sided))

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_2d_same_mean(self, one_sided=None):
        attrs = [x.strip() for x in self.d.dimension.split(',')]
        assert len(attrs) == 2, 'two dimensions only'

        def perm(table):
            concat_table = Table(pd.DataFrame(data=table.df[attrs[0]].append(table.df[attrs[1]]), columns=[self.d.dimension]))
            concat_table.permute(self.d.dimension)
            table.df[attrs[0]] = concat_table.df[self.d.dimension].iloc[np.arange(int(len(concat_table.df) / 2))]
            table.df[attrs[1]] = concat_table.df[self.d.dimension].iloc[np.arange(int(len(concat_table.df) / 2), int(len(concat_table.df)))]
            return table

        def t(table):
            assert 'filter' not in self.d, 'filter not supported'
            diff = table.mean(attrs[0]) - table.mean(attrs[1])
            if one_sided == None:
                return abs(diff)
            elif one_sided == 'higher':
                return diff
            elif one_sided == 'smaller':
                return -diff
            else:
                raise RuntimeError('unsupported one_sided argument: {}'.format(one_sided))

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_same_shape(self):
        def perm(table):
            return table.permute(self.d.dimension)

        def t(table):
            (hist_0, hist_1) = map(lambda x: table.where(x).select([self.d.dimension]).histogram(self.d.dimension, self.d.bucket_width, self.d.bucket_ref, self.d.bucket_agg), [self.d.dist_null, self.d.dist_alt])
            return hist_0.distance(hist_1)

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_independence(self):
        attrs = list(map(lambda x: x.strip(), self.d.dimension.split(',')))
        assert len(attrs) == 2, 'only two attributes'

        # fix attr[0], permute attr[1], (or vice versa)
        # for each permutation, calculate the empirical correlation coefficient as t
        # use mc_kernel to approxiamte Pr(t(perm) >= t(obs; H0), where H0 corresponds to rho = 0.
        def perm(table):
            return table.permute(attrs[1])

        def t(table):
            rho = self.perm_table.correlation_coefficient(attrs[0], attrs[1])
            return abs(rho)

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_uniform_shape(self):
        #'{"dimension":"ad_campaign_a", "bucket_width":1, "bucket_ref":0, "bucket_agg":"count", "filter":"", "target_buckets":"ad_campaign_a == 1, ad_campaign_a == 0", "test":"buckets_same"}'
        target_buckets = [x for x in map(lambda x: x.strip(), self.d.target_buckets.split(',')) if len(x) > 0]
        target_buckets_filter = ' or '.join(target_buckets)

        def hist(table):
            return table.where(self.d.filter).where(target_buckets_filter).select([self.d.dimension]).histogram(self.d.dimension, self.d.bucket_width, self.d.bucket_ref, self.d.bucket_agg)

        def perm(table):
            return table

        src_hist = hist(self.src_table)
        uniform_exact_h = Histogram.uniform_exact(src_hist.xs, src_hist.bin_labels, sum(src_hist.ys))

        def t(table):
            if table is self.src_table:
                t = src_hist.distance(uniform_exact_h)
            else:
                bernoulli_hist = src_hist.uniform_bernoulli(src_hist.xs, src_hist.bin_labels, sum(src_hist.ys))
                t = bernoulli_hist.distance(uniform_exact_h)
            #print(t)
            return t

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_uniform_buckets(self, bucket_feature='', bucket_junction=''):
        # {'bucket_ref': -1, 'test': 'min_bucket_either', 'target_buckets': "region=='Northeast'", 'prediction': 'positive', 'dimension': 'region', 'filter': '', 'bucket_agg': 'count', 'bucket_width': 1}
        # {'bucket_ref': 15, 'test': 'max_bucket_either', 'target_buckets': 'age >= 30 and age < 35', 'prediction': 'positive', 'dimension': 'age', 'filter': '', 'bucket_agg': 'count', 'bucket_width': 5}

        # NOTE: hack: table is historgram
        self.src_table = self.src_table.where(self.d.filter).select([self.d.dimension]).histogram(self.d.dimension, self.d.bucket_width, self.d.bucket_ref, self.d.bucket_agg)
        self.perm_table = self.src_table.copy()

        def perm(hist):
            bernoulli_hist = hist.uniform_bernoulli(self.src_table.xs, self.src_table.bin_labels, sum(self.src_table.ys))
            return bernoulli_hist

        def t(hist):
            # extract to labels, for debug purpose
            if self.d.bucket_ref == -1 and self.d.bucket_width == 1:
                # nominal attr
                extrema_labels = re.search(r"==\s*'([\w\d\s]+)'", self.d.target_buckets).groups()
            else:
                # numerical attr
                extrema_labels = tuple(map(float, re.search(r">=\s*([\d.]+)", self.d.target_buckets).groups()))

            t = hist.extrema_distance(self.src_table, extrema_labels, bucket_feature, bucket_junction)
            return t

        assert t(self.src_table) == 1, 'histogram has to contain user observed extrema'

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_aggregate_buckets(self, bucket_feature='', bucket_junction=''):
        #'{"dimension":"age", "bucket_width":5, "bucket_ref":15, "bucket_agg":"avg,hours_of_sleep", "filter":"", "target_buckets":"age >= 40 and age < 50", "test":"min_bucket_either"}',
        assert self.d.bucket_agg.startswith('avg'), 'only avg is supported'

        def hist(table):
            return table.where(self.d.filter).histogram(self.d.dimension, self.d.bucket_width, self.d.bucket_ref, self.d.bucket_agg)

        src_hist = hist(self.src_table)

        def perm(table):
            return table.where(self.d.filter).permute(self.d.dimension)

        def t(table):
            # extract to labels, for debug purpose
            if self.d.bucket_ref == -1 and self.d.bucket_width == 1:
                # nominal attr
                extrema_labels = re.search(r"==\s*'([\w\d\s]+)'", self.d.target_buckets).groups()
            else:
                # numerical attr
                extrema_labels = tuple(map(float, re.search(r">=\s*([\d.]+)", self.d.target_buckets).groups()))

            t = hist(table).extrema_distance(src_hist, extrema_labels, bucket_feature, bucket_junction)
            return t

        assert t(self.src_table) == 1, 'histogram has to contain user observed extrema'

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_uniform_rank_buckets(self):
        # '{"filter": "", "test": "rank_buckets_count", "target_buckets": "age >= 35 and age < 55, age >=15 and age < 35, age >= 55 and age <= 75", "prediction": "positive", "dimension": "age"}',
        target_buckets = [x.strip() for x in self.d.target_buckets.split(',')]
        target_buckets_filter = ' or '.join(target_buckets)

        def rank_buckets(table):
            return list(map(lambda x: table.where(x).select([self.d.dimension]).count(), target_buckets))

        src_rank_buckets = rank_buckets(self.src_table)
        assert len(src_rank_buckets) == len(target_buckets)
        assert sorted(src_rank_buckets, reverse=True) == src_rank_buckets, 'user observation has to be true: {}'.format(src_rank_buckets)

        def perm(table):
            return table

        def t(table):
            if table is self.src_table:
                assert sorted(src_rank_buckets, reverse=True) == src_rank_buckets, 'user observation has to be true'
                return 1
            else:
                perm_rank_buckets = [0] * len(target_buckets)
                for x in map(lambda _: np.random.choice(np.arange(len(target_buckets))), table.where(target_buckets_filter).df[self.d.dimension]):
                    perm_rank_buckets[x] += 1

                return 1 if sorted(perm_rank_buckets, reverse=True) == perm_rank_buckets else 0

        assert t(self.src_table) == 1, 'histogram has to contain user observed extrema'

        p_est = self.mc_kernel(t, perm)
        return p_est

    def test_on_same_subpopulation(self, bucket_feature='', bucket_junction=''):
        #'{"dimension":"age", "bucket_width":5, "bucket_ref":15, "bucket_agg":"count", "filter":"ad_campaign_a==1", "target_buckets":"age>=15 and age < 20", "test":"min_bucket_either"}',
        buckets = [x.strip() for x in self.d.target_buckets.split(',')]
        assert len(buckets) > 0, 'should have at least one target buckets'

        def perm(table):
            return table.permute(self.d.dimension)

        def satisfied(hist, counts):
            if bucket_feature == 'min':
                extreme_buckets = hist.sorted()[:len(buckets)]
            elif bucket_feature == 'max':
                extreme_buckets = hist.sorted()[-len(buckets):]
            else:
                raise RuntimeError('unsupported bucket feature: {}'.format(bucket_feature))

            if bucket_junction == 'either':
                satisfied = len([x for x in counts if x in extreme_buckets]) > 0
            elif bucket_junction == 'all':
                satisfied = len([x for x in counts if x in extreme_buckets]) == len(buckets)
            else:
                raise RuntimeError('unsupported bucket junction: {}'.format(bucket_junction))

            return satisfied

        def t(table):
            hist = table.where(self.d.filter).select([self.d.dimension]).histogram(self.d.dimension, self.d.bucket_width, self.d.bucket_ref, self.d.bucket_agg)
            counts = [table.where(self.d.filter).where(x).count() for x in buckets]
            #print(hist)
            #print(counts)
            return satisfied(hist, counts)

        def run_test():
            p_est = self.mc_kernel(t, perm)

            global_hist = self.src_table.select([self.d.dimension]).histogram(self.d.dimension, self.d.bucket_width, self.d.bucket_ref, self.d.bucket_agg)
            global_counts = [self.src_table.where(x).count() for x in buckets]
            global_sat = satisfied(global_hist, global_counts)
            if global_sat == True:
                # user is betting on h0: consistent subpopulation
                sig = p_est > alpha
                flipped = True
            else:
                # fine, user is betting on h1: inconsistent subpopulation
                sig = p_est <= alpha
                flipped = False

            return (p_est, sig, flipped)

        sub_sat = t(self.src_table)
        sub_sat_msg = 'user observation has to be true on the dataset'
        if self.exec_mode == 'test':
            assert sub_sat == True, sub_sat_msg
            res = run_test()
        else:
            assert self.exec_mode == 'truth'
            if sub_sat == False:
                print('[warn] {}'.format(sub_sat_msg))
                res = (1, False, False)
            else:
                res = run_test()

        return res

    def test_one_sample_mean(self, interval):
        # {'filter': 'work_per_week >=0 and work_per_week < 40', 'test': 'mean == 7.5', 'prediction': 'positive', 'dimension': 'hours_of_sleep'}
        # reference: http://stats.stackexchange.com/questions/65831/permutation-test-comparing-a-single-sample-against-a-mean#171748
        #mean_hat = float(re.search(r'mean == (\d*.?\d+)', self.d.test).groups()[0])
        assert len(interval) == 2 and interval[0] <= interval[1]
        mean_hat = float(sum(interval)) / 2.0
        half_width = abs(float(interval[1] - interval[0]) / 2.0)
        # normalize to zero mean
        self.src_table = Table(pd.DataFrame(data=self.src_table.where(self.d.filter).df[self.d.dimension] - mean_hat, columns=[self.d.dimension]))
        self.perm_table = self.src_table.copy()

        def perm(table):
            return Table(pd.DataFrame(data=list(map(lambda x: np.random.choice([-1, 1]) * abs(x), table.df[self.d.dimension])), columns=[self.d.dimension]))

        def t(table):
            if table is self.src_table:
                return abs(table.mean(self.d.dimension))
            else:
                return abs(table.mean(self.d.dimension)) + half_width

        p_est = self.mc_kernel(t, perm)
        return p_est

    def mc_kernel(self, t, perm):
        """Use Monte-Carlo simulation to estimate the p-value, i.e. Pr(t(perm) >= t(obs); H0), where Pr(t; H0) is obtained by permutation.

        t    -- a function that takes a table and returns a staitstic
        perm -- a function that takes a table and permutes it in place
        """
        def poisson_trial(t_obs):
            #self.perm_table.permute(self.d.dimension)
            t_perm = t(perm(self.perm_table))
            #print(t_obs, t_perm)
            return 1 if t_perm >= t_obs else 0

        # get the fraction of t_perm >= t_obs as p-value
        t_obs = t(self.src_table)
        poisson_trials = list(map(lambda _: poisson_trial(t_obs), range(int(self.n_perms))))
        #print(poisson_trials)
        assert len(poisson_trials) == self.n_perms, '{},{}'.format(len(poisson_trials), self.n_perms)
        p_est = sum(poisson_trials) / float(self.n_perms)

        return p_est


class Discovery(ObjectDict):
    def __init__(self, json_str, table, exec_mode):
        super().__init__(json_str)
        self.table = table
        self.exec_mode = exec_mode

    def hypothesis_test(self, n_perms):
        """Returns a HypothesisTestResult."""

        resampler = Resampler(self.table, self, n_perms, self.exec_mode)
        flipped = False

        #
        # variance
        #
        if self.test == 'variance_smaller':
            p_est = resampler.test_on_same_variance(one_sided='smaller')
            sig = p_est <= alpha

        #
        # mean
        #
        elif self.test == 'mean_higher':
            p_est = resampler.test_on_same_mean(one_sided='higher')
            sig = p_est <= alpha

        elif self.test == 'mean_smaller':
            p_est = resampler.test_on_same_mean(one_sided='smaller')
            sig = p_est <= alpha

        elif self.test == 'mean_same':
            flipped = True
            # flip the decision rule to indicate the user is betting on h0
            p_est = resampler.test_on_same_mean()
            sig = p_est > alpha

        elif self.test == 'mean_different':
            p_est = resampler.test_on_same_mean()
            sig = p_est <= alpha

        #
        # 2d mean
        #
        elif self.test == '2d_mean_higher':
            p_est = resampler.test_on_2d_same_mean(one_sided='higher')
            sig = p_est <= alpha

        elif self.test == '2d_mean_smaller':
            p_est = resampler.test_on_2d_same_mean(one_sided='smaller')
            sig = p_est <= alpha

        elif self.test == '2d_mean_same':
            flipped = True
            # flip the decision rule to indicate the user is betting on h0
            p_est = resampler.test_on_2d_same_mean()
            sig = p_est > alpha

        elif self.test == '2d_mean_different':
            p_est = resampler.test_on_2d_same_mean()
            sig = p_est <= alpha

        #
        # shape
        #
        elif self.test == 'shape_same':
            flipped = True
            # flip the decision rule to indicate the user is betting on h0
            p_est = resampler.test_on_same_shape()
            sig = p_est > alpha

        elif self.test == 'shape_different':
            p_est = resampler.test_on_same_shape()
            sig = p_est <= alpha

        elif self.test == 'buckets_same':
            flipped = True
            # flip the decision rule to indicate the user is betting on h0
            p_est = resampler.test_on_uniform_shape()
            sig = p_est > alpha

        elif self.test == 'buckets_different':
            p_est = resampler.test_on_uniform_shape()
            sig = p_est <= alpha

        elif self.test == 'min_bucket_either':
            if self.bucket_agg == 'count':
                p_est = resampler.test_on_uniform_buckets(bucket_feature='min', bucket_junction='either')
            elif self.bucket_agg.startswith('avg'):
                p_est = resampler.test_on_aggregate_buckets(bucket_feature='min', bucket_junction='either')
            else:
                raise RuntimeError('unsupported')
            sig = p_est <= alpha

        elif self.test == 'max_bucket_either':
            if self.bucket_agg == 'count':
                p_est = resampler.test_on_uniform_buckets(bucket_feature='max', bucket_junction='either')
            elif self.bucket_agg.startswith('avg'):
                p_est = resampler.test_on_aggregate_buckets(bucket_feature='max', bucket_junction='either')
            else:
                raise RuntimeError('unsupported')
            sig = p_est <= alpha

        elif self.test == 'min_bucket_all':
            if self.bucket_agg == 'count':
                p_est = resampler.test_on_uniform_buckets(bucket_feature='min', bucket_junction='all')
            elif self.bucket_agg.startswith('avg'):
                p_est = resampler.test_on_aggregate_buckets(bucket_feature='min', bucket_junction='all')
            else:
                raise RuntimeError('unsupported')
            sig = p_est <= alpha

        elif self.test == 'max_bucket_all':
            if self.bucket_agg == 'count':
                p_est = resampler.test_on_uniform_buckets(bucket_feature='max', bucket_junction='all')
            elif self.bucket_agg.startswith('avg'):
                p_est = resampler.test_on_aggregate_buckets(bucket_feature='max', bucket_junction='all')
            else:
                raise RuntimeError('unsupported')
            sig = p_est <= alpha

        elif self.test == 'rank_buckets_count':
            p_est = resampler.test_on_uniform_rank_buckets()
            sig = p_est <= alpha

        #
        # correlation
        #
        elif self.test == 'not_corr':
            flipped = True
            # flip the decision rule to indicate the user is betting on h0
            p_est = resampler.test_on_independence()
            sig = p_est > alpha

        elif self.test == 'corr':
            p_est = resampler.test_on_independence()
            sig = p_est <= alpha

        #
        # one sample mean
        #
        elif self.test.startswith('mean =='):
            # flip the decision rule to indicate the user is betting on h0
            flipped = True
            mean = float(re.search(r'mean == (\d*.?\d+)', self.test).groups()[0])
            p_est = resampler.test_one_sample_mean([mean, mean])
            sig = p_est > alpha

        elif self.test.startswith('mean >='):
            # flip the decision rule to indicate the user is betting on h0
            flipped = True
            interval = list(map(float, re.search(r'mean >= (\d*.?\d+) and mean < (\d*.?\d+)', self.test).groups()))
            p_est = resampler.test_one_sample_mean(interval)
            sig = p_est > alpha

        #
        # Unsupported
        #
        else:
            p_est = None
            sig = None
            flipped = None

        return HypothesisTestResult(p_est, alpha, sig, flipped) if p_est != None else HypothesisTestResult.unsupported_test()


class HypothesisTestResult(ObjectDict):
    def __init__(self, p_value, alpha, sig, flipped):
        self.p_value = p_value
        self.alpha = alpha
        self.sig = sig
        self.flipped = flipped

    @staticmethod
    def unsupported_test():
        return HypothesisTestResult(None, None, 'unsupported test', None)


def main(argv):
    if argv[1] == '-h':
        print('usage: python3 macau.py <exec_mode> <data_path> <permutations> <hypotheses_path>')
    else:
        exec_mode = argv[1]
        assert exec_mode in ['test', 'truth'], 'exec_mode should be either "test" or "truth"'
        data_path = argv[2]
        n_perms = float(argv[3])
        hypotheses_path = argv[4]

        #lines = [
        #    '{"dimension":"purchases", "dist_alt":"age>=15 and age < 20", "dist_null":"", "test":"mean_higher"}',
        #    '{"dimension":"purchases", "dist_alt":"age>=15 and age < 20", "dist_null":"", "test":"mean_smaller"}',
        #    '{"dimension":"purchases", "dist_alt":"age>=15 and age < 20", "dist_null":"", "test":"mean_same"}',
        #    '{"dimension":"purchases", "dist_alt":"age>=15 and age < 20", "dist_null":"", "test":"mean_different"}',

        #    '{"dimension":"ad_campaign_a,ad_campaign_b", "test":"2d_mean_higher"}',
        #    '{"dimension":"ad_campaign_a,ad_campaign_b", "test":"2d_mean_smaller"}',
        #    '{"dimension":"ad_campaign_a,ad_campaign_b", "test":"2d_mean_same"}',
        #    '{"dimension":"ad_campaign_a,ad_campaign_b", "test":"2d_mean_different"}',

        #    '{"dimension":"min_on_site", "bucket_width":10, "bucket_ref":0, "bucket_agg":"count", "dist_alt":"design == \'blue\'", "dist_null":"", "test":"shape_same"}',
        #    '{"dimension":"min_on_site", "bucket_width":10, "bucket_ref":0, "bucket_agg":"count", "dist_alt":"design == \'blue\'", "dist_null":"", "test":"shape_different"}',
        #    '{"dimension":"region", "bucket_width":1, "bucket_ref":-1, "bucket_agg":"count", "dist_alt":"ad_campaign_a==1", "dist_null":"", "test":"shape_same"}',

        #    '{"dimension":"ad_campaign_b,age", "test":"not_corr"}',
        #    '{"dimension":"nr_of_visits,purchases", "test":"not_corr"}',
        #    '{"dimension":"nr_of_visits,purchases", "test":"corr"}',

        #    '{"dimension":"nr_of_visits", "bucket_width":2, "bucket_ref":0, "bucket_agg":"count", "filter":"age>=35 and age < 40", "target_buckets":"nr_of_visits >= 6 and nr_of_visits < 8", "test":"max_bucket_either"}',
        #    '{"dimension":"nr_of_visits", "bucket_width":2, "bucket_ref":0, "bucket_agg":"count", "filter":"age>=25 and age < 30", "target_buckets":"nr_of_visits >= 6 and nr_of_visits < 8", "test":"max_bucket_either"}',
        #    '{"dimension":"age", "bucket_width":5, "bucket_ref":15, "bucket_agg":"avg,hours_of_sleep", "filter":"", "target_buckets":"age >= 40 and age < 50", "test":"min_bucket_either"}',

        #    '{"dimension":"mobile", "filter":"income > 200000", "test":"mean == 7.5"}',
        #    '{"dimension": "income", "bucket_width": 20000, "bucket_ref": 0, "bucket_agg": "avg,purchase_amount", "filter": "", "target_buckets": "", "test": "buckets_different"}',

        #    '{"dimension":"age", "filter":"", "target_buckets":"age >= 35 and age < 55, age >=15 and age < 35, age >= 55 and age <= 75", "test":"rank_buckets_count"}',

        #    '{"dimension":"ad_campaign_a", "bucket_width":1, "bucket_ref":0, "bucket_agg":"count", "filter":"", "target_buckets":"ad_campaign_a == 1, ad_campaign_a == 0", "test":"buckets_same"}',

        #    '{"dimension":"hours_of_sleep", "dist_alt":"work_per_week >=60 and work_per_week < 120 and stress_level >= 3 and stress_level < 6", "dist_null":"", "test":"variance_smaller"}',

        #    '{"dimension":"ad_campaign_a", "bucket_width":1, "bucket_ref":0, "bucket_agg":"count", "filter":"", "target_buckets":"ad_campaign_a == 1, ad_campaign_a == 0", "test":"buckets_same"}',
        #    '{"dimension":"ad_campaign_a", "bucket_width":1, "bucket_ref":0, "bucket_agg":"count", "filter":"", "target_buckets":"ad_campaign_a == 1, ad_campaign_a == 0", "test":"buckets_different"}',

        #    '{"dimension":"ad_campaign_b", "dist_alt":"region == \'Midwest\' or region == \'West\'", "dist_null":"", "test":"mean_higher"}',

        #    '{"filter": "work_per_week >=0 and work_per_week < 40", "test": "mean == 7.5", "prediction": "positive", "dimension": "hours_of_sleep"}',
        #    '{"filter": "stress_level >= 2 and stress_level < 3", "dimension": "hours_of_sleep", "prediction": "positive", "test": "mean >= 5 and mean < 10"}',
        #    '{"filter": "stress_level >= 2 and stress_level < 3", "test": "mean >= 5 and mean < 10", "prediction": "positive", "dimension": "hours_of_sleep"}',

        #    '{"bucket_ref": -1, "test": "min_bucket_either", "target_buckets": "region==\'Northeast\'", "prediction": "positive", "dimension": "region", "filter": "", "bucket_agg": "count", "bucket_width": 1}',
        #    '{"bucket_ref": 15, "test": "max_bucket_either", "target_buckets": "age >= 30 and age < 35", "prediction": "positive", "dimension": "age", "filter": "", "bucket_agg": "count", "bucket_width": 5}',

        #    '{"dist_null": "", "test": "variance_smaller", "prediction": "positive", "dist_alt": "work_per_week >=60 and work_per_week < 120 and stress_level >= 3 and stress_level < 6", "dimension": "hours_of_sleep"}',

        #    '{"filter": "", "test": "rank_buckets_count", "target_buckets": "age >= 35 and age < 55, age >=15 and age < 35, age >= 55 and age <= 75", "prediction": "positive", "dimension": "age"}',
        #    '{"dimension": "hours_of_sleep", "filter": "gender == \'female\'", "test": "mean == 7.8", "prediction": "positive"}'

        #    '{"dimension": "purchase_amount", "filter":"", "test":"mean >= 110 and mean < 111"}',
        #]
        with open(hypotheses_path) as lines:
            table = Table.fromFile(data_path)

            for l in lines:
                d = Discovery(l, table, exec_mode)
                res = d.hypothesis_test(n_perms)

                print('{}\n{}\n'.format(l, res))

        print('done')


if __name__ == '__main__':
    main(sys.argv)


