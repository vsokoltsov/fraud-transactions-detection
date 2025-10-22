import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from api.db.predictions import (
    generate_features,
    tx_amount_columns,
    time_since_prev_trx,
    burst_deviation,
    rs_anomaly,
    hour_deviates,
    robust_amount_outlier,
    day_of_week_zscore_outlier,
    fraud_burst_candidate,
    EPS
)


class TestPredictions:
    """Test suite for prediction feature engineering functions."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'customer_id': [1001, 1001, 1001, 1002, 1002],
            'tx_datetime': pd.to_datetime([
                '2023-01-01 10:00:00',
                '2023-01-01 11:00:00', 
                '2023-01-01 12:00:00',
                '2023-01-01 13:00:00',
                '2023-01-01 14:00:00'
            ]),
            'tx_amount': [100.0, 200.0, 150.0, 300.0, 250.0],
            'tx_fraud': [0, 0, 1, 0, 0]
        })

    @pytest.fixture
    def single_customer_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with single customer for testing."""
        return pd.DataFrame({
            'customer_id': [1001, 1001, 1001],
            'tx_datetime': pd.to_datetime([
                '2023-01-01 10:00:00',
                '2023-01-01 11:00:00',
                '2023-01-01 12:00:00'
            ]),
            'tx_amount': [100.0, 200.0, 150.0],
            'tx_fraud': [0, 0, 1]
        })

    def test_generate_features_calls_all_functions(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that generate_features calls all feature engineering functions."""
        with patch('api.db.predictions.tx_amount_columns') as mock_tx, \
             patch('api.db.predictions.time_since_prev_trx') as mock_time, \
             patch('api.db.predictions.burst_deviation') as mock_burst, \
             patch('api.db.predictions.rs_anomaly') as mock_rs, \
             patch('api.db.predictions.hour_deviates') as mock_hour, \
             patch('api.db.predictions.robust_amount_outlier') as mock_robust, \
             patch('api.db.predictions.day_of_week_zscore_outlier') as mock_day, \
             patch('api.db.predictions.fraud_burst_candidate') as mock_fraud:
            
            mock_tx.return_value = sample_dataframe
            mock_time.return_value = sample_dataframe
            mock_burst.return_value = sample_dataframe
            mock_rs.return_value = sample_dataframe
            mock_hour.return_value = sample_dataframe
            mock_robust.return_value = sample_dataframe
            mock_day.return_value = sample_dataframe
            mock_fraud.return_value = sample_dataframe
            
            result = generate_features(sample_dataframe)
            
            mock_tx.assert_called_once_with(sample_dataframe)
            mock_time.assert_called_once_with(sample_dataframe)
            mock_burst.assert_called_once_with(sample_dataframe)
            mock_rs.assert_called_once_with(sample_dataframe)
            mock_hour.assert_called_once_with(sample_dataframe)
            mock_robust.assert_called_once_with(sample_dataframe)
            mock_day.assert_called_once_with(sample_dataframe)
            mock_fraud.assert_called_once_with(sample_dataframe)
            assert result is sample_dataframe

    def test_tx_amount_columns_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of tx_amount_columns."""
        result = tx_amount_columns(sample_dataframe.copy())
        
        # Check that new columns are created
        assert 'tx_amount_log' in result.columns
        assert 'tx_amount_log_mean' in result.columns
        assert 'tx_amount_log_std' in result.columns
        assert 'tx_amount_log_deviates' in result.columns
        
        # Check tx_amount_log calculation
        expected_log = np.log1p(sample_dataframe['tx_amount'])
        np.testing.assert_array_almost_equal(result['tx_amount_log'], expected_log)
        
        # Check that deviates column is binary
        assert result['tx_amount_log_deviates'].isin([0, 1]).all()

    def test_tx_amount_columns_groupby_calculations(self, sample_dataframe: pd.DataFrame) -> None:
        """Test groupby calculations in tx_amount_columns."""
        result = tx_amount_columns(sample_dataframe.copy())
        
        # Check that mean and std are calculated per customer
        customer_1001_data = result[result['customer_id'] == 1001]
        customer_1002_data = result[result['customer_id'] == 1002]
        
        # Customer 1001 should have same mean/std for all rows
        assert customer_1001_data['tx_amount_log_mean'].nunique() == 1
        assert customer_1001_data['tx_amount_log_std'].nunique() == 1
        
        # Customer 1002 should have same mean/std for all rows
        assert customer_1002_data['tx_amount_log_mean'].nunique() == 1
        assert customer_1002_data['tx_amount_log_std'].nunique() == 1

    def test_time_since_prev_trx_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of time_since_prev_trx."""
        result = time_since_prev_trx(sample_dataframe.copy())
        
        assert 'secs_since_prev_tx' in result.columns
        
        # First transaction for each customer should have -1
        first_transactions = result.groupby('customer_id').first()
        assert (first_transactions['secs_since_prev_tx'] == -1).all()
        
        # Other transactions should have positive values
        other_transactions = result[result['secs_since_prev_tx'] != -1]
        assert (other_transactions['secs_since_prev_tx'] > 0).all()

    def test_time_since_prev_trx_calculation_accuracy(self, single_customer_dataframe: pd.DataFrame) -> None:
        """Test accuracy of time calculation."""
        result = time_since_prev_trx(single_customer_dataframe.copy())
        
        # Check that time differences are calculated correctly
        expected_diff_1 = (pd.Timestamp('2023-01-01 11:00:00') - pd.Timestamp('2023-01-01 10:00:00')).total_seconds()
        expected_diff_2 = (pd.Timestamp('2023-01-01 12:00:00') - pd.Timestamp('2023-01-01 11:00:00')).total_seconds()
        
        assert result.iloc[1]['secs_since_prev_tx'] == expected_diff_1
        assert result.iloc[2]['secs_since_prev_tx'] == expected_diff_2

    def test_burst_deviation_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of burst_deviation."""
        # First add required columns
        df_with_time = time_since_prev_trx(sample_dataframe.copy())
        df_with_log = tx_amount_columns(df_with_time)
        
        result = burst_deviation(df_with_log.copy())
        
        assert 'burst_id' in result.columns
        assert 'n_tx_in_burst' in result.columns
        assert 'burst_mean' in result.columns
        assert 'burst_std' in result.columns
        assert 'n_trx_per_burst_deviates' in result.columns
        
        # Check that burst_id is calculated per customer
        customer_1001_data = result[result['customer_id'] == 1001]
        customer_1002_data = result[result['customer_id'] == 1002]
        
        # Each customer should have their own burst calculations
        assert customer_1001_data['burst_mean'].nunique() == 1
        assert customer_1002_data['burst_mean'].nunique() == 1

    def test_rs_anomaly_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of rs_anomaly."""
        df_with_log = tx_amount_columns(sample_dataframe.copy())
        
        result = rs_anomaly(df_with_log.copy())
        
        assert 'zscore' in result.columns
        assert 'is_zscore_outlier' in result.columns
        assert 'is_iqr_outlier' in result.columns
        assert 'tx_amount_log_scaled' in result.columns
        assert 'is_rs_anomaly' in result.columns
        
        # Check that outlier columns are binary
        assert result['is_zscore_outlier'].isin([0, 1]).all()
        assert result['is_iqr_outlier'].isin([0, 1]).all()
        assert result['is_rs_anomaly'].isin([0, 1]).all()

    def test_hour_deviates_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of hour_deviates."""
        result = hour_deviates(sample_dataframe.copy())
        
        assert 'day_of_week' in result.columns
        assert 'hour' in result.columns
        assert 'month' in result.columns
        assert 'is_month_start' in result.columns
        assert 'is_month_end' in result.columns
        assert 'is_weekend' in result.columns
        assert 'hour_zscore' in result.columns
        assert 'hour_zscore_deviates' in result.columns
        
        # Check that binary columns are correct
        assert result['is_month_start'].isin([0, 1]).all()
        assert result['is_month_end'].isin([0, 1]).all()
        assert result['is_weekend'].isin([0, 1]).all()
        assert result['hour_zscore_deviates'].isin([0, 1]).all()

    def test_hour_deviates_weekend_calculation(self, sample_dataframe: pd.DataFrame) -> None:
        """Test weekend calculation in hour_deviates."""
        # Create data with weekend transactions
        weekend_df = pd.DataFrame({
            'customer_id': [1001, 1001],
            'tx_datetime': pd.to_datetime(['2023-01-07 10:00:00', '2023-01-08 11:00:00']),  # Saturday and Sunday
            'tx_amount': [100.0, 200.0],
            'tx_fraud': [0, 0]
        })
        
        result = hour_deviates(weekend_df)
        
        # Both should be weekend (day_of_week 5 and 6)
        assert result['is_weekend'].all()

    def test_robust_amount_outlier_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of robust_amount_outlier."""
        df_with_log = tx_amount_columns(sample_dataframe.copy())
        
        result = robust_amount_outlier(df_with_log.copy())
        
        assert 'rolling_median' in result.columns
        assert 'q1' in result.columns
        assert 'q3' in result.columns
        assert 'iqr' in result.columns
        assert 'amount_robust_rolling20' in result.columns
        assert 'is_amount_robust_rolling_outlier' in result.columns
        
        # Check that outlier column is binary
        assert result['is_amount_robust_rolling_outlier'].isin([0, 1]).all()

    def test_robust_amount_outlier_sorting(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that robust_amount_outlier sorts data correctly."""
        df_with_log = tx_amount_columns(sample_dataframe.copy())
        
        # Shuffle the dataframe
        shuffled_df = df_with_log.sample(frac=1).reset_index(drop=True)
        
        result = robust_amount_outlier(shuffled_df)
        
        # Check that data is sorted by customer_id and tx_datetime
        assert result.groupby('customer_id')['tx_datetime'].is_monotonic_increasing.all()

    def test_day_of_week_zscore_outlier_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of day_of_week_zscore_outlier."""
        df_with_hour = hour_deviates(sample_dataframe.copy())
        
        result = day_of_week_zscore_outlier(df_with_hour.copy())
        
        assert 'day_of_week_mean' in result.columns
        assert 'day_of_week_std' in result.columns
        assert 'is_day_of_week_mean_outlier' in result.columns
        assert 'day_of_week_zscore' in result.columns
        assert 'is_day_of_week_zscore_outlier' in result.columns
        
        # Check that outlier columns are binary
        assert result['is_day_of_week_mean_outlier'].isin([0, 1]).all()
        assert result['is_day_of_week_zscore_outlier'].isin([0, 1]).all()

    def test_fraud_burst_candidate_basic_functionality(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic functionality of fraud_burst_candidate."""
        df_with_features = sample_dataframe.copy()
        # Add required columns
        df_with_features = tx_amount_columns(df_with_features)
        df_with_features = time_since_prev_trx(df_with_features)
        df_with_features = burst_deviation(df_with_features)
        df_with_features = hour_deviates(df_with_features)
        
        result = fraud_burst_candidate(df_with_features)
        
        assert 'n_tx_in_prev_24h' in result.columns
        assert 'q90_prev' in result.columns
        assert 'is_24h_burst' in result.columns
        assert 'is_24h_burst_fixed' in result.columns
        assert 'day' in result.columns
        assert 'z_in_day_robust' in result.columns
        assert 'is_anomalous_in_day' in result.columns
        assert 'fraud_burst_candidate' in result.columns
        
        # Check that binary columns are correct
        assert result['is_24h_burst'].isin([0, 1]).all()
        assert result['is_24h_burst_fixed'].isin([0, 1]).all()
        assert result['is_anomalous_in_day'].isin([0, 1]).all()
        assert result['fraud_burst_candidate'].isin([0, 1]).all()

    def test_fraud_burst_candidate_24h_calculation(self, single_customer_dataframe: pd.DataFrame) -> None:
        """Test 24h burst calculation accuracy."""
        df_with_features = single_customer_dataframe.copy()
        df_with_features = tx_amount_columns(df_with_features)
        df_with_features = time_since_prev_trx(df_with_features)
        df_with_features = burst_deviation(df_with_features)
        df_with_features = hour_deviates(df_with_features)
        
        result = fraud_burst_candidate(df_with_features)
        
        # First transaction should have 0 transactions in previous 24h
        assert result.iloc[0]['n_tx_in_prev_24h'] == 0
        
        # Second transaction should have 1 transaction in previous 24h
        assert result.iloc[1]['n_tx_in_prev_24h'] == 1
        
        # Third transaction should have 2 transactions in previous 24h
        assert result.iloc[2]['n_tx_in_prev_24h'] == 2

    def test_empty_dataframe_handling(self) -> None:
        """Test handling of empty DataFrame."""
        # Create empty DataFrame with proper dtypes
        empty_df = pd.DataFrame({
            'customer_id': pd.Series(dtype='int64'),
            'tx_datetime': pd.Series(dtype='datetime64[ns]'),
            'tx_amount': pd.Series(dtype='float64'),
            'tx_fraud': pd.Series(dtype='int8')
        })
        
        # Test functions that don't have dependencies first
        result_tx = tx_amount_columns(empty_df.copy())
        result_time = time_since_prev_trx(empty_df.copy())
        result_hour = hour_deviates(empty_df.copy())
        
        # Test functions that have dependencies with proper setup
        df_with_time = time_since_prev_trx(empty_df.copy())
        df_with_log = tx_amount_columns(df_with_time.copy())
        result_burst = burst_deviation(df_with_log.copy())
        result_rs = rs_anomaly(df_with_log.copy())
        
        # Test robust_amount_outlier with proper setup
        result_robust = robust_amount_outlier(df_with_log.copy())
        
        # Test day_of_week_zscore_outlier with proper setup
        df_with_hour = hour_deviates(empty_df.copy())
        result_day = day_of_week_zscore_outlier(df_with_hour.copy())
        
        # Test fraud_burst_candidate with proper setup
        df_with_features = empty_df.copy()
        df_with_features = tx_amount_columns(df_with_features)
        df_with_features = time_since_prev_trx(df_with_features)
        df_with_features = burst_deviation(df_with_features)
        df_with_features = hour_deviates(df_with_features)
        result_fraud = fraud_burst_candidate(df_with_features)
        
        # All should return empty DataFrames
        assert len(result_tx) == 0
        assert len(result_time) == 0
        assert len(result_burst) == 0
        assert len(result_rs) == 0
        assert len(result_hour) == 0
        assert len(result_robust) == 0
        assert len(result_day) == 0
        assert len(result_fraud) == 0

    def test_single_row_dataframe_handling(self) -> None:
        """Test handling of single row DataFrame."""
        single_row_df = pd.DataFrame({
            'customer_id': [1001],
            'tx_datetime': pd.to_datetime(['2023-01-01 10:00:00']),
            'tx_amount': [100.0],
            'tx_fraud': [0]
        })
        
        # Test functions that should handle single row
        result_tx = tx_amount_columns(single_row_df.copy())
        result_time = time_since_prev_trx(single_row_df.copy())
        result_hour = hour_deviates(single_row_df.copy())
        
        assert len(result_tx) == 1
        assert len(result_time) == 1
        assert len(result_hour) == 1

    def test_eps_constant(self) -> None:
        """Test that eps constant is defined correctly."""
        assert EPS == 1e-9
        assert isinstance(EPS, float)

    def test_dataframe_modification_behavior(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that functions modify the input DataFrame in place."""
        df_copy = sample_dataframe.copy()
        original_columns = set(df_copy.columns)
        
        result = tx_amount_columns(df_copy)
        
        # Check that new columns were added
        new_columns = set(result.columns) - original_columns
        assert len(new_columns) > 0
        
        # Check that result is the same object as input
        assert result is df_copy

    def test_nan_handling_in_calculations(self, sample_dataframe: pd.DataFrame) -> None:
        """Test handling of NaN values in calculations."""
        # Create DataFrame with some NaN values
        df_with_nan = sample_dataframe.copy()
        df_with_nan.loc[1, 'tx_amount'] = np.nan
        
        result = tx_amount_columns(df_with_nan.copy())
        
        # Should handle NaN values gracefully
        assert not result['tx_amount_log'].isna().all()
        
        # Check that groupby operations handle NaN
        customer_1001_data = result[result['customer_id'] == 1001]
        assert not customer_1001_data['tx_amount_log_mean'].isna().all()

    def test_large_dataframe_performance(self) -> None:
        """Test performance with larger DataFrame."""
        # Create larger dataset
        n_rows = 1000
        large_df = pd.DataFrame({
            'customer_id': np.random.randint(1, 10, n_rows),
            'tx_datetime': pd.date_range('2023-01-01', periods=n_rows, freq='1H'),
            'tx_amount': np.random.uniform(10, 1000, n_rows),
            'tx_fraud': np.random.randint(0, 2, n_rows)
        })
        
        # Test that functions complete without error
        result = tx_amount_columns(large_df.copy())
        assert len(result) == n_rows
        
        result = time_since_prev_trx(large_df.copy())
        assert len(result) == n_rows