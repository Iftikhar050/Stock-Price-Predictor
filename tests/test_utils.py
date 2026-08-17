import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from src.psx_predictor.models.utils import choose_global_cutoff

class TestUtils(unittest.TestCase):
    @patch('src.psx_predictor.models.utils.engine.connect')
    @patch('src.psx_predictor.db.repository.get_active_tickers')
    def test_choose_global_cutoff_fallback(self, mock_get_tickers, mock_connect):
        # Create 10 mock dates, sorted descending (newest first)
        base_date = datetime(2023, 1, 1).date()
        mock_dates = [(base_date + timedelta(days=i),) for i in range(10)][::-1]
        
        # Mock database returns
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # First call fetches dates, second call fetches ticker counts
        mock_conn.execute.side_effect = [
            MagicMock(fetchall=lambda: mock_dates),
            MagicMock(fetchall=lambda: [('PSO', 10)])
        ]
        
        mock_get_tickers.return_value = ['PSO']
        
        # Request a split that triggers the fallback: 
        # test_trading_days + min_train_trading_days > 10
        cutoff_str, valid_tickers = choose_global_cutoff(test_trading_days=5, min_train_trading_days=10)
        
        # With 10 dates, len(all_dates) = 10.
        # Fallback index = 10 // 5 = 2
        # Since dates are descending (index 0 is newest, 9 is oldest), 
        # index 2 is the 3rd newest date.
        # This leaves 8 dates before or on the cutoff (older), and 2 dates after the cutoff (newer).
        # This is an 80/20 train/test split.
        expected_cutoff = mock_dates[2][0]
        
        self.assertEqual(cutoff_str, expected_cutoff.strftime('%Y-%m-%d'))
        
if __name__ == '__main__':
    unittest.main()
