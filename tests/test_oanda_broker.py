\"\"\"Tests for OANDABroker v20 REST interface.\"\"\"
import json
import os
import urllib.error
import pytest
from trading.live_engine import OANDABroker

def test_oanda_env_and_keys(monkeypatch):
    monkeypatch.setenv('OANDA_API_KEY', 'test-key-123')
    monkeypatch.setenv('OANDA_ACCOUNT_ID', '101-004-12345-001')
    monkeypatch.setenv('OANDA_ENV', 'live')
    monkeypatch.delenv('OANDA_API_URL', raising=False)
    monkeypatch.delenv('OANDA_API_HOST', raising=False)
    monkeypatch.delenv('OANDA_BEARER_TOKEN', raising=False)
    monkeypatch.delenv('OANDA_API_TOKEN', raising=False)
    
    broker = OANDABroker()
    assert broker._token == 'test-key-123'
    assert broker._account_id == '101-004-12345-001'
    assert broker._host == 'https://api-fxtrade.oanda.com'
    assert broker.connect() is True

def test_oanda_practice_default(monkeypatch):
    monkeypatch.setenv('OANDA_API_KEY', 'test-key-123')
    monkeypatch.setenv('OANDA_ACCOUNT_ID', '101-004-12345-001')
    monkeypatch.delenv('OANDA_ENV', raising=False)
    monkeypatch.delenv('OANDA_API_URL', raising=False)
    monkeypatch.delenv('OANDA_API_HOST', raising=False)
    
    broker = OANDABroker()
    assert broker._host == 'https://api-fxpractice.oanda.com'

def test_oanda_market_order_fill(monkeypatch):
    monkeypatch.setenv('OANDA_API_KEY', 'dummy')
    monkeypatch.setenv('OANDA_ACCOUNT_ID', 'dummy')
    broker = OANDABroker()

    class MockResp:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                'orderFillTransaction': {
                    'id': '10',
                    'price': '1.1005',
                    'units': '10000',
                }
            }).encode('utf-8')

    monkeypatch.setattr('urllib.request.urlopen', lambda req, **kwargs: MockResp())
    res = broker.market_order('EURUSD', 'BUY', 1.0)
    assert res['ok'] is True
    assert res['price'] == 1.1005
    assert res['units'] == 10000.0

def test_oanda_market_order_cancel_detected(monkeypatch):
    monkeypatch.setenv('OANDA_API_KEY', 'dummy')
    monkeypatch.setenv('OANDA_ACCOUNT_ID', 'dummy')
    broker = OANDABroker()

    class MockCancelResp:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                'orderCancelTransaction': {
                    'id': '11',
                    'reason': 'INSUFFICIENT_MARGIN',
                }
            }).encode('utf-8')

    monkeypatch.setattr('urllib.request.urlopen', lambda req, **kwargs: MockCancelResp())
    res = broker.market_order('EURUSD', 'BUY', 5.0)
    assert res['ok'] is False
    assert res['reason'] == 'INSUFFICIENT_MARGIN'

def test_oanda_market_order_http_error(monkeypatch):
    monkeypatch.setenv('OANDA_API_KEY', 'dummy')
    monkeypatch.setenv('OANDA_ACCOUNT_ID', 'dummy')
    broker = OANDABroker()

    import io
    def mock_urlopen_err(req, **kwargs):
        fp = io.BytesIO(b'{\"errorMessage\": \"Invalid instrument\"}')
        raise urllib.error.HTTPError('http://dummy', 400, 'Bad Request', {}, fp)

    monkeypatch.setattr('urllib.request.urlopen', mock_urlopen_err)
    res = broker.market_order('INVALID', 'BUY', 1.0)
    assert res['ok'] is False
    assert res['reason'] == 'http_error_400'
    assert 'Invalid instrument' in str(res['error'])
