import asyncio
import json
import io
import urllib.error
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from agent import Config, Book, Agent, AIUnavailable, indicators, closed_candles, validate_quote, parse_decision

class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'test.db'
        self.c = Config()
        self.b = Book(self.path,self.c)
        self.q = {'bid':100.0,'ask':100.1,'last':100,'ts':time.time()}
    def tearDown(self):
        self.b.db.close()
        self.tmp.cleanup()
    def test_fee_accounting_roundtrip(self):
        self.assertTrue(self.b.buy(self.q,1,'test'))
        p = self.b.s['position'].copy()
        q = {**self.q,'bid':105}
        expected = p['qty']*105*(1-self.c.slippage)*(1-self.c.fee)-p['cost']
        self.assertTrue(self.b.sell(q,'test'))
        self.assertAlmostEqual(self.b.s['cash'],1000+expected)
        self.assertAlmostEqual(self.b.s['pnl'],expected)
        self.assertEqual(self.b.s['trades'],1)
    def test_one_position_and_no_short(self):
        self.assertFalse(self.b.sell(self.q,'no short'))
        self.assertTrue(self.b.buy(self.q,1,'test'))
        self.assertFalse(self.b.buy(self.q,1,'duplicate'))
    def test_stop_risk_includes_costs(self):
        self.b.buy(self.q,1,'test')
        p = self.b.s['position']
        loss = p['cost']-p['qty']*p['stop']*(1-self.c.slippage)*(1-self.c.fee)
        self.assertLessEqual(loss,1000*self.c.risk+1e-8)
        self.assertLessEqual(p['cost'],1000*self.c.allocation+1e-8)
    def test_gap_stop_fills_at_current_quote(self):
        self.b.buy(self.q,1,'test')
        self.b.protect({**self.q,'bid':90})
        self.assertIsNone(self.b.s['position'])
        row = self.b.db.execute("SELECT data FROM events WHERE kind='SELL'").fetchone()
        self.assertAlmostEqual(json.loads(row[0])['exit'],90*(1-self.c.slippage))
    def test_paused_still_protects(self):
        self.b.buy(self.q,1,'test')
        self.b.s['paused'] = True
        self.b.protect({**self.q,'bid':90})
        self.assertIsNone(self.b.s['position'])
        self.assertFalse(self.b.buy(self.q,1,'paused'))
    def test_daily_limit_latches(self):
        self.b.s['cash'] = 960
        self.b.mark(100)
        self.assertTrue(self.b.s['day_halted'])
        self.b.s['cash'] = 1000
        self.b.mark(100)
        self.assertTrue(self.b.s['day_halted'])
        self.b.mark(100,'2099-01-01')
        self.assertFalse(self.b.s['day_halted'])
    def test_restart_preserves_account_and_candle(self):
        self.b.buy(self.q,1,'test')
        self.b.s['last_candle'] = 123
        self.b.s['paused'] = True
        self.b.save()
        restored = Book(self.path,Config(initial=9000))
        self.assertEqual(restored.s,self.b.s)
        restored.db.close()
    def test_quote_stale_future_and_crossed(self):
        for row in [dict(ts='1000',bidPx='100',askPx='101',last='100'),
                    dict(ts='2000000',bidPx='100',askPx='101',last='100'),
                    dict(ts='1000000',bidPx='102',askPx='101',last='100')]:
            with self.assertRaises(ValueError): validate_quote(row,1000)
    def test_ai_schema_rejects_bad_outputs(self):
        for raw in [{'action':'SHORT','confidence':90,'reason':'x'},
                    {'action':'BUY','confidence':float('nan'),'reason':'x'},
                    {'action':'BUY','confidence':101,'reason':'x'},
                    {'action':'BUY','confidence':80,'reason':''}]:
            with self.assertRaises(ValueError): parse_decision(raw)
    def test_ai_failure_does_not_buy(self):
        bot = Agent(self.c,self.b)
        bot.quote = self.q
        bot.ai_key = "test-key"
        with patch('agent.ask_ai',new=AsyncMock(side_effect=ValueError('bad'))):
            asyncio.run(bot.analyze(int(time.time())-60,{'1m':{'atr14':1}}))
        self.assertIsNone(self.b.s['position'])
        self.assertEqual(self.b.s['ai_failures'],1)
    def test_expired_decision_does_not_buy(self):
        bot = Agent(self.c,self.b)
        bot.quote = self.q
        bot.ai_key = "test-key"
        answer = {'action':'BUY','confidence':90,'reason':'x'}
        with patch('agent.ask_ai',new=AsyncMock(return_value=answer)):
            asyncio.run(bot.analyze(int(time.time())-120,{'1m':{'atr14':1}}))
        self.assertIsNone(self.b.s['position'])
    def test_outbox_survives_restart(self):
        self.b.save(notice='pending')
        other = Book(self.path,self.c)
        self.assertEqual(other.db.execute('SELECT text FROM outbox').fetchone()[0],'pending')
        other.db.close()

class PairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'pair.db'
        self.b = Book(self.path,Config())
        self.env = patch.dict('os.environ', {'TELEGRAM_BOT_TOKEN':'test-token'})
        self.env.start()
        self.bot = Agent(Config(),self.b)
    def tearDown(self):
        self.env.stop()
        self.b.db.close()
        self.tmp.cleanup()
    def test_only_private_start_binds(self):
        for kind,text in [('private','/status'),('private','hello'),('group','/start')]:
            self.assertFalse(self.bot.pair_telegram('123',kind,text))
        self.assertFalse(self.bot.chat)
        self.assertTrue(self.bot.pair_telegram('123','private','/start'))
    def test_pair_persists_and_cannot_be_replaced(self):
        self.assertTrue(self.bot.pair_telegram('123','private','/start'))
        self.assertFalse(self.bot.pair_telegram('456','private','/start'))
        other = Book(self.path,Config())
        restored = Agent(Config(),other)
        self.assertEqual(restored.chat,'123')
        other.db.close()
    def test_no_token_or_invalid_sender_cannot_bind(self):
        self.assertFalse(self.bot.pair_telegram('-123','private','/start'))
        self.bot.token = ''
        self.assertFalse(self.bot.pair_telegram('123','private','/start'))

class ModelFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b = Book(Path(self.tmp.name)/'models.db',Config())
        self.bot = Agent(Config(),self.b)
        self.bot.ai_key = 'test-key'
        self.bot.models = ['model-a','model-b','model-c']
        self.candle = int(time.time())-60
        self.answer = {'action':'HOLD','confidence':50,'reason':'test'}
    def tearDown(self):
        self.b.db.close()
        self.tmp.cleanup()
    def error(self,status,body=None):
        return urllib.error.HTTPError('https://example.invalid',status,'error',{},io.BytesIO(json.dumps(body or {}).encode()))
    def test_404_fallback_and_sticky_success(self):
        mock = AsyncMock(side_effect=[self.error(404),self.answer,self.answer])
        with patch('agent.ask_ai',new=mock):
            result=asyncio.run(self.bot.ai_decision(self.candle,{}))
            self.assertEqual(result['model'],'model-b')
            asyncio.run(self.bot.ai_decision(self.candle,{}))
        self.assertEqual([c.args[2] for c in mock.await_args_list],['model-a','model-b','model-b'])
        self.assertEqual(self.b.s['ai_calls'],3)
    def test_unknown_quota_stops_all_models_and_persists(self):
        mock=AsyncMock(side_effect=self.error(429))
        with patch('agent.ask_ai',new=mock):
            with self.assertRaises(AIUnavailable): asyncio.run(self.bot.ai_decision(self.candle,{}))
            with self.assertRaises(AIUnavailable): asyncio.run(self.bot.ai_decision(self.candle,{}))
        self.assertEqual(mock.await_count,1)
        self.assertGreater(self.b.s['ai_global_until'],time.time())
    def test_model_specific_quota_uses_next_model(self):
        body={'error':{'details':[{'violations':[{'quotaId':'RequestsPerDay','quotaDimensions':{'model':'model-a'}}]}]}}
        mock=AsyncMock(side_effect=[self.error(429,body),self.answer])
        with patch('agent.ask_ai',new=mock):
            result=asyncio.run(self.bot.ai_decision(self.candle,{}))
        self.assertEqual(result['model'],'model-b')
        self.assertGreater(self.b.s['ai_cooldowns']['model-a'],time.time()+86000)
    def test_invalid_key_stops_all(self):
        body={'error':{'details':[{'reason':'API_KEY_INVALID'}]}}
        mock=AsyncMock(side_effect=self.error(400,body))
        with patch('agent.ask_ai',new=mock):
            with self.assertRaises(AIUnavailable): asyncio.run(self.bot.ai_decision(self.candle,{}))
        self.assertEqual(mock.await_count,1)
    def test_budget_counts_fallback_attempts(self):
        self.bot.c.ai_daily_limit=1
        mock=AsyncMock(side_effect=self.error(404))
        with patch('agent.ask_ai',new=mock):
            with self.assertRaises(AIUnavailable): asyncio.run(self.bot.ai_decision(self.candle,{}))
        self.assertEqual(mock.await_count,1)
        self.assertEqual(self.b.s['ai_calls'],1)
    def test_deadline_does_not_send_stale_request(self):
        mock=AsyncMock(return_value=self.answer)
        with patch('agent.ask_ai',new=mock):
            with self.assertRaises(AIUnavailable): asyncio.run(self.bot.ai_decision(int(time.time())-120,{}))
        self.assertEqual(mock.await_count,0)
    def test_all_models_failure_holds(self):
        mock=AsyncMock(side_effect=ValueError('invalid response'))
        with patch('agent.ask_ai',new=mock):
            asyncio.run(self.bot.analyze(self.candle,{}))
        self.assertEqual(mock.await_count,3)
        self.assertEqual(self.b.s['last_decision']['action'],'HOLD')
        self.assertIsNone(self.b.s['position'])

class ModelConfigTests(unittest.TestCase):
    def test_key_in_model_field_is_not_echoed(self):
        with tempfile.TemporaryDirectory() as directory:
            book=Book(Path(directory)/'test.db',Config())
            fake='AQ.FAKE_SECRET_FOR_TEST_ONLY'
            with patch.dict('os.environ',{'GEMINI_MODEL':fake,'GEMINI_MODELS':''}):
                with self.assertLogs('paper',level='WARNING') as logs:
                    bot=Agent(Config(),book)
            self.assertNotIn(fake,bot.report())
            self.assertNotIn(fake,' '.join(logs.output))
            self.assertEqual(bot.models,['gemini-3.8-flash','gemini-3.7-flash','gemini-3.6-flash'])
            book.db.close()
    def test_bad_model_list_and_legacy_state_are_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            book=Book(Path(directory)/'test.db',Config())
            fake='AQ.FAKE_SECRET_FOR_TEST_ONLY'
            book.s['ai_last_model']=fake
            book.s['ai_last_error']=fake+': HTTP_404'
            with patch.dict('os.environ',{'GEMINI_MODELS':fake}):
                with self.assertLogs('paper',level='WARNING'):
                    bot=Agent(Config(),book)
            self.assertNotIn(fake,bot.report())
            self.assertEqual(len(bot.models),3)
            book.db.close()

class CandleTests(unittest.TestCase):
    def rows(self):
        return [[str(i*60000),'100','101','99','100','10','0','0','1'] for i in range(200)]
    def test_closed_only_and_flat_indicators(self):
        rows = self.rows()+[['12000000','100','101','99','100','10','0','0','0']]
        cs = closed_candles(rows,60,12001)
        self.assertEqual(len(cs),200)
        x = indicators(cs)
        self.assertEqual(x['rsi14'],50)
        self.assertEqual(x['atr14'],2)
        self.assertEqual(x['macd_hist'],0)
    def test_gap_duplicate_stale_rejected(self):
        rows = self.rows()
        for data,now in [(rows[:50]+rows[51:],12001),(rows+[rows[-1]],12001),(rows,13000)]:
            with self.assertRaises(ValueError): closed_candles(data,60,now)

if __name__ == '__main__':
    unittest.main()
