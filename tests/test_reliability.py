"""Offline behavioral regressions; all requests/model calls are intercepted."""
import contextlib
import csv
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pypdf
import requests
import config

# Tests always run against the bundled example profile, whatever RESEARCH_PROFILE says.
config.PROFILE_PATH = config.ROOT_DIR / "profiles" / "example_profile.json"
import fulltext
import identity
import llm_draft
import output
import paper_scout
import vault
from sources import arxiv, arxiv_oai, common, openalex, semantic_scholar
from sources.common import SearchResult


def raw(title="Synthetic futures study", paper_id="one", abstract="Out-of-sample futures trading transaction costs", **kwargs):
    return {"title": title, "paperId": paper_id, "abstract": abstract,
            "year": 2025, "authors": [{"name": "A Researcher"}],
            "source": "semantic_scholar", **kwargs}


def response(data=None, status=200, content=b"", headers=None):
    r = Mock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = data
    r.content = content
    r.close = Mock()
    return r


class OfflineCase(unittest.TestCase):
    def setUp(self):
        # Unexpected network traffic fails rather than quietly reaching a service.
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('unexpected network'))
        self.network.start()
        self.addCleanup(self.network.stop)


class IdentityTests(OfflineCase):
    def test_richer_duplicate_and_provenance(self):
        records = [raw(abstract='', query='a'), raw(paper_id='two', source='openalex', query='b')]
        merged = identity.merge_records(records)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]['abstract'])
        self.assertEqual(len(merged[0]['provenance']), 2)
        self.assertEqual(merged[0]['queries'], ['a', 'b'])
        self.assertGreater(paper_scout.parse_paper(merged[0])['score'], 0)

    def test_prefix_collision_does_not_merge(self):
        rows = identity.merge_records([raw('A'*60+' one'), raw('A'*60+' two', paper_id='two')])
        self.assertEqual(len(rows), 2)

    def test_missing_ids_do_not_collapse(self):
        self.assertEqual(len(identity.merge_records([raw('one', ''), raw('two', '')])), 2)

    def test_normalized_doi_and_missing_field_merge(self):
        rows = identity.merge_records([
            raw(abstract='', externalIds={'DOI': 'https://doi.org/10.123/ABC'}),
            raw('Updated title', 'two', externalIds={'DOI': '10.123/abc'})])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['externalIds']['DOI'], '10.123/abc')

    def test_conflicting_dois_same_title_stay_separate(self):
        rows = identity.merge_records([raw(externalIds={'DOI': '10.1/a'}),
                                      raw(paper_id='two', externalIds={'DOI': '10.1/b'})])
        self.assertEqual(len(rows), 2)

    def test_ambiguous_authors_stay_separate(self):
        self.assertEqual(len(identity.merge_records([raw(authors=[]), raw(paper_id='two', authors=[])])), 2)

    def test_bridge_record_merges_both_groups(self):
        rows = identity.merge_records([
            raw('one', 'one', externalIds={'DOI': '10.1/a'}),
            raw('two', 'two', externalIds={'ArXiv': '1234.56789v1'}),
            raw('three', 'three', externalIds={'DOI': '10.1/a', 'ArXiv': '1234.56789v2'})])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['provenance']), 3)


class OutputTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        for name in ('VAULT_PATH', 'RESULTS_DIR'):
            p = patch.object(config, name, self.path)
            p.start()
            self.addCleanup(p.stop)

    def test_note_preserves_user_text_on_rerun_and_title_change(self):
        paper = paper_scout.parse_paper(raw())
        path = vault.save_obsidian_note(paper, 'query')
        path.write_text('USER CONTENT', encoding='utf-8')
        paper['title'] = 'New title'
        self.assertEqual(vault.save_obsidian_note(paper, 'query', 'new draft'), path)
        self.assertEqual(path.read_text(encoding='utf-8'), 'USER CONTENT')

    def test_note_title_collisions_get_distinct_files(self):
        a = vault.save_obsidian_note(paper_scout.parse_paper(raw('A'*60+' one')), 'q')
        b = vault.save_obsidian_note(paper_scout.parse_paper(raw('A'*60+' two', 'two')), 'q')
        self.assertNotEqual(a, b)

    def test_csv_runs_cannot_overwrite(self):
        a, _ = output.save_csv([paper_scout.parse_paper(raw())], 'same query')
        b, _ = output.save_csv([], 'same query')
        self.assertNotEqual(a, b)
        self.assertIn('Synthetic futures study', a.read_text(encoding='utf-8'))

    def test_json_update_is_valid_and_leaves_no_temp(self):
        path = self.path / 'manifest.json'
        output.save_json(path, {'status': 'running'})
        output.save_json(path, {'status': 'complete'})
        self.assertEqual(json.loads(path.read_text())['status'], 'complete')
        self.assertEqual(list(self.path.glob('*.tmp')), [])


class ProviderTests(OfflineCase):
    def test_bad_abstract_type_is_failure_before_identity_merge(self):
        with patch.object(semantic_scholar, 'get_json', return_value={'data': [raw(abstract=123)]}):
            result = semantic_scholar.search_semantic_scholar('q', 1)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.records, [])

    def test_openalex_cursor_limit_150_returns_150_unique(self):
        calls = []
        def get(*args, **kwargs):
            params = kwargs['params']
            calls.append(dict(params))
            start = 0 if params['cursor'] == '*' else 100
            return {'results': [{'id': str(i), 'title': f'Paper {i}'} for i in range(start, start+100)],
                    'meta': {'next_cursor': 'next'}}
        with patch.object(openalex, 'get_json', side_effect=get):
            result = openalex.search_openalex('q', 150)
        self.assertEqual(len({r['paperId'] for r in result.records}), 150)
        self.assertEqual([c['per-page'] for c in calls], [100, 100])
        self.assertTrue(result.limit_reached)

    def test_openalex_failure_retains_first_page(self):
        first = {'results': [{'id': 'a', 'title': 'A'}], 'meta': {'next_cursor': 'next'}}
        with patch.object(openalex, 'get_json', side_effect=[first, common.ProviderError('HTTP 503')]):
            result = openalex.search_openalex('q', 150)
        self.assertEqual(result.status, 'partial')
        self.assertEqual(result.pages, 1)
        self.assertEqual(len(result.records), 1)

    def test_invalid_json_schema_is_failure_not_empty(self):
        with patch.object(openalex, 'get_json', return_value={'error': 'bad query'}):
            self.assertEqual(openalex.search_openalex('q').status, 'failed')

    def test_empty_success_is_distinct(self):
        with patch.object(openalex, 'get_json', return_value={'results': []}):
            self.assertEqual(openalex.search_openalex('q').status, 'complete_empty')

    def test_openalex_pdf_uses_direct_location(self):
        result = openalex._parse_work({'open_access': {'oa_url': 'https://host/landing'},
                                     'best_oa_location': {'pdf_url': 'https://host/paper.pdf'}})
        self.assertEqual(result['openAccessPdf']['url'], 'https://host/paper.pdf')

    def test_semantic_pagination_uses_next(self):
        with patch.object(semantic_scholar, 'get_json', side_effect=[
                {'data': [raw(paper_id=str(i)) for i in range(100)], 'next': 100},
                {'data': [raw(paper_id=str(i)) for i in range(100, 150)]}]) as get:
            result = semantic_scholar.search_semantic_scholar('q', 150)
        self.assertEqual(len(result.records), 150)
        self.assertEqual(get.call_args.kwargs['params']['offset'], 100)
        self.assertEqual(get.call_args.kwargs['params']['limit'], 50)

    def test_arxiv_old_identifier_preserved(self):
        xml = b'<entry xmlns="http://www.w3.org/2005/Atom"><id>http://arxiv.org/abs/cond-mat/1234567v2</id><title>A</title></entry>'
        parsed = arxiv._parse_entry(arxiv.ET.fromstring(xml))
        self.assertEqual(parsed['externalIds']['ArXiv'], 'cond-mat/1234567v2')

    def test_arxiv_error_feed_is_not_a_paper(self):
        xml = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#x</id><title>Error</title></entry></feed>'
        with patch.object(arxiv, 'request', return_value=response(content=xml)):
            self.assertEqual(arxiv.search_arxiv('q').status, 'failed')

    def test_arxiv_edge_refusal_is_explained(self):
        with patch.object(arxiv, 'request', side_effect=common.ProviderError('HTTP 406')):
            result = arxiv.search_arxiv('q')
        self.assertEqual(result.status, 'failed')
        self.assertIn('openalex/semantic_scholar', result.error)

    def test_default_arxiv_is_local_index_not_refusing_api(self):
        self.assertIn('arxiv', config.DEFAULT_SOURCES)
        self.assertNotIn('arxiv_api', config.DEFAULT_SOURCES)
        self.assertIs(paper_scout.SEARCH_FUNCS['arxiv'], arxiv_oai.search_arxiv_index)


OAI_PAGE = b'''<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:2401.00001</identifier><datestamp>2026-09-01</datestamp></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2401.00001</id><created>2026-08-30</created>
<authors><author><keyname>Doe</keyname><forenames>Jane</forenames></author></authors>
<title>Intraday   momentum in equity index
 futures</title><doi>10.1/X</doi><abstract>We test hedging flows with transaction costs.</abstract></arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:2301.99999</identifier><datestamp>2026-09-02</datestamp></header></record>
<resumptionToken>{token}</resumptionToken></ListRecords></OAI-PMH>'''


class ArxivIndexTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'idx.sqlite'

    def page(self, token=''):
        return response(content=OAI_PAGE.replace(b'{token}', token.encode()))

    def test_parse_normalizes_and_tracks_deletions(self):
        papers, deleted, stamp, token = arxiv_oai.parse_records(arxiv_oai.ET.fromstring(self.page('abc').content))
        self.assertEqual(papers[0]['title'], 'Intraday momentum in equity index futures')
        self.assertEqual(json.loads(papers[0]['authors']), ['Jane Doe'])
        self.assertEqual((deleted, stamp, token), (['2301.99999'], '2026-09-02', 'abc'))

    def test_sync_follows_token_and_search_finds_stemmed_terms(self):
        with patch.object(arxiv_oai, 'request', side_effect=[self.page('t1'), self.page('')]) as req:
            summary = arxiv_oai.sync(['q-fin'], self.db, log=lambda *_: None)
        self.assertEqual(req.call_args_list[1].kwargs['params'], {'verb': 'ListRecords', 'resumptionToken': 't1'})
        self.assertEqual(summary['q-fin']['error'], '')
        result = arxiv_oai.search_arxiv_index('hedge flow futures', 5, self.db)
        self.assertEqual(result.status, 'complete')
        rec = result.records[0]
        self.assertEqual((rec['year'], rec['publicationDate']), (2024, '2024-01'))
        self.assertEqual(rec['externalIds'], {'ArXiv': '2401.00001', 'DOI': '10.1/X'})

    def test_incremental_sync_uses_stored_datestamp(self):
        with patch.object(arxiv_oai, 'request', return_value=self.page()):
            arxiv_oai.sync(['q-fin'], self.db, log=lambda *_: None)
        with patch.object(arxiv_oai, 'request', return_value=self.page()) as req:
            arxiv_oai.sync(['q-fin'], self.db, log=lambda *_: None)
        self.assertEqual(req.call_args.kwargs['params']['from'], '2026-09-02')

    def test_resync_does_not_duplicate_fts_rows(self):
        with patch.object(arxiv_oai, 'request', return_value=self.page()):
            arxiv_oai.sync(['q-fin'], self.db, log=lambda *_: None)
            arxiv_oai.sync(['q-fin'], self.db, log=lambda *_: None)
        self.assertEqual(len(arxiv_oai.search_arxiv_index('futures', 10, self.db).records), 1)

    def test_missing_index_and_oai_error_are_failures(self):
        self.assertEqual(arxiv_oai.search_arxiv_index('q', 5, self.db).status, 'failed')
        bad = response(content=b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badArgument"/></OAI-PMH>')
        with patch.object(arxiv_oai, 'request', return_value=bad):
            self.assertIn('badArgument', arxiv_oai.sync(['q-fin'], self.db, log=lambda *_: None)['q-fin']['error'])

    def test_arxiv_malformed_xml_is_failure(self):
        with patch.object(arxiv, 'request', return_value=response(content=b'<broken')):
            self.assertEqual(arxiv.search_arxiv('q').status, 'failed')


class TransportTests(OfflineCase):
    def setUp(self):
        super().setUp()
        common._sessions.clear()
        common._last_request.clear()

    def test_rate_limit_spans_separate_queries(self):
        session = Mock()
        session.get.return_value = response({'data': [raw()]})
        common._sessions['semantic_scholar'] = session
        with patch.object(common.time, 'monotonic', return_value=10), patch.object(common.time, 'sleep') as sleep:
            semantic_scholar.search_semantic_scholar('first', 1)
            semantic_scholar.search_semantic_scholar('second', 1)
        sleep.assert_called_once_with(config.SS_RATE_LIMIT)

    def test_retry_after_and_transient_recovery(self):
        session = Mock()
        session.get.side_effect = [response(status=429, headers={'Retry-After': '2'}), response({'results': []})]
        common._sessions['test'] = session
        result = SearchResult()
        with patch.object(common.time, 'sleep') as sleep:
            data = common.get_json('test', 'https://example.test', 0, result)
        self.assertEqual(data, {'results': []})
        self.assertEqual(result.retries, 1)
        sleep.assert_called_once_with(2)

    def test_long_retry_after_fails_without_early_retry(self):
        session = Mock()
        session.get.return_value = response(status=429, headers={'Retry-After': '3600'})
        common._sessions['test'] = session
        with patch.object(common.time, 'sleep') as sleep, self.assertRaises(common.ProviderError):
            common.request('test', 'https://example.test', 0, SearchResult())
        sleep.assert_not_called()
        self.assertEqual(session.get.call_count, 1)

    def test_retry_date_and_invalid_header(self):
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
        self.assertTrue(28 <= common.retry_delay(future, 1) <= 31)
        self.assertEqual(common.retry_delay('garbage', 4), 4)

    def test_errors_do_not_leak_url_secrets(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError('https://host/?api_key=SECRET')
        common._sessions['test'] = session
        with patch.object(common.time, 'sleep'), self.assertRaises(common.ProviderError) as raised:
            common.request('test', 'https://example.test', 0, SearchResult())
        self.assertNotIn('SECRET', str(raised.exception))
        self.assertEqual(session.get.call_count, 3)

    def test_malformed_json_closes_response(self):
        r = response()
        r.json.side_effect = ValueError('bad')
        with patch.object(common, 'request', return_value=r), self.assertRaises(common.ProviderError):
            common.get_json('test', 'https://example.test', 0, SearchResult())
        r.close.assert_called_once()


class PdfTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.paper = paper_scout.parse_paper(raw(externalIds={'ArXiv': '1234.56789'},
                                                  openAccessPdf={'url': 'https://arxiv.org/pdf/1234.56789'}))
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        stream = io.BytesIO()
        writer.write(stream)
        self.pdf = stream.getvalue()

    def mock_download(self, payload, headers=None):
        r = Mock()
        r.headers = headers or {}
        r.iter_content.return_value = [payload]
        context = Mock()
        context.__enter__ = Mock(return_value=r)
        context.__exit__ = Mock(return_value=False)
        return context

    def test_direct_arxiv_without_doi_or_email_and_cache(self):
        with patch.object(config, 'UNPAYWALL_EMAIL', ''), patch.object(fulltext.requests, 'get', return_value=self.mock_download(self.pdf)) as get:
            path = fulltext.fetch_fulltext_pdf(self.paper, self.path)
            self.assertTrue(Path(path).exists())
            self.assertEqual(self.paper['fulltext_status'], 'downloaded')
            self.assertEqual(fulltext.fetch_fulltext_pdf(self.paper, self.path), path)
            self.assertEqual(self.paper['fulltext_status'], 'cached')
            self.assertEqual(get.call_count, 1)

    def test_html_with_pdf_suffix_is_rejected(self):
        with patch.object(fulltext.requests, 'get', return_value=self.mock_download(b'<html>error</html>')):
            self.assertEqual(fulltext.fetch_fulltext_pdf(self.paper, self.path), '')
        self.assertEqual(self.paper['fulltext_status'], 'failed')
        self.assertEqual(list(self.path.iterdir()), [])

    def test_size_limit_and_cleanup(self):
        with patch.object(fulltext, 'MAX_PDF_BYTES', 20), patch.object(fulltext.requests, 'get', return_value=self.mock_download(self.pdf)):
            self.assertEqual(fulltext.fetch_fulltext_pdf(self.paper, self.path), '')
        self.assertEqual(list(self.path.iterdir()), [])

    def test_corrupt_cache_is_refetched(self):
        dest = self.path / f'{fulltext.cache_file_id(self.paper)}.pdf'
        dest.write_bytes(b'corrupt')
        with patch.object(fulltext.requests, 'get', return_value=self.mock_download(self.pdf)):
            self.assertEqual(fulltext.fetch_fulltext_pdf(self.paper, self.path), str(dest))
        self.assertEqual(dest.read_bytes(), self.pdf)

    def test_new_preprint_version_has_separate_cache(self):
        v1 = dict(self.paper, arxiv_id='1234.56789v1', oa_pdf='https://arxiv.org/pdf/1234.56789v1')
        v2 = dict(self.paper, arxiv_id='1234.56789v2', oa_pdf='https://arxiv.org/pdf/1234.56789v2')
        self.assertEqual(identity.file_id(v1), identity.file_id(v2))
        self.assertNotEqual(fulltext.cache_file_id(v1), fulltext.cache_file_id(v2))

    def test_unpaywall_fallback_after_bad_direct_pdf(self):
        self.paper['doi'] = '10.123/example'
        lookup = self.mock_download(b'')
        lookup.__enter__.return_value.json.return_value = {'best_oa_location': {'url_for_pdf': 'https://publisher/paper.pdf'}}
        with patch.object(config, 'UNPAYWALL_EMAIL', 'fixture@example.test'), \
             patch.object(fulltext.requests, 'get', side_effect=[self.mock_download(b'<html/>'), lookup, self.mock_download(self.pdf)]):
            result = fulltext.fetch_fulltext_pdf(self.paper, self.path)
        self.assertTrue(Path(result).exists())
        self.assertEqual(self.paper['fulltext_attempts'][0]['status'], 'invalid_pdf')
        self.assertEqual(self.paper['fulltext_status'], 'downloaded')


class DraftTests(OfflineCase):
    def test_invalid_model_schema_returns_failure_marker(self):
        for data in ([], {'response': 123}):
            with self.subTest(data=data), patch.object(llm_draft.requests, 'post', return_value=response(data)):
                result = llm_draft.draft_hypothesis({'abstract': 'Synthetic abstract'})
            self.assertTrue(result.startswith('[OLLAMA ERROR]'))


class MainTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        p = patch.object(config, 'RESULTS_DIR', self.path)
        p.start()
        self.addCleanup(p.stop)

    def run_main(self, result, *flags):
        with patch.dict(paper_scout.SEARCH_FUNCS, {'openalex': Mock(return_value=result)}), contextlib.redirect_stdout(io.StringIO()):
            rc = paper_scout.main(['synthetic', '--sources', 'openalex', '--select', 'score', '--min-score', '0', *flags])
        run = sorted((self.path / 'runs').iterdir())[-1]
        return rc, run, json.loads((run/'manifest.json').read_text())

    def test_all_failed_nonzero_and_manifest(self):
        rc, run, manifest = self.run_main(SearchResult(error='HTTP 503'))
        self.assertEqual(rc, 1)
        self.assertEqual(manifest['status'], 'failed')
        self.assertTrue(list(run.glob('*_all.csv')))

    def test_partial_saves_records_and_nonzero(self):
        rc, run, manifest = self.run_main(SearchResult(records=[raw()], error='HTTP 503'))
        self.assertEqual(rc, 2)
        self.assertEqual(manifest['status'], 'partial')
        self.assertEqual(len(json.loads((run/'papers.json').read_text())), 1)

    def test_empty_success_zero(self):
        rc, _, manifest = self.run_main(SearchResult(pages=1))
        self.assertEqual(rc, 0)
        self.assertEqual(manifest['status'], 'complete')

    def test_drafts_without_vault_and_only_draft_budget_fetched(self):
        rows = [raw(f'Paper {i}', str(i)) for i in range(5)]
        with patch.object(paper_scout, 'fetch_fulltext_pdf', return_value='') as fetch, \
             patch.object(paper_scout, 'draft_hypothesis', return_value='Synthetic draft') as draft, \
             patch.object(paper_scout, 'save_obsidian_note') as note:
            rc, run, _ = self.run_main(SearchResult(records=rows), '--draft-hypothesis', '--draft-limit', '2')
        self.assertEqual(rc, 0)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(draft.call_count, 2)
        note.assert_not_called()
        self.assertEqual(len(json.loads((run/'drafts.json').read_text())), 2)

    def test_draft_failure_saved_and_partial(self):
        with patch.object(paper_scout, 'fetch_fulltext_pdf', return_value=''), \
             patch.object(paper_scout, 'draft_hypothesis', return_value='[OLLAMA ERROR] unavailable'):
            rc, run, _ = self.run_main(SearchResult(records=[raw()]), '--draft-hypothesis')
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads((run/'drafts.json').read_text())[0]['status'], 'failed')

    def test_invalid_cli_rejected_before_writes(self):
        for flags in (['--sources', ''], ['--limit', '0'], ['--draft-limit', '-1'], ['--top', '0']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                paper_scout.main(flags)
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(list(self.path.iterdir()), [])


if __name__ == '__main__':
    unittest.main()


import research_profile
import routing


def paper(title, abstract):
    return {"title": title, "abstract_full": abstract}


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = research_profile.load()

    def route(self, title, abstract):
        return routing.route(paper(title, abstract), self.profile)

    def test_negated_rigor_is_contradicted_not_credited(self):
        r = self.route("A simple rule-based equity index futures trading strategy",
                       "A simple rule-based equity index futures trading strategy is tested on intraday "
                       "S&P 500 E-mini data. No out-of-sample testing was performed and transaction costs "
                       "were ignored. We find positive intraday returns and predictable reversals.")
        self.assertEqual(r["evidence"], {"out_of_sample": "contradicted", "costs": "contradicted"})

    def test_non_finance_and_missing_abstract(self):
        self.assertEqual(self.route("Iceberg detection", "We detect icebergs in sea ice imagery using "
                         "unsupervised learning on satellite data across twenty Antarctic seasons and "
                         "validate the detected populations against independent ship-based field surveys.")["route"], "irrelevant")
        self.assertEqual(self.route("Momentum in futures", "")["route"], "insufficient_evidence")

    def test_closed_mechanism_overlap_names_the_hypothesis(self):
        r = self.route("Lead-lag between index futures", "We study the lead-lag relation between "
                       "S&P 500 and Nasdaq futures intraday trading, predicting returns from the leader "
                       "with minute data and examine profitability after transaction costs.")
        self.assertEqual(r["route"], "overlaps_closed_work")
        self.assertIn("es_nq_leadlag", [h["id"] for h in r["closed_overlaps"]])

    def test_order_book_paper_requires_unowned_data(self):
        r = self.route("Queue imbalance", "Using limit order book depth for E-mini futures we predict "
                       "the next intraday price move and evaluate trading profitability of the signal "
                       "net of transaction costs on stock index futures.")
        self.assertEqual((r["route"], r["missing_data"]), ("requires_data", ["NQ: order_book_l2", "ES: order_book_l2"]))

    def test_unowned_market_and_multiday_and_candidate(self):
        oil = self.route("Crude oil futures intraday reversal", "We document intraday reversal in crude "
                         "oil futures trading and test a trading strategy that predicts returns using "
                         "minute bars with transaction costs and out-of-sample evaluation.")
        self.assertEqual(oil["route"], "requires_data")
        monthly = self.route("Monthly rebalanced index ETF strategy", "We test a monthly rebalanced "
                             "trading strategy on S&P 500 ETFs and stock index returns that predicts "
                             "excess returns across a long sample of investors' portfolios.")
        self.assertEqual(monthly["route"], "knowledge_only")
        good = self.route("Intraday volatility-scaled reversal in equity index futures", "We document "
                          "an intraday mean-reversion signal in S&P 500 E-mini futures after large "
                          "volatility shocks, predicting returns over the next hour; the strategy is "
                          "profitable out-of-sample after transaction costs.")
        self.assertEqual(good["route"], "candidate_for_review")


class ProfileTests(unittest.TestCase):
    def test_bundled_profile_is_valid_and_draft_until_approved(self):
        prof = research_profile.load()
        self.assertIn(research_profile.status(prof)["state"], ("draft", "approved", "stale"))

    def test_approval_goes_stale_when_a_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "notes.md"
            src.write_text("notes v1", encoding="utf-8")
            prof = json.loads(Path(config.PROFILE_PATH).read_text(encoding="utf-8"))
            prof["source_documents"] = {"research_notes": str(src)}
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(prof), encoding="utf-8")
            research_profile.approve(path)
            self.assertEqual(research_profile.status(research_profile.load(path))["state"], "approved")
            src.write_text("notes v2", encoding="utf-8")
            info = research_profile.status(research_profile.load(path))
            self.assertEqual((info["state"], info["changed_sources"]), ("stale", ["research_notes"]))

    def test_mechanism_without_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prof = json.loads(Path(config.PROFILE_PATH).read_text(encoding="utf-8"))
            prof["closed_mechanisms"][0].pop("source")
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(prof), encoding="utf-8")
            with self.assertRaises(research_profile.ProfileError):
                research_profile.load(path)


class OptionsLaneTests(unittest.TestCase):
    """Options lane: input cadence, underlying
    and prediction horizon are separate questions."""
    @classmethod
    def setUpClass(cls):
        cls.profile = research_profile.load()

    def lane(self, title, abstract):
        return routing.route(paper(title, abstract), self.profile)

    def test_spx_skew_is_not_qqq_testable(self):
        r = self.lane("SPX option skew and futures", "Daily S&P 500 index options skew measured at the close "
                      "predicts next-day S&P 500 futures returns; a trading strategy based on the signal "
                      "earns excess returns after transaction costs over twenty years.")
        self.assertEqual(r["options_lane"]["underlying"], "SPX")
        self.assertEqual(r["options_lane"]["compatibility"], "requires_data")
        self.assertIn("proposed adaptation", r["options_lane"]["reasons"][0])
        self.assertNotEqual(r["route"], "candidate_for_review")

    def test_prior_day_qqq_skew_predicting_intraday_is_compatible(self):
        r = self.lane("QQQ skew and Nasdaq intraday returns", "Prior-day QQQ implied volatility skew "
                      "predicts next-day intraday returns of Nasdaq futures; the signal is observable "
                      "at the previous close and the strategy is profitable after transaction costs.")
        lane = r["options_lane"]
        self.assertEqual((lane["underlying"], lane["input_cadence"], lane["compatibility"]),
                         ("QQQ", "daily", "potentially_compatible"))
        self.assertIn("usable D+1", lane["reasons"][0])
        self.assertTrue(any("not validated" in x for x in lane["reasons"]))

    def test_daily_signed_flow_still_needs_flow_data(self):
        r = self.lane("Options order flow", "Daily QQQ options order flow imbalance, signed by customer "
                      "buys and sells, predicts next-day Nasdaq futures returns and supports a trading "
                      "strategy with excess returns net of transaction costs.")
        self.assertEqual(r["options_lane"]["compatibility"], "requires_data")
        self.assertIn("NQ: options_signed_flow", r["missing_data"])

    def test_unknown_underlying_stays_unknown(self):
        r = self.lane("Put-call ratios", "Daily put-call open interest ratios measured at the close predict "
                      "next-day equity index futures returns, and a trading strategy on the signal "
                      "earns excess returns after costs over a long sample.")
        self.assertEqual(r["options_lane"]["compatibility"], "unknown")

    def test_non_options_paper_has_no_lane(self):
        self.assertIsNone(self.lane("Futures momentum", "We document momentum in S&P 500 futures returns "
                          "and test a trading strategy that predicts intraday returns net of transaction "
                          "costs across two decades of minute data.")["options_lane"])


class FeasibilityTests(unittest.TestCase):
    """Feasibility comes from profile datasets,
    per instrument; one market's data never satisfies another's."""
    TICK = ("Aggressor flow in Nasdaq and Bitcoin", "Using aggressor-signed transaction-level trades we "
            "predict intraday returns of Nasdaq futures and Bitcoin perpetual futures; the trading "
            "strategy is profitable out-of-sample after transaction costs. The sample spans many years across several distinct market regimes.")
    OHLCV = ("Nasdaq intraday reversal", "Using one-minute bars we document an intraday mean-reversion "
             "signal in Nasdaq futures that predicts returns over the next hour and remains profitable "
             "after transaction costs out-of-sample.")

    @classmethod
    def setUpClass(cls):
        cls.profile = research_profile.load()

    def without_owned_ticks(self):
        # The example profile owns NQ/ES ticks; these tests check that other-market
        # or sample-only ticks never satisfy NQ, so they drop the owned set first.
        return dict(self.profile, datasets=[d for d in self.profile["datasets"] if d["id"] != "nq_es_ticks"])

    def test_crypto_ticks_do_not_satisfy_nq(self):
        r = routing.route(paper(*self.TICK), self.without_owned_ticks())
        self.assertEqual(r["feasibility"]["NQ"]["requirements"]["trades_tick"], "missing")
        self.assertEqual(r["feasibility"]["BTCUSDT_PERP"]["status"], "feasible")
        self.assertIn("NQ: trades_tick", r["missing_data"])
        self.assertTrue(any("not feasible on NQ" in x for x in r["reasons"]))

    def test_nq_trade_sample_is_not_research_usable(self):
        tick_only_nq = paper("Aggressor flow in Nasdaq futures", self.TICK[1].replace(" and Bitcoin perpetual futures", ""))
        r = routing.route(tick_only_nq, self.without_owned_ticks())
        self.assertEqual(r["route"], "requires_data")

    def test_owned_ticks_make_nq_tick_papers_feasible(self):
        tick_only_nq = paper("Aggressor flow in Nasdaq futures", self.TICK[1].replace(" and Bitcoin perpetual futures", ""))
        r = routing.route(tick_only_nq, self.profile)
        self.assertEqual(r["feasibility"]["NQ"]["requirements"]["trades_tick"], "owned")
        self.assertEqual(r["route"], "candidate_for_review")

    def test_unvalidated_cl_ticks_are_not_research_usable(self):
        r = routing.route(paper("Crude oil order flow", "Using aggressor-signed transaction-level trades in crude "
                                "oil futures we predict intraday returns; the trading strategy is profitable "
                                "out-of-sample after transaction costs over a long multi-year sample."), self.profile)
        self.assertEqual(r["feasibility"]["CL"]["status"], "requires_data")
        self.assertEqual(r["route"], "requires_data")


    def test_empty_inventory_is_not_a_candidate(self):
        empty = dict(self.profile, datasets=[])
        r = routing.route(paper(*self.OHLCV), empty)
        self.assertEqual(r["route"], "requires_data")
        self.assertIn("NQ: ohlcv", r["missing_data"])

    def test_closed_work_on_other_market_does_not_reroute(self):
        r = routing.route(paper("Funding rates in equity index futures", "We study how funding rate "
                          "conditions predict intraday returns of S&P 500 futures, and a trading strategy "
                          "earns excess returns after transaction costs out-of-sample. The sample spans many years across several distinct market regimes."), self.profile)
        hit = [h for h in r["closed_overlaps"] if h["id"] == "btc_funding"]
        self.assertTrue(hit and not hit[0]["same_market"])
        self.assertNotEqual(r["route"], "overlaps_closed_work")


class ScreenTests(unittest.TestCase):
    """Trade-frequency class and reported bps are flagged against the profile
    screen; flags annotate candidates, they never change a route."""
    @classmethod
    def setUpClass(cls):
        cls.profile = research_profile.load()

    def test_frequency_classes(self):
        s = lambda t: routing.screen(t, self.profile)["trade_frequency"]
        self.assertEqual(s("Order flow imbalance over 5-minute windows predicts returns"), "multi_per_day")
        self.assertEqual(s("The first half-hour return predicts the last half-hour return"), "at_most_daily")
        self.assertEqual(s("Order flow on announcement days predicts next-day returns"), "at_most_daily")
        self.assertEqual(s("Monthly rebalanced portfolios"), "at_most_daily")
        self.assertEqual(s("We study equity index futures"), "unclear")

    def test_flags_low_frequency_and_low_bps(self):
        r = routing.screen("A daily signal earning 1.5 bps per trade", self.profile)
        self.assertEqual(r["reported_bps"], [1.5])
        self.assertTrue(any(f.startswith("low_frequency") for f in r["flags"]))
        self.assertTrue(any("below 2.7" in f for f in r["flags"]))
        self.assertEqual(routing.screen("Earns 4 basis points every 5-minute window", self.profile)["flags"], [])

    def test_no_screen_block_means_no_flags(self):
        bare = {k: v for k, v in self.profile.items() if k != "screen"}
        self.assertEqual(routing.screen("A daily signal earning 1.5 bps", bare)["flags"], [])

    def test_flags_reach_candidate_reasons_without_rerouting(self):
        r = routing.route(paper("Intraday volatility-scaled reversal in equity index futures", "We document "
                          "an intraday mean-reversion signal in S&P 500 E-mini futures after large volatility "
                          "shocks, predicting next-day returns; the strategy is profitable out-of-sample after "
                          "transaction costs."), self.profile)
        self.assertEqual(r["route"], "candidate_for_review")
        self.assertTrue(any(x.startswith("low_frequency") for x in r["reasons"]))

import evidence


class EvidenceTests(OfflineCase):
    SOURCE = ("[p. 3]\nWe trade E-mini futures. Transaction costs of 1 tick per side are deducted.\n"
              "The out-of-sample period 2019-2022 yields a Sharpe ratio of 0.8.")

    def extraction(self, **over):
        base = {"mechanism": {"claim": "c", "quote": "We trade E-mini futures."},
                "markets": [], "data": {"fields_used": [], "resolution": "", "sample_period": "", "quote": ""},
                "holding_horizon": {"value": "", "quote": ""},
                "out_of_sample": {"status": "supported", "quote": "The out-of-sample period 2019-2022 yields a Sharpe ratio of 0.8."},
                "costs": {"status": "supported", "quote": "Transaction costs of 1 tick per side are deducted."},
                "reported_results": [{"metric": "Sharpe", "value": "0.8", "units": "", "quote": "yields an information ratio of 19.4 overall"}],
                "limitations": []}
        base.update(over)
        return base

    def test_verified_quotes_pass_and_fabricated_quote_is_flagged(self):
        data = self.extraction()
        stats = evidence.verify(data, self.SOURCE)
        self.assertEqual((stats["verified"], stats["unverified"]), (3, ["reported_results[0]"]))
        self.assertEqual(data["out_of_sample"]["status"], "supported")

    def test_supported_needs_quote_that_names_the_claim(self):
        data = self.extraction(out_of_sample={"status": "supported",
                                              "quote": "Transaction costs of 1 tick per side are deducted."})
        evidence.verify(data, self.SOURCE)
        self.assertEqual(data["out_of_sample"]["status"], "uncertain")

    def test_supported_with_unfound_quote_is_downgraded(self):
        data = self.extraction(costs={"status": "supported", "quote": "Costs are fully modelled with slippage."})
        evidence.verify(data, self.SOURCE)
        self.assertEqual(data["costs"]["status"], "uncertain")

    def test_text_selection_tags_pages_and_respects_budget(self):
        pages = ["intro " * 50, "unrelated " * 50, "results and transaction costs out-of-sample " * 20]
        text, used = evidence.select_text({"abstract_full": "An abstract."}, pages)
        self.assertEqual(used, [1, 2, 3])
        self.assertIn("[p. 3]", text)
        big = ["x" * 5000] * 20
        text, used = evidence.select_text({}, big)
        self.assertLessEqual(len(text), evidence.MAX_TEXT_CHARS + 20 * 10)

    def test_failed_extraction_gives_failed_card_and_request_uses_schema(self):
        with patch.object(evidence.requests, 'post', side_effect=requests.exceptions.ConnectionError()):
            self.assertIn("error", evidence.extract("t", "text"))
        ok = Mock()
        ok.json.return_value = {"message": {"content": json.dumps(self.extraction())}}
        ok.raise_for_status = Mock()
        with patch.object(evidence.requests, 'post', return_value=ok) as post:
            data = evidence.extract("t", "text")
        body = post.call_args.kwargs["json"]
        self.assertEqual((body["format"], body["options"]["num_ctx"], body["options"]["temperature"]),
                         (evidence.SCHEMA, evidence.NUM_CTX, 0))
        card = evidence.build_card({"title": "t", "routing": {"route": "candidate_for_review"}},
                                   {"error": "x"}, "", [])
        self.assertEqual(card["status"], "failed")
        self.assertNotIn("error", data)


import evaluate


class EvaluateTests(unittest.TestCase):
    def test_score_reports_agreement_and_candidate_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=evaluate.LABEL_COLUMNS)
                w.writeheader()
                w.writerow({"item_id": 1, "title": "Icebergs", "human_route": "irrelevant",
                            "abstract": "We map icebergs in sea ice imagery with unsupervised learning across "
                                        "twenty Antarctic seasons and validate the detected populations against independent "
                                        "ship-based field surveys collected by three expeditions."})
                w.writerow({"item_id": 2, "title": "Unlabelled", "human_route": "", "abstract": "x"})
            result = evaluate.score(path)
        self.assertEqual((result["labelled"], result["agreement"]), (1, 1.0))

    def test_unknown_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("item_id,title,abstract,human_route\n1,t,a,maybe\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                evaluate.score(path)


class EvidencePolarityTests(unittest.TestCase):
    """Evidence-card quote verification."""
    def rigor(self, key, status, quote, source=None):
        data = {key: {"status": status, "quote": quote}}
        stats = evidence.verify(data, source or quote)
        return data[key], stats

    def test_negated_quotes_become_contradicted(self):
        data = {"costs": {"status": "supported", "quote": "Transaction costs were ignored."},
                "out_of_sample": {"status": "supported", "quote": "No out-of-sample testing was performed."}}
        stats = evidence.verify(data, "Transaction costs were ignored. No out-of-sample testing was performed.")
        self.assertEqual((data["costs"]["status"], data["out_of_sample"]["status"]), ("contradicted", "contradicted"))
        self.assertEqual(data["costs"]["model_status"], "supported")
        self.assertEqual(stats["verified"], 2)            # text presence and claim validity are separate
        self.assertEqual(sorted(stats["contradicted"]), ["costs", "out_of_sample"])

    def test_prior_literature_attribution_is_uncertain(self):
        item, _ = self.rigor("out_of_sample", "supported",
                             "Prior studies report out-of-sample predictability of index returns.")
        self.assertEqual((item["status"], item["claim_check"]), ("uncertain", "attributed_elsewhere"))

    def test_mixed_statement_is_uncertain(self):
        item, _ = self.rigor("costs", "supported",
                             "Returns are net of transaction costs for futures but ignore costs for options.")
        self.assertEqual((item["status"], item["claim_check"]), ("uncertain", "mixed"))

    def test_affirmative_quote_stays_supported(self):
        item, _ = self.rigor("costs", "supported", "All returns are reported net of transaction costs of one tick.")
        self.assertEqual((item["status"], item["claim_check"]), ("supported", "consistent"))

    def test_unsupported_contradiction_is_not_trusted(self):
        item, _ = self.rigor("costs", "contradicted", "We study index futures.", "Something else entirely here.")
        self.assertEqual(item["status"], "uncertain")

    def test_fabricated_number_with_real_unrelated_quote(self):
        source = "We use minute bars from 2010 to 2020. The strategy trades E-mini futures."
        data = {"reported_results": [{"metric": "Sharpe", "value": "2.4", "units": "",
                                      "quote": "The strategy trades E-mini futures."}]}
        stats = evidence.verify(data, source)
        self.assertEqual(data["reported_results"][0]["claim_check"], "value_not_in_quote")
        self.assertEqual((stats["verified"], stats["value_mismatch"]), (0, ["reported_results[0]"]))


class SchemaValidationTests(OfflineCase):
    def valid(self):
        return EvidenceTests.extraction(EvidenceTests())

    def test_valid_extraction_has_no_errors(self):
        self.assertEqual(evidence.validate_schema(self.valid(), evidence.SCHEMA), [])

    def test_malformed_fixtures_are_rejected(self):
        cases = {
            "wrong nested type": dict(self.valid(), costs="wrong type"),
            "null object": dict(self.valid(), out_of_sample=None),
            "bad enum": dict(self.valid(), costs={"status": "yes", "quote": ""}),
            "missing nested field": dict(self.valid(), data={"fields_used": [], "resolution": ""}),
            "array of wrong items": dict(self.valid(), markets=["NQ"]),
            "array field as string": dict(self.valid(), limitations="none"),
        }
        for name, data in cases.items():
            with self.subTest(name):
                self.assertTrue(evidence.validate_schema(data, evidence.SCHEMA))

    def test_extract_returns_error_instead_of_raising_later(self):
        ok = Mock()
        ok.json.return_value = {"message": {"content": json.dumps(dict(self.valid(), costs="wrong type"))}}
        ok.raise_for_status = Mock()
        with patch.object(evidence.requests, 'post', return_value=ok):
            result = evidence.extract("t", "text")
        self.assertIn("invalid extraction", result["error"])
        card = evidence.build_card({"title": "t", "routing": {}}, result, "text", [])
        self.assertEqual(card["status"], "failed")


class ApprovalBindingTests(unittest.TestCase):
    """Approval is bound to the profile content and its source documents."""
    def make(self, tmp, source_exists=True):
        src = Path(tmp) / "notes.md"
        if source_exists:
            src.write_text("notes v1", encoding="utf-8")
        prof = json.loads(Path(config.PROFILE_PATH).read_text(encoding="utf-8"))
        for key in ("reviewed_source_hashes", "approved_content_sha256", "approved_at"):
            prof.pop(key, None)
        prof["review_status"] = "draft"
        prof["source_documents"] = {"research_notes": str(src)}
        path = Path(tmp) / "profile.json"
        path.write_text(json.dumps(prof), encoding="utf-8")
        return path

    def test_missing_source_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make(tmp, source_exists=False)
            with self.assertRaises(research_profile.ProfileError):
                research_profile.approve(path)
            self.assertEqual(research_profile.status(research_profile.load(path))["state"], "draft")

    def test_policy_edit_after_approval_goes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make(tmp)
            research_profile.approve(path)
            info = research_profile.status(research_profile.load(path))
            self.assertEqual((info["state"], info["provisional"]), ("approved", False))
            prof = json.loads(path.read_text(encoding="utf-8"))
            prof["sample_locks"] = {}
            path.write_text(json.dumps(prof), encoding="utf-8")
            info = research_profile.status(research_profile.load(path))
            self.assertEqual((info["state"], info["content_changed_since_approval"], info["provisional"]),
                             ("stale", True, True))

    def test_review_metadata_edits_do_not_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make(tmp)
            research_profile.approve(path)
            prof = json.loads(path.read_text(encoding="utf-8"))
            prof["review_note"] = "reviewed by a human"
            path.write_text(json.dumps(prof), encoding="utf-8")
            self.assertEqual(research_profile.status(research_profile.load(path))["state"], "approved")


class QueryProfileAndCardLaneTests(MainTests):
    OPTIONS_ABSTRACT = ("Prior-day QQQ implied volatility skew predicts next-day intraday returns of Nasdaq "
                        "futures; the signal is observable at the previous close and the strategy is "
                        "profitable after transaction costs across many years of data.")

    def test_query_profile_changes_queries_not_scoring(self):
        seen = []
        search = Mock(side_effect=lambda q, limit: seen.append(q) or SearchResult(pages=1))
        with patch.dict(paper_scout.SEARCH_FUNCS, {'openalex': search}), \
             patch.object(paper_scout, 'parse_paper', wraps=paper_scout.parse_paper) as parse, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = paper_scout.main(['--sources', 'openalex', '--query-profile', 'options_eod'])
        self.assertEqual((rc, seen), (0, config.QUERY_PROFILES['options_eod']))
        manifest = json.loads((sorted((self.path / 'runs').iterdir())[-1] / 'manifest.json').read_text())
        self.assertEqual((manifest['arguments']['profile'], manifest['arguments']['query_profile']),
                         ('general', 'options_eod'))

    def test_existing_query_sets_unchanged(self):
        self.assertEqual(len(config.QUERY_PROFILES['general']), 5)
        self.assertEqual(config.QUERY_PROFILES['microstructure'][0], 'iceberg order detection')

    def test_cards_from_options_lane_keeps_routes(self):
        rows = [raw('QQQ skew and Nasdaq intraday returns', 'q1', abstract=self.OPTIONS_ABSTRACT),
                raw('Unrelated futures note', 'q2')]
        card = {"status": "complete", "verification": {}}
        with patch.object(paper_scout, 'fetch_fulltext_pdf', return_value=''), \
             patch.object(paper_scout.evidence, 'extract', return_value={"error": "offline"}) as extract:
            rc, run, _ = self.run_main(SearchResult(records=rows), '--select', 'route',
                                       '--evidence-cards', '--cards-from', 'options', '--draft-limit', '1')
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(extract.call_args.args[0], 'QQQ skew and Nasdaq intraday returns')
        saved = json.loads(next((run / 'cards').glob('*.json')).read_text(encoding='utf-8'))
        self.assertTrue(saved['project_context']['route_provisional'])
        self.assertEqual(saved['project_context']['options_lane']['compatibility'], 'potentially_compatible')
        papers = {p['title']: p for p in json.loads((run / 'papers.json').read_text(encoding='utf-8'))}
        self.assertEqual(papers['QQQ skew and Nasdaq intraday returns']['route'],
                         saved['project_context']['route'])
