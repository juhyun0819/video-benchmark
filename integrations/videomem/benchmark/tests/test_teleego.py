from teleego import parse_srt

def test_srt_order_and_invalid():
    rows, issues = parse_srt('''1
00:00:10,000 --> 00:00:12,500
阅读

2
00:00:01,000 --> 00:00:02,000
学习

3
00:00:05,000 --> 00:00:04,000
invalid
''')
    assert [r['source_start'] for r in rows] == [1,10]
    assert rows[1]['source_end'] == 12.5
    assert {i['type'] for i in issues} == {'non_chronological_srt_order','invalid_interval_or_empty_text'}
