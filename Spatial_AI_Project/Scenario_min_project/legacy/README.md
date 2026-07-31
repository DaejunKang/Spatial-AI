# legacy — 구세대 코드 (공통 폴더, 참고 보존)

활성 파이프라인이 **import 하지 않는** 이전 세대 코드. 삭제하지 않고 참고용 보존. Task 라벨은 파일별로 아래 표에 명시(컨벤션은 `../docs/README.md`).

## 파일 → Task / 세대 / 대체
| 파일 | Task | 세대 | 대체(활성) |
|---|---|---|---|
| `tagger.py` | episode(Track1) | v0.7.1 | Track1=`tag_v08`, client pool=`common/client.py` |
| `vocab.py` · `prompts.py` | episode(Track1) | v0.7.1 | 어휘=`common/vocab073`, guided_json 스키마 |
| `window_tagger.py` · `event_tagger.py` | episode | v0.7.1 | 세그먼트=`classify073.consolidate_episodes` |
| `test_readout.py` · `batch_readout.py` | episode | v0.7.1 | 진입점 재작성 예정(활성 러너) |
| `norm_embed.py` | episode(Track2) | 임베딩 정규화 | guided decoding + `retrieve` 정규화 |
| `meta_tagging_vocab_v0.7.1.json` · `meta_tagging_segment_schema_v0.7.1.json` | episode | v0.7.1 | v0.7.3/taxonomy |

## 주의
- import는 `run.sh` PYTHONPATH/venv `.pth`로 유지되나, **신규 작업은 활성(common/task_*)에서** 한다.
- `tagger.build_client_pool` 은 `common/client.py`로 추출됨 — 활성 러너는 `from client import build_client_pool` 사용.
