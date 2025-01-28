# word2vec
> Vanilla Skip-Gram 방식의 Word2Vec 모델을 구현하고, 한국어 영화리뷰 데이터셋인 nsmc 데이터셋으로 학습 및 검증하는 코드를 담고 있음

## Project Structure
### `utils` directory
- `mps_support.py` : mps acceleration 체크
### `dataset` directory
- `preprocess.py` : NSMC 데이터셋 다운로드 / konlpy okt 토크나이저로 토크나이징 / Vocabulary 구축 / Skip-Gram 학습용 데이터셋 추출 (WordPair, IndexPair)
### `modeling` directory
- `word2vec.py` : VanillaSkipGram 모델 클래스 구현
- `train.py` : 모델 학습 및 검증 코드


## Environment
- Apple Silicon Mac에서 mps acceleration 사용

## Disclaimer
> 본 리포지토리는 위키북스 저 [자연어 처리와 컴퓨터비전 심층학습] 도서의 내용을 참고함