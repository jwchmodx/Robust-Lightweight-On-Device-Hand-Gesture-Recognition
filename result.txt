## baseline_static_conf05_gru
---
processed_baseline_static_conf05
Label                   Samples  Avg Frames   Min F   Max F  Avg Detected   Avg Ratio
No gesture                  533       36.01      23      42          1.42      0.0400
Doing other things         1468       36.05      20      49         17.21      0.4761
Swiping Up                  508       36.01      24      40         12.14      0.3362
Swiping Down                520       36.04      24      39         11.64      0.3231
Swiping Left                494       36.02      23      39         10.58      0.2936
Swiping Right               486       35.98      20      39          9.28      0.2568
Stop Sign                   536       36.00      24      40         23.29      0.6468
---
=== Finished ===
Best validation accuracy: 0.8059

=== Validation Accuracy by Class ===
 0 | No gesture           |   469/533   | 0.8799
 1 | Doing other things   |  1132/1468  | 0.7711
 2 | Swiping Up           |   422/508   | 0.8307
 3 | Swiping Down         |   435/520   | 0.8365
 4 | Swiping Left         |   374/494   | 0.7571
 5 | Swiping Right        |   331/486   | 0.6811
 6 | Stop Sign            |   500/536   | 0.9328

=== Confusion Matrix ===
rows = true label, columns = predicted label
true\pred                   0      1      2      3      4      5      6
0 No gesture            469     57      3      1      0      3      0
1 Doing other things    254   1132     27     11     15     18     11
2 Swiping Up             38     37    422      5      2      3      1
3 Swiping Down           41     21      4    435      3      2     14
4 Swiping Left           65     33      2      2    374     17      1
5 Swiping Right          90     32      1      6     25    331      1
6 Stop Sign              19     10      0      7      0      0    500

=== Class Index ===
0: No gesture
1: Doing other things
2: Swiping Up
3: Swiping Down
4: Swiping Left
5: Swiping Right
6: Stop Sign
---
문제점: mediapipe 전처리 과정에서 특정 제스처(Swiping Left, Right)의 데이터에서 평균적인 손 검출 프레임 비율이 35퍼센트 이하로 매우 낮음 -> 랜드마크 자체가 동영상에서 많이 남지 않아 No gesture 또는 Doing other things로 많이 분류하게 된다 
가능성 두가지
1. 특정 동작 영상의 절대적인 프레임 수가 부족한 경우
2. 원본 프레임은 충분한데 손을 못찾음
--- 
src/test/check_frame_and_detection_by_class.py
Label                   Samples  Avg Frames   Min F   Max F  Avg Detected   Avg Ratio
No gesture                  326       36.00      23      42          3.33      0.0924
Doing other things          887       36.08      23      49         19.48      0.5384
Swiping Up                  305       36.01      24      40         13.99      0.3880
Swiping Down                294       36.07      24      38         13.06      0.3627
Swiping Left                305       35.99      28      38         12.43      0.3448
Swiping Right               296       35.96      20      39         11.43      0.3173
Stop Sign                   314       35.99      28      40         24.05      0.6682
2번의 경우로 원본 프레임은 충분한것을 확인 -> Detected 되는 프레임 수를 늘려야한다.
---
min_detection_confidence=0.3 
min_tracking_confidence=0.3
resize_scale=2.0
다음과 같은 설정으로 다시 데이터셋 생성
---
Label                   Samples  Avg Frames   Min F   Max F  Avg Detected   Avg Ratio
No gesture                  533       36.01      23      42          3.17      0.0879
Doing other things         1468       36.05      20      49         19.25      0.5329
Swiping Up                  508       36.01      24      40         14.19      0.3935
Swiping Down                520       36.04      24      39         13.51      0.3756
Swiping Left                494       36.02      23      39         12.53      0.3477
Swiping Right               486       35.98      20      39         11.12      0.3082
Stop Sign                   536       36.00      24      40         24.03      0.6672
유의미한 향상폭x
---
원본 프레임 자체는 비슷하지만 특정동작은 실제로 손이 등장하는 비율이 적다. -> 동작 구간만 잘라서 학습할 필요성 존재 
실제 검출된 동작 구간 (check_detection_segments.py)
=== Swiping Left ===
3925     det= 2/38 span=2  longest=2  ·················■■···················
28701    det= 8/37 span=11 longest=7  ········■···■■■■■■■··················
85642    det=11/37 span=19 longest=7  ················■··■■■······■■■■■■■··
105115   det=18/33 span=23 longest=9  ■■■■■■■■■·····■■■■■■■■■··········
28247    det=35/37 span=35 longest=35 ··■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

=== Swiping Right ===
50576    det=26/37 span=32 longest=11 ·····■■■■■■■·■■■■■■■■■■■·■■··■··■■■■■
97976    det= 1/36 span=1  longest=1  ························■···········
83953    det=10/38 span=11 longest=6  ·············■■■■·■■■■■■··············
39324    det= 0/36 span=0  longest=0  ····································
143849   det= 8/36 span=10 longest=6  ···············■■■■■■··■■···········

=== Stop Sign ===
92921    det=16/37 span=16 longest=16 ··············■■■■■■■■■■■■■■■■·······
24023    det=17/37 span=17 longest=17 ················■■■■■■■■■■■■■■■■■····
117876   det=28/36 span=28 longest=28 ········■■■■■■■■■■■■■■■■■■■■■■■■■■■■
234      det=35/37 span=37 longest=21 ■■■■■■■■·■■■■■■·■■■■■■■■■■■■■■■■■■■■■
55845    det=12/32 span=12 longest=12 ■■■■■■■■■■■■····················
---
샘플 품질의 편차가 매우 큰 것을 확인, 동작 샘플 자체에서 끊기는 경우 + 검출된 프레임 자체가 별로 없는 경우가 있다
해결방안
1. 쓸 수 있는 샘플만 크롭 or 보간
2. 인식이 거의 불가능한 샘플은 학습에서 제외
해결방안의 문제점
1. validation 데이터셋을 정제하게 되면 실제 성능이 숨겨짐
-> 랜드마크 추출 실패가 성능의 병목이므로 유효 landmark가 확보된 샘플에서의 정확도와 전체 데이터셋 기준 정확도를 분리하여 평가해보자
(최종 데모에서는 최대한 사용자의 손을 크게 보이게끔 하고 지정된 영역 내에서 제스처를 하도록 제한하는게 현실적)
---
전체 클래스중 usable한 데이터 비율 계산
check_usable_samples_by_class.py
Doing other things        1468     1120      0.7629     19.25      0.5329     16.40
Swiping Up                 508      397      0.7815     14.19      0.3935     10.58
Swiping Down               520      390      0.7500     13.51      0.3756     11.07
Swiping Left               494      327      0.6619     12.53      0.3477      8.26
Swiping Right              486      287      0.5905     11.12      0.3082      7.45
Stop Sign                  536      514      0.9590     24.03      0.6672     22.90
Swiping Right의 경우 40퍼센트의 손실 발생
---
프로젝트 방향성 검토
프로젝트 목표가 실제 제어프로그램을 지향한다면, 모델 하나만 사용하여 제스처를 인식하는 것이 아니라 단계적으로 상태머신을 통해 감지하는 것이 합당한 방식
그러므로 
IDLE
  ↓ 손이 일정 시간 이상 검출됨
ARMED / READY
  ↓ 움직임 시작 감지
RECORDING
  ↓ 동작 종료 또는 최대 프레임 도달
CLASSIFYING
  ↓ confidence 충분함
EXECUTE COMMAND
  ↓ 잠깐 재입력 방지
COOLDOWN
  ↓
IDLE 또는 ARMED
다음과 같은 상태 머신을 도입
---
가정
제어 모드 진입 조건: 최근 5프레임중 4프레임 이상 손 검출
분류 클래스: Doing other things, Swiping Up, Swiping Down, Stop Sign
(No gesture의 경우 분류기 학습에서는 제외, 오작동 검증 데이터로 유지한다. 나머지 클래스는 1. usable 조건 통과 샘플 사용 2. 첫 검출 ~ 마지막 검출 구간 crop 3. 앞뒤 margin 2프레임 추가 4. 24프레임으로 리샘플링)
실시간 분류 조건: 제스처 confidence >= 0.85, cooldown 적용, 기존의 명령이 유지된 것인지
---
# 손이 이미 인식된 조건 하에서의 학습 과정
exp_name: armed4
기존 processed_baseline_static_conf05을 사용하여 data/model_ready/armed4_24f에 새로운 데이터셋 생성
---
=== Finished ===
Best epoch: 30
Best validation accuracy: 0.9456
Model saved to: models/gru_armed4_24f_best.pt
History saved to: outputs/gru_armed4_24f_history.json
Metrics saved to: outputs/gru_armed4_24f_metrics.json

=== Validation Accuracy by Class ===
 0 | Doing other things     |   970/1000  | 0.9700
 1 | Swiping Up             |   370/402   | 0.9204
 2 | Swiping Down           |   379/421   | 0.9002
 3 | Stop Sign              |   488/511   | 0.9550

=== Confusion Matrix ===
rows = true label, columns = predicted label
true\pred                        0       1       2       3
0 Doing other things         970       9      11      10
1 Swiping Up                  27     370       3       2
2 Swiping Down                17       3     379      22
3 Stop Sign                    8       0      15     488
---
에포크: 50
=== Finished ===
Best epoch: 37
Best validation accuracy: 0.9490
Model saved to: models/gru_armed4_24f_best.pt
History saved to: outputs/gru_armed4_24f_history.json
Metrics saved to: outputs/gru_armed4_24f_metrics.json

=== Validation Accuracy by Class ===
 0 | Doing other things     |   971/1000  | 0.9710
 1 | Swiping Up             |   373/402   | 0.9279
 2 | Swiping Down           |   378/421   | 0.8979
 3 | Stop Sign              |   493/511   | 0.9648

=== Confusion Matrix ===
rows = true label, columns = predicted label
true\pred                        0       1       2       3
0 Doing other things         971       7      13       9
1 Swiping Up                  24     373       3       2
2 Swiping Down                12       3     378      28
3 Stop Sign                    5       0      13     493
---
