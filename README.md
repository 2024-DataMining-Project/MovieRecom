# MovieRecom

Movie Recommendation System: Contents Based &amp; Collaborative Filtering Hybrid 방식

## Collaborative Filtering

- recom_train.py: 모델 학습 코드
- recom_system.py: 학습된 모델을 통해 사용자가 rating하지 않은 영화를 추천하는 코드
- training 과정 - 시작 단계: loss 값 = 10.5883
  ![training 과정 - 시작 단계: loss 값 = 10.5883](./images/training2.png)
- training 과정 - 학습 종료: loss 값 = 0.0615
  ![training 과정 - 학습 종료: loss 값 = 0.0615](./images/training1.png)
- Recommendation 예시: rating 높게 예측한 상위 10개 영화 추천
  ![Recommendation 예시: rating 높게 예측한 상위 10개 영화 추천](./images/recom.png)
