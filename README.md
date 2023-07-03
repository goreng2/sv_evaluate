# 화자인증 Metric Graph 그리기
- Threshold 설정을 위한 긍/부정 분포도 출력
![Positive & Negative Distribution.png](Positive%20%26%20Negative%20Distribution.png)

- EER 측정을 위한 DET 곡선 출력
![Detection Error Tradeoff (DET) curves.png](Detection%20Error%20Tradeoff%20%28DET%29%20curves.png)

## Requirement
```bash
pip install -r requirements.txt
```

## Usage
### Docker build
```bash
docker build -t 이미지이름:태그 .
```

### Docker run
`이미지이름:태그` 수정 & 실행
```bash
vi launch.sh
# docker run -d \
#   -p 38060:80 \
#   -v $(pwd)/result:/workspace/result \
#   -e HOST_PWD=$(pwd) \
#   --name sv-graph \
#   이미지이름:태그
```
```bash
bash launch.sh
```

### Test
- Swagger UI(`http://0.0.0.0:80/docs`)에서 테스트 가능
- Port 변경: [Dockerfile](Dockerfile) `CMD` 수정 후 재빌드