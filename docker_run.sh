docker run -d \
  -p 38060:80 \
  -v $(pwd)/result:/workspace/result \
  -e HOST_PWD=$(pwd) \
  --name sv-graph \
  sv-graph:v0.4
