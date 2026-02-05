# 34. Performance analysis
## Block Size 32 vs 64
- Wave: SM에서 한번에 실행할 수 있는 블록 수로 전체 블록 수를 나누면 여러 번의 웨이브가 발생
- Block Size를 32로 하면 Warp 수가 적어서 dependency가 있을 때, 멍 때리고 있는 warp 발생
