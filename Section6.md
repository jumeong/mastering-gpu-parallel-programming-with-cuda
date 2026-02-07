# 36. Matrices addition using 2D of block and threads
- `ix = threadIdx.x + blockIdx.x * blockDim.x`
- `iy = threadIdx.y + blockIdx.y * blockDim.y`
- `idx = iy * nx + ix`으로 표현 가능
<img width="1363" height="772" alt="image" src="https://github.com/user-attachments/assets/db37a2bc-1d10-457e-9a76-4a85f64d779a" />

# 37. Why L1 Hit-rate is zero?
