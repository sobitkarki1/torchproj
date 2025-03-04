import torch
import time

def measure_gflops(mat_size=15000, num_iters=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    A = torch.randn(mat_size, mat_size, device=device)
    B = torch.randn(mat_size, mat_size, device=device)

    # Warm-up
    C = torch.mm(A, B)
    torch.cuda.synchronize()

    # Measure time for multiple iterations
    start_time = time.time()
    for _ in range(num_iters):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = (end_time - start_time) / num_iters  # Average time per iteration
    flops_per_matrix_multiplication = 2 * (mat_size ** 3)  # 2 * N^3 FLOPs for matrix multiplication
    gflops = (flops_per_matrix_multiplication / elapsed_time) / 1e9  # Convert to GigaFLOPs

    print(f"Matrix size: {mat_size}x{mat_size}, GFLOPs: {gflops:.2f}")

if __name__ == "__main__":
    measure_gflops()
