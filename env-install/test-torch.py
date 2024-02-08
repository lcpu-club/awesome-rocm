print("Importing torch")
import torch
import time
print(f"Current torch.version.hip: {torch.version.hip}")
print(f"Device name: {torch.cuda.get_device_name()}")

precision = input("Type the precision: e.g. torch.float16")
torch.set_default_dtype(eval(precision))

def test_vector_add(n: int):
        print("Creating tensor")
        a = torch.arange(0, n, 1)
        b = torch.arange(0, n, 1)
        print("Moving tensors to GPU")
        a = a.cuda()
        b = b.cuda()
        print(a, b)
        print("Calculating a+b")
        c = a+b
        print(c)

def test_matmul(n: int, m: int, k: int):
        print("Creating tensor")
        a = torch.rand((n, k), device="cuda")
        b = torch.rand((k, m), device="cuda")
        print(a.dtype)
        print("Calculating AB")
        start_time = time.time()
        c = a @ b
        torch.cuda.synchronize()
        end_time = time.time()
        time_usage = end_time - start_time
        flops = n*m*k*2 / time_usage
        print(c.sum())
        print(f"Average TFlops = {flops/1e12:.2f}")

for n in range(1, 100):
        print(f"vector add with n = {n}")
        test_vector_add(n)

for n in [2**k for k in range(0, 16)]:
        print(f"matmul with n, m = 16384, k = {n}")
        test_matmul(16384, 16384, n)
