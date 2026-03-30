try:
    from .model import SalaryPredictor
except ImportError:
    from model import SalaryPredictor


p = SalaryPredictor()
print("SUCCESS: Model loaded!")
print("Top jobs:", p.top_jobs[:3])
