try:
    from .model import SalaryPredictor
except ImportError:
    from model import SalaryPredictor


p = SalaryPredictor()
print("SUCCESS: Model loaded!")
print("Top jobs:", p.top_jobs[:3])
print(
    "Sample prediction:",
    p.predict(
        {
            "min_exp": 2,
            "max_exp": 5,
            "job_title": "Data Scientist",
            "location": "Bangalore",
        }
    ),
)
print(
    "Legacy payload prediction:",
    p.predict(
        {
            "min_exp": 2,
            "max_exp": 5,
            "posted_days": 30,
            "job_title": "Data Scientist",
            "location": "Bangalore",
        }
    ),
)
