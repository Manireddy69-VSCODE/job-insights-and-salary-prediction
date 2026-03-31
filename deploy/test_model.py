from src.predict import SalaryPredictor


p = SalaryPredictor()
print("SUCCESS: Model loaded!")
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
